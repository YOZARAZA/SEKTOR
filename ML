# coding: utf-8

# === UTILS ===
import os
import sys
import math
import datetime
import warnings
import logging
import random
import numpy as np
from collections import defaultdict, deque, Counter
from scipy.stats import entropy
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TBB_NUM_THREADS'] = '2'
warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.calibration import CalibratedClassifierCV
    from catboost import CatBoostClassifier
except ImportError as e:
    missing_lib = str(e).split("'")[-2]
    logging.error(f"Не удалось импортировать {missing_lib}. Установите: pip install {missing_lib}")
    raise

try:
    import sklearn_crfsuite
    from sklearn_crfsuite import CRF as SKCRF
    from sklearn_crfsuite.metrics import flat_classification_report
except ImportError as e:
    logging.error("Не удалось импортировать sklearn-crfsuite. Установите: pip install sklearn-crfsuite")
    raise

try:
    from numba import njit
except ModuleNotFoundError as e:
    logging.error("Не удалось импортировать numba. Установите: pip install numba")
    def njit(func):
        return func
    raise

logging.basicConfig(
    filename='prediction_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

yellow_color = '\033[93m'
gray_color   = '\033[90m'
reset_color  = '\033[0m'
green_color  = '\033[92m'
blue_color   = '\033[94m'

NUMBER_OF_SECTORS = 39
NEIGHBOR_RANGE = 2
MAX_HMM_SEQUENCE_LENGTH = 1000
BAMF_ACCURACY_WINDOW = 100
BAMF_TOLERANCE = 1
BAMF_MAX_HISTORY = 3000
ALPHA_PRIOR = 0.2
DECAY_FACTOR = 0.995
min_data_threshold = 3
TRAIN_INTERVAL = 10
MIN_AGGR_TRAIN_SIZE = 10
LR_WEIGHT = 0.8
MEM_WEIGHT = 0.2

sectors = []
for i in range(1, 20):
    sectors.append(str(i))
    sectors.append(f"{i}/{i+1}")
sectors.append('20')

sector_label_to_index = {label: index for index, label in enumerate(sectors)}
sector_index_to_label = {index: label for index, label in enumerate(sectors)}

@lru_cache(maxsize=1000)
def get_sector_label(sector_index):
    if 0 <= sector_index < len(sectors):
        return sectors[sector_index]
    else:
        return None

def circular_distance(a, b, total_sectors):
    return min(abs(a - b), total_sectors - abs(a - b))

def smooth_circular_probs(probs, total_sectors=39, sigma=0.5):
    indices = np.arange(total_sectors)
    i_idx, j_idx = np.meshgrid(indices, indices, indexing='ij')
    dist = np.abs(i_idx - j_idx)
    dist = np.minimum(dist, total_sectors - dist)
    weights = np.exp(-dist**2 / (2.0 * sigma**2))
    smoothed = weights @ probs
    sum_sm = smoothed.sum() + 1e-9
    return smoothed / sum_sm

# === DATA_PREPROCESSOR ===
def replace_with_last_known(X):
    X = X.copy()
    last_valid = 0.0
    for i in range(len(X)):
        if np.isnan(X[i]) or np.isinf(X[i]):
            X[i] = last_valid
            logging.debug(f"Заменил NaN/inf на {last_valid} в позиции {i}")
        else:
            last_valid = X[i]
    return X

def replace_with_mean(X):
    X = np.where(np.isinf(X), np.nan, X)
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    mask = np.isnan(X)
    if mask.any():
        logging.debug(f"Заменил NaN на средние в {mask.sum()} ячейках")
    X[mask] = col_means[np.where(mask)[1]]
    return X

def remove_highly_correlated_features(X, threshold=0.9):
    if X.shape[1] < 2:
        return X, list(range(X.shape[1]))
    corr_matrix = np.corrcoef(X, rowvar=False)
    n = corr_matrix.shape[0]
    to_remove = set()
    for i in range(n):
        for j in range(i+1, n):
            if abs(corr_matrix[i, j]) > threshold:
                to_remove.add(j)
    keep_indices = [i for i in range(n) if i not in to_remove]
    return X[:, keep_indices], keep_indices

def filter_outliers(data):
    if len(data) < 5:
        return data, np.arange(len(data), dtype=bool)
    iso = IsolationForest(contamination=0.1, random_state=42)
    preds = iso.fit_predict(data)
    mask = (preds == 1)
    return data[mask], mask

def validate_input(data_str: str) -> bool:
    if not data_str:
        logging.warning("Empty input string received.")
        return False
    direction = data_str[0].upper()
    if direction not in ('R', 'L'):
        logging.warning(f"Invalid direction: '{direction}' in '{data_str}'")
        return False
    sector_part = data_str[1:].strip()
    if not sector_part:
        logging.warning(f"No sector info in '{data_str}'")
        return False
    return True

def parse_result(data_str):
    try:
        data_str = data_str.strip()
    except AttributeError:
        logging.error("parse_result received non-string input.")
        return None
    if not validate_input(data_str):
        return None
    direction = data_str[0].upper()
    sector_str = data_str[1:].strip()
    if sector_str in sector_label_to_index:
        sector_index = sector_label_to_index[sector_str]
    else:
        logging.warning(f"Неверный сектор '{sector_str}' в '{data_str}'")
        return None
    return {
        'direction': direction,
        'sector_index': sector_index,
        'timestamp': datetime.datetime.now()
    }

def build_signed_step_features(sector_indices, directions, idx, previous_results, large_step_threshold, recent_steps_cache=None):
    short_window_size = 5
    long_window_size = 10
    ewma_alpha = 0.3
    feature = []
    sector_index = sector_indices[idx]
    direction = 1 if previous_results[idx]['direction'] == 'R' else -1
    prev_sector = sector_indices[idx-1] if idx > 0 else sector_index
    total = NUMBER_OF_SECTORS
    signed = ((sector_index - prev_sector + total/2) % total) - total/2
    signed_step = signed * direction
    feature.append(signed_step)
    step_norm = signed_step / (NUMBER_OF_SECTORS / 2)
    step_angle = step_norm * math.pi
    feature.extend([np.sin(step_angle), np.cos(step_angle)])
    if recent_steps_cache is None:
        indices = np.arange(max(0, idx-long_window_size+1), idx)
        if len(indices) > 0:
            curr_secs = sector_indices[indices]
            prev_secs = np.where(indices > 0, sector_indices[indices-1], curr_secs)
            curr_dirs = np.array([1 if previous_results[i]['direction'] == 'R' else -1 for i in indices])
            steps = circular_distance(curr_secs, prev_secs, NUMBER_OF_SECTORS)
            recent_steps = steps * curr_dirs
        else:
            recent_steps = np.array([])
    else:
        recent_steps = recent_steps_cache[max(0, idx-long_window_size+1):idx]
    if len(recent_steps) > 0:
        weights = (1 - ewma_alpha) ** np.arange(len(recent_steps)-1, -1, -1)
        ewma_step = np.sum(weights * recent_steps) / np.sum(weights)
    else:
        ewma_step = 0
    feature.append(ewma_step)
    short_steps = recent_steps[-short_window_size:] if len(recent_steps) >= short_window_size else recent_steps
    mean_step_short = np.mean(short_steps) if len(short_steps) > 0 else 0
    step_variance_short = np.var(short_steps) if len(short_steps) > 0 else 0
    max_step_short = np.max(np.abs(short_steps)) if len(short_steps) > 0 else 0
    large_step_ratio_short = np.mean(np.abs(short_steps) > large_step_threshold) if len(short_steps) > 0 else 0
    feature.extend([mean_step_short, step_variance_short, max_step_short, large_step_ratio_short])
    mean_step_long = np.mean(recent_steps) if len(recent_steps) > 0 else 0
    step_variance_long = np.var(recent_steps) if len(recent_steps) > 0 else 0
    max_step_long = np.max(np.abs(recent_steps)) if len(recent_steps) > 0 else 0
    large_step_ratio_long = np.mean(np.abs(recent_steps) > large_step_threshold) if len(recent_steps) > 0 else 0
    feature.extend([mean_step_long, step_variance_long, max_step_long, large_step_ratio_long])
    return np.array(feature)

def build_features_for_spin(i, sector_indices, directions, previous_results, large_step_threshold, recent_steps_cache, max_order):
    feature = []
    for j in range(i - max_order, i):
        feature.extend(build_signed_step_features(
            sector_indices, directions, j, previous_results, large_step_threshold, recent_steps_cache
        ))
        direction_code = directions[j]
        feature.append(direction_code)
        sector_sin = math.sin(2.0 * math.pi * sector_indices[j] / NUMBER_OF_SECTORS)
        sector_cos = math.cos(2.0 * math.pi * sector_indices[j] / NUMBER_OF_SECTORS)
        feature.extend([sector_sin, sector_cos])
        prev_sector = sector_indices[j-1] if j > 0 else sector_indices[j]
        relative_cyclic_distance = circular_distance(sector_indices[j], prev_sector, NUMBER_OF_SECTORS)
        feature.append(relative_cyclic_distance)
        close_far_indicator = 1 if circular_distance(sector_indices[j], prev_sector, NUMBER_OF_SECTORS) <= 2 else 0
        feature.append(close_far_indicator)
        recent_directions = directions[max(0, j-10):j]
        direction_freq = np.mean(recent_directions == 1) if len(recent_directions) > 0 else 0.5
        feature.append(direction_freq)
    return feature

def build_features_labels(previous_results, max_order=7):
    if len(previous_results) < max_order + 1:
        return [], []
    sector_indices = np.array([result['sector_index'] for result in previous_results])
    directions = np.array([1 if result['direction'] == 'R' else -1 for result in previous_results])
    recent_steps = []
    for i in range(max(0, len(previous_results)-100), len(previous_results)):
        curr_sec = sector_indices[i]
        prev_sec = sector_indices[i-1] if i > 0 else curr_sec
        curr_dir = 1 if previous_results[i]['direction'] == 'R' else -1
        step = circular_distance(curr_sec, prev_sec, NUMBER_OF_SECTORS)
        recent_steps.append(step * curr_dir)
    large_step_threshold = np.percentile(np.abs(recent_steps), 75) if recent_steps else 10.0
    recent_steps_cache = np.array(recent_steps)
    raw_features = []
    raw_labels = []
    for i in range(max_order, len(previous_results)):
        raw_features.append(
            build_features_for_spin(
                i, sector_indices, directions, previous_results, large_step_threshold, recent_steps_cache, max_order
            )
        )
        raw_labels.append(sector_indices[i])
    valid_mask = [not (np.isnan(f).any() or np.isinf(f).any()) for f in raw_features]
    features = [f for f, ok in zip(raw_features, valid_mask) if ok]
    labels = [l for l, ok in zip(raw_labels, valid_mask) if ok]
    if not features:
        logging.warning("No valid feature vectors generated.")
        return [], []
    return np.array(features), np.array(labels)

def calculate_sector_frequencies(previous_results):
    sector_counts = Counter([result['sector_index'] for result in previous_results])
    total = sum(sector_counts.values())
    sector_frequencies = {i: count / total for i, count in sector_counts.items()}
    class_weights = {i: 1 / freq if freq > 0 else 1.0 for i, freq in sector_frequencies.items()}
    return class_weights

# === ML_STACKING ===
cb_params = {
    'iterations': 200,
    'depth': 5,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3,
    'early_stopping_rounds': 50,
    'loss_function': 'MultiClass',
    'random_seed': 42,
    'thread_count': -1,
    'verbose': 0
}

hgb_params = {
    'max_iter': 200,
    'max_depth': 5,
    'learning_rate': 0.05,
    'l2_regularization': 1.0,
    'early_stopping': True,
    'n_iter_no_change': 20,
    'random_state': 42
}

lgb_params = {
    'n_estimators': 200,
    'max_depth': 5,
    'num_leaves': 15,
    'learning_rate': 0.05,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'objective': 'multiclass',
    'num_class': 39,
    'metric': 'multi_logloss',
    'n_jobs': -1,
    'random_state': 42,
    'verbosity': -1
}

lr_params = {
    'C': 1.0,
    'solver': 'lbfgs',
    'multi_class': 'multinomial',
    'max_iter': 1000,
    'class_weight': 'balanced',
    'random_state': 42
}

class MLStacker:
    def __init__(self, cb_params: dict, hgb_params: dict, lgb_params: dict, lr_params: dict):
        self.cb_params = cb_params
        self.hgb_params = hgb_params
        self.lgb_params = lgb_params
        self.lr_params = lr_params
        self.cb_model = None
        self.hgb_model = None
        self.lgb_model = None
        self.meta_model = None
        self.scaler = None
        self.keep_idx = None

    def train(self, X: np.ndarray, y: np.ndarray, test_size=0.2, random_state=42):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        self.cb_model = CalibratedClassifierCV(
            CatBoostClassifier(**self.cb_params),
            cv=2, method='sigmoid', n_jobs=-1
        )
        hgb_no_es = HistGradientBoostingClassifier(
            **{**self.hgb_params, 'early_stopping': False}
        )
        self.hgb_model = CalibratedClassifierCV(
            hgb_no_es,
            cv=2, method='sigmoid', n_jobs=-1
        )
        self.lgb_model = CalibratedClassifierCV(
            lgb.LGBMClassifier(**self.lgb_params),
            cv=2, method='sigmoid', n_jobs=-1
        )
        self.cb_model.fit(X_tr, y_tr)
        self.hgb_model.fit(X_tr, y_tr)
        self.lgb_model.fit(X_tr, y_tr)
        prob_cb_tr = self.cb_model.predict_proba(X_tr)
        prob_hgb_tr = self.hgb_model.predict_proba(X_tr)
        prob_lgb_tr = self.lgb_model.predict_proba(X_tr)
        X_meta_tr = np.hstack([prob_cb_tr, prob_hgb_tr, prob_lgb_tr])
        prob_cb_val = self.cb_model.predict_proba(X_val)
        prob_hgb_val = self.hgb_model.predict_proba(X_val)
        prob_lgb_val = self.lgb_model.predict_proba(X_val)
        X_meta_val = np.hstack([prob_cb_val, prob_hgb_val, prob_lgb_val])
        self.meta_model = CalibratedClassifierCV(
            LogisticRegression(**self.lr_params),
            cv=2, method='sigmoid', n_jobs=-1
        )
        self.meta_model.fit(X_meta_tr, y_tr)
        y_pred = self.meta_model.predict(X_meta_val)
        prec = precision_score(y_val, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_val, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_val, y_pred)
        print(f"[MLStacker] CV precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f}")
        print(f"[MLStacker] Confusion matrix:\n{cm}")
        return {
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'confusion_matrix': cm
        }

    def predict(self, X_new: np.ndarray):
        p_cb = self.cb_model.predict_proba(X_new)
        p_hgb = self.hgb_model.predict_proba(X_new)
        p_lgb = self.lgb_model.predict_proba(X_new)
        X_meta = np.hstack([p_cb, p_hgb, p_lgb])
        final_probs = self.meta_model.predict_proba(X_meta)
        idxs = np.argmax(final_probs, axis=1)
        return final_probs, idxs

ml_stacker_global = None
entropy_threshold_global = np.log(NUMBER_OF_SECTORS) * 0.95

def predict_next_sector_ml(previous_results, ml_stacker, max_order=7):
    if len(previous_results) < max_order + 1 or ml_stacker is None:
        return None, None
    sector_indices = np.array([r['sector_index'] for r in previous_results])
    directions = np.array([1 if r['direction'] == 'R' else -1 for r in previous_results])
    recent_steps = []
    for i in range(max(0, len(previous_results)-100), len(previous_results)):
        curr_sec = sector_indices[i]
        prev_sec = sector_indices[i-1] if i > 0 else curr_sec
        curr_dir = 1 if previous_results[i]['direction'] == 'R' else -1
        step = circular_distance(curr_sec, prev_sec, NUMBER_OF_SECTORS)
        recent_steps.append(step * curr_dir)
    large_step_threshold = np.percentile(np.abs(recent_steps), 75) if recent_steps else 10.0
    recent_steps_cache = np.array(recent_steps)
    feature = []
    for idx in range(len(previous_results) - max_order, len(previous_results)):
        feature.extend(build_signed_step_features(
            sector_indices, directions, idx, previous_results, large_step_threshold, recent_steps_cache
        ))
        direction_code = directions[idx]
        feature.append(direction_code)
        sector_sin = math.sin(2.0 * math.pi * sector_indices[idx] / NUMBER_OF_SECTORS)
        sector_cos = math.cos(2.0 * math.pi * sector_indices[idx] / NUMBER_OF_SECTORS)
        feature.extend([sector_sin, sector_cos])
        prev_sector = sector_indices[idx-1] if idx > 0 else sector_indices[idx]
        relative_cyclic_distance = circular_distance(sector_indices[idx], prev_sector, NUMBER_OF_SECTORS)
        feature.append(relative_cyclic_distance)
        close_far_indicator = 1 if circular_distance(sector_indices[idx], prev_sector, NUMBER_OF_SECTORS) <= 2 else 0
        feature.append(close_far_indicator)
        recent_directions = directions[max(0, idx-10):idx]
        direction_freq = np.mean(recent_directions == 1) if len(recent_directions) > 0 else 0.5
        feature.append(direction_freq)
    try:
        recent_features = np.array(feature).reshape(1, -1)
        if np.isnan(recent_features).any() or np.isinf(recent_features).any():
            recent_features = replace_with_mean(recent_features)
    except Exception as e:
        logging.error(f"Ошибка при формировании признаков: {e}")
        return None, None
    try:
        final_probs, pred_idxs = ml_stacker.predict(recent_features)
        final_probs = final_probs[0]
        predicted_sector_index = pred_idxs[0]
    except Exception as e:
        logging.error(f"Ошибка предсказания MLStacker: {e}")
        return None, None
    if entropy(final_probs) > entropy_threshold_global:
        final_probs = smooth_circular_probs(final_probs, NUMBER_OF_SECTORS, sigma=1.5)
    predicted_sector_label = get_sector_label(predicted_sector_index)
    prediction_confidence = final_probs[predicted_sector_index] * 100
    last_direction = previous_results[-1]['direction']
    predicted_direction = 'L' if last_direction == 'R' else 'R'
    try:
        cb_prob = ml_stacker.cb_model.predict_proba(recent_features)[0]
        hgb_prob = ml_stacker.hgb_model.predict_proba(recent_features)[0]
        lgb_prob = ml_stacker.lgb_model.predict_proba(recent_features)[0]
    except Exception as e:
        logging.error(f"Ошибка получения вероятностей базовых моделей: {e}")
        cb_prob = np.ones(NUMBER_OF_SECTORS) / NUMBER_OF_SECTORS
        hgb_prob = np.ones(NUMBER_OF_SECTORS) / NUMBER_OF_SECTORS
        lgb_prob = np.ones(NUMBER_OF_SECTORS) / NUMBER_OF_SECTORS
    logging.debug(f"Prob distributions: CB_max={np.max(cb_prob):.4f}, HGB_max={np.max(hgb_prob):.4f}, LGB_max={np.max(lgb_prob):.4f}")
    return {
        'direction': predicted_direction,
        'sector_label': predicted_sector_label,
        'predicted_sector_index': predicted_sector_index,
        'probabilities': final_probs,
        'cb_prob': cb_prob,
        'hgb_prob': hgb_prob,
        'lgb_prob': lgb_prob
    }, prediction_confidence

# === BAMF_MODEL ===
bamf_transition_matrix_R = np.ones((39, 39)) * ALPHA_PRIOR
bamf_transition_matrix_L = np.ones((39, 39)) * ALPHA_PRIOR
bamf_frequency_counts_R  = np.ones(39) * ALPHA_PRIOR
bamf_frequency_counts_L  = np.ones(39) * ALPHA_PRIOR
bamf_prediction_history = deque(maxlen=100)
last_processed_index = -1
adaptive_decay = DECAY_FACTOR

def calculate_bamf_accuracy(prediction_history, directions, total_sectors=39, tolerance=1):
    if len(prediction_history) == 0 or len(prediction_history) != len(directions):
        logging.warning("Invalid prediction_history or directions length for BAMF accuracy")
        return {'R': 0.5, 'L': 0.5}
    valid_list = []
    accuracies = []
    for (p_idx, a_idx, dir_hist), dir_ext in zip(prediction_history, directions):
        if not isinstance(dir_hist, str):
            continue
        valid_list.append((p_idx, a_idx, dir_ext))
        dist = circular_distance(p_idx, a_idx, total_sectors)
        is_correct = 1 if dist <= tolerance else 0
        accuracies.append(is_correct)
    correct_R, total_R = 0, 0
    correct_L, total_L = 0, 0
    for (pred_idx, actual_idx, direction), is_correct in zip(valid_list, accuracies):
        if direction not in ('R', 'L'):
            direction = 'R'
            logging.warning(f"Invalid direction in accuracy calc, defaulting to 'R'")
        if direction == 'R':
            correct_R += is_correct
            total_R += 1
        else:
            correct_L += is_correct
            total_L += 1
    accuracy_R = correct_R / total_R if total_R > 0 else 0.5
    accuracy_L = correct_L / total_L if total_L > 0 else 0.5
    return {'R': accuracy_R, 'L': accuracy_L, 'accuracies': accuracies}

def adapt_bamf_weights(accuracy_dict, entropy_val, max_entropy=np.log(39), accuracies=None):
    weights = {}
    for direction in ['R', 'L']:
        accuracy = accuracy_dict.get(direction, 0.5)
        entropy_factor = 1 - (entropy_val / max_entropy)
        weight_mc = 0.5 + 0.5 * accuracy
        weight_mc *= entropy_factor
        weight_freq = 1.0 - weight_mc
        weights[direction] = (weight_mc, weight_freq)
    return weights

def update_matrix(direction, transitions, frequency_counts, matrix, total_data, new_data, freq_updates):
    global adaptive_decay
    logging.debug(f"update_matrix {direction}: total_data={total_data}, new_data_size={len(new_data)}, min_data_threshold={min_data_threshold}")
    matrix *= adaptive_decay
    frequency_counts *= adaptive_decay
    if total_data < min_data_threshold:
        matrix *= adaptive_decay
        return
    adaptive_alpha = ALPHA_PRIOR * (len(new_data) / (total_data + 1e-9)) * 1.5
    logging.debug(f"Adaptive ALPHA_PRIOR for {direction}: {adaptive_alpha:.4f}")
    for prev_idx, curr_idx, weight in transitions:
        if not (0 <= prev_idx < 39 and 0 <= curr_idx < 39):
            logging.warning(f"Invalid sector indices: {prev_idx} -> {curr_idx} in update_matrix {direction}")
            continue
        matrix[prev_idx, curr_idx] += weight
    for sec, weight in freq_updates:
        if not (0 <= sec < 39):
            logging.warning(f"Invalid sector index {sec} in update_matrix {direction}")
            continue
        frequency_counts[sec] += weight
    for i in range(39):
        matrix[i] = smooth_circular_probs(matrix[i], total_sectors=39, sigma=0.5)
        row_sum = matrix[i].sum() + 39 * adaptive_alpha
        matrix[i] /= row_sum

def train_bamf(previous_results):
    global bamf_transition_matrix_R, bamf_transition_matrix_L
    global bamf_frequency_counts_R, bamf_frequency_counts_L
    global last_processed_index
    global adaptive_decay
    if len(previous_results) < 2:
        return
    data = previous_results[-BAMF_MAX_HISTORY:] if len(previous_results) > BAMF_MAX_HISTORY else previous_results
    seq = [r['sector_index'] for r in data]
    if len(seq) != len(data):
        logging.error(f"Mismatch: len(seq)={len(seq)}, len(data)={len(data)} in train_bamf")
        return
    steps = [circular_distance(seq[i], seq[i-1], 39) for i in range(1, len(seq))]
    volatility = np.std(steps) if steps else 1.0
    adaptive_decay = 0.995 * (1 - 0.1 * volatility / 10)
    half_life = 50
    lambda_decay = np.log(2) / half_life
    timestamps = [r['timestamp'] for r in data]
    latest_time = timestamps[-1] if timestamps else datetime.datetime.now()
    time_deltas = [(latest_time - t).total_seconds() / 3600 for t in timestamps]
    weights = np.exp(-lambda_decay * np.array(time_deltas))
    transitions_R = []
    transitions_L = []
    frequency_counts_updates_R = []
    frequency_counts_updates_L = []
    start_idx = max(last_processed_index + 1, len(data) - BAMF_MAX_HISTORY)
    for i in range(start_idx, len(data)):
        direction = data[i].get('direction', 'R')
        if direction not in ('R', 'L'):
            logging.warning(f"Invalid direction at index {i}, defaulting to 'R'")
            direction = 'R'
        weight = weights[i]
        if i > 0:
            prev_idx, curr_idx = seq[i-1], seq[i]
            if direction == 'R':
                transitions_R.append((prev_idx, curr_idx, weight))
            else:
                transitions_L.append((prev_idx, curr_idx, weight))
        sec = seq[i]
        if direction == 'R':
            frequency_counts_updates_R.append((sec, weight))
        else:
            frequency_counts_updates_L.append((sec, weight))
    total_R = sum(1 for res in data[start_idx:] if res.get('direction', 'R') == 'R')
    total_L = len(data[start_idx:]) - total_R
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(
            update_matrix,
            'R',
            transitions_R,
            bamf_frequency_counts_R,
            bamf_transition_matrix_R,
            total_R,
            data[start_idx:],
            frequency_counts_updates_R
        )
        executor.submit(
            update_matrix,
            'L',
            transitions_L,
            bamf_frequency_counts_L,
            bamf_transition_matrix_L,
            total_L,
            data[start_idx:],
            frequency_counts_updates_L
        )
    last_processed_index = max(last_processed_index, len(previous_results) - 1)

@lru_cache(maxsize=1000)
def get_transition_probs(last_sector, direction, matrix_version):
    matrix = bamf_transition_matrix_R if direction == 'R' else bamf_transition_matrix_L
    return matrix[last_sector].copy()

def predict_next_sector_bamf(previous_results):
    global bamf_transition_matrix_R, bamf_transition_matrix_L
    global bamf_frequency_counts_R, bamf_frequency_counts_L
    if not previous_results:
        final_probs = np.ones(39) / 39
        predicted_sector_index = np.argmax(final_probs)
        predicted_direction = 'R'
        confidence = final_probs[predicted_sector_index] * 100
        logging.info("No sequence data, using uniform probabilities in predict_next_sector_bamf")
        return {
            'direction': predicted_direction,
            'sector_label': get_sector_label(predicted_sector_index),
            'predicted_sector_index': predicted_sector_index,
            'probabilities': final_probs
        }, confidence
    last_direction = previous_results[-1].get('direction', 'R')
    if last_direction not in ('R', 'L'):
        logging.warning(f"Invalid last_direction: {last_direction}, defaulting to 'R'")
        last_direction = 'R'
    seq = [r['sector_index'] for r in previous_results[-BAMF_MAX_HISTORY:]]
    if not seq:
        final_probs = np.ones(39) / 39
        predicted_sector_index = np.argmax(final_probs)
        predicted_direction = 'R'
        confidence = final_probs[predicted_sector_index] * 100
        logging.info("Empty seq after slicing in predict_next_sector_bamf, uniform fallback")
        return {
            'direction': predicted_direction,
            'sector_label': get_sector_label(predicted_sector_index),
            'predicted_sector_index': predicted_sector_index,
            'probabilities': final_probs
        }, confidence
    last_sector = seq[-1]
    matrix_version = hash(str(bamf_transition_matrix_R) + str(bamf_transition_matrix_L))
    mc_probs = get_transition_probs(last_sector, last_direction, matrix_version)
    if mc_probs.sum() < 1e-9:
        logging.warning(f"Transition matrix for {last_direction} is invalid, using uniform")
        mc_probs = np.ones(39) / 39
    freq_counts = (
        bamf_frequency_counts_R if last_direction == 'R' else bamf_frequency_counts_L
    )
    total_freq = freq_counts.sum()
    freq_probs = freq_counts / total_freq
    smoothed_freq = freq_probs + 0.1
    inverse_freq = 1 / smoothed_freq
    inverse_freq = np.clip(inverse_freq, 0, 10)
    inverse_freq /= inverse_freq.sum()
    directions = [r.get('direction', 'R') for r in previous_results[-len(bamf_prediction_history):]]
    accuracy_info = calculate_bamf_accuracy(
        bamf_prediction_history,
        directions,
        total_sectors=NUMBER_OF_SECTORS,
        tolerance=BAMF_TOLERANCE
    )
    current_accuracy = {
        'R': accuracy_info['R'],
        'L': accuracy_info['L']
    }
    entropy_val = -np.sum(mc_probs * np.log(mc_probs + 1e-9))
    weights = adapt_bamf_weights(
        current_accuracy, entropy_val, accuracies=accuracy_info.get('accuracies', [])
    )
    weight_mc, weight_freq = weights.get(last_direction, (0.5, 0.5))
    combined_probs = mc_probs * weight_mc + inverse_freq * weight_freq
    final_probs = combined_probs / (combined_probs.sum() + 1e-9)
    adaptive_sigma = 0.5 + (1.0 - entropy_val / np.log(NUMBER_OF_SECTORS)) * 0.5
    final_probs = smooth_circular_probs(final_probs, total_sectors=NUMBER_OF_SECTORS, sigma=adaptive_sigma)
    predicted_sector_index = np.argmax(final_probs)
    predicted_direction = 'L' if last_direction == 'R' else 'R'
    confidence = final_probs[predicted_sector_index] * 100
    logging.debug(
        f"[BAMF] Predicted sector={get_sector_label(predicted_sector_index)}, "
        f"Direction={predicted_direction}, Confidence={confidence:.3f}"
    )
    return {
        'direction': predicted_direction,
        'sector_label': get_sector_label(predicted_sector_index),
        'predicted_sector_index': predicted_sector_index,
        'probabilities': final_probs
    }, confidence

# === HHMM_MODEL ===
cat_hmm_global = None
crf_global = None
meta_model_global = None
best_crf_params_global = None

hhmm_params = {
    'n_components': 10,
    'n_iter': 100,
    'tol': 1e-4,
    'init_params': 'ste',
    'params': 'ste',
    'algorithm': 'viterbi',
    'random_state': 42
}

def build_crf_dataset(previous_results, sequence_length=5):
    if not previous_results or len(previous_results) < sequence_length:
        return [[]], [[]]
    X_seqs = []
    y_seqs = []
    sector_indices = [res['sector_index'] for res in previous_results]
    for start in range(len(previous_results) - sequence_length + 1):
        end = start + sequence_length
        window = previous_results[start:end]
        X_seq = []
        y_seq = []
        for i, result in enumerate(window):
            direction_code = 1 if result['direction'] == 'R' else -1
            sector_index = result['sector_index']
            feat_dict = {
                'direction_code': str(direction_code),
                'sector_index': str(sector_index),
            }
            if i > 0:
                prev_sector_index = window[i - 1]['sector_index']
                sector_diff = circular_distance(sector_index, prev_sector_index, 39)
                feat_dict['sector_diff'] = str(sector_diff)
                close_far_indicator = 1 if sector_diff <= 2 else 0
                feat_dict['close_far_indicator'] = str(close_far_indicator)
            else:
                feat_dict['sector_diff'] = '0'
                feat_dict['close_far_indicator'] = '0'
            cyc_dist_from_0 = circular_distance(sector_index, 0, 39)
            feat_dict['cyc_dist_from_0'] = str(cyc_dist_from_0)
            j_global = start + i
            recent_span = 5
            start_recent = max(0, j_global - recent_span + 1)
            recent_window = np.array(sector_indices[start_recent:j_global+1])
            freq_in_recent_5 = np.sum(recent_window == sector_index) / (len(recent_window)+1e-9)
            feat_dict['freq_in_recent_5'] = str(freq_in_recent_5)
            X_seq.append(feat_dict)
            y_seq.append(str(sector_index))
        X_seqs.append(X_seq)
        y_seqs.append(y_seq)
    return X_seqs, y_seqs

def predict_next_sector_hhmm(previous_results):
    global cat_hmm_global, crf_global, meta_model_global, best_crf_params_global
    if len(previous_results) < 5:
        return None, None
    sector_indices = [res['sector_index'] for res in previous_results]
    observations = np.array(sector_indices).reshape(-1, 1)
    try:
        from hmmlearn.hmm import CategoricalHMM
        cat_hmm = CategoricalHMM(**hhmm_params)
        cat_hmm.fit(observations)
        X_seqs, y_seqs = build_crf_dataset(previous_results, sequence_length=5)
        if not X_seqs or not y_seqs or (len(X_seqs) == 1 and not X_seqs[0]):
            logging.warning("Недостаточно данных для обучения CRF.")
            return None, None
        max_crf_iterations = 100
        if crf_global is None:
            parameters = {'c1': [0.001, 0.01, 0.1], 'c2': [0.001, 0.01, 0.1]}
            crf = SKCRF(
                algorithm='lbfgs',
                max_iterations=max_crf_iterations,
                all_possible_transitions=True
            )
            gs = GridSearchCV(crf, parameters, cv=2, verbose=0, n_jobs=1)
            gs.fit(X_seqs, y_seqs)
            best_crf = gs.best_estimator_
            best_crf_params_global = gs.best_params_
            crf_global = best_crf
        else:
            c1_val = best_crf_params_global.get('c1', 0.01)
            c2_val = best_crf_params_global.get('c2', 0.01)
            best_crf = SKCRF(
                algorithm='lbfgs',
                max_iterations=max_crf_iterations,
                all_possible_transitions=True,
                c1=c1_val,
                c2=c2_val
            )
            best_crf.fit(X_seqs, y_seqs)
        state_sequence_cat = cat_hmm.predict(observations)
        X_meta = []
        y_meta = []
        limit_len = min(len(state_sequence_cat) - 1, len(X_seqs))
        for i in range(limit_len):
            cat_vec = cat_hmm.emissionprob_[state_sequence_cat[i]]
            if len(X_seqs[i]) < 1:
                logging.warning(f"Пустая последовательность X_seqs[{i}], использую равномерное распределение.")
                crf_vec = np.ones(39) / 39
            else:
                crf_seq_marg = best_crf.predict_marginals_single(X_seqs[i])[-1]
                crf_vec = np.zeros(39)
                for lbl_str, p_val in crf_seq_marg.items():
                    idx = int(lbl_str)
                    if 0 <= idx < 39:
                        crf_vec[idx] = p_val
            combined_vec = np.hstack([cat_vec, crf_vec])
            X_meta.append(combined_vec)
            y_meta.append(sector_indices[i+1])
        meta_model = CatBoostClassifier(
            iterations=200,
            random_seed=42,
            verbose=False,
            od_type='Iter',
            od_wait=20
        )
        if len(X_meta) > 10:
            X_meta_arr = np.array(X_meta)
            y_meta_arr = np.array(y_meta)
            meta_model.fit(X_meta_arr, y_meta_arr)
        else:
            meta_model = None
        cat_hmm_global = cat_hmm
        crf_global = best_crf
        meta_model_global = meta_model
        last_state_cat = state_sequence_cat[-1]
        transmat_cat = cat_hmm.transmat_[last_state_cat]
        emission_probs_cat = np.zeros(39)
        for state_idx, state_prob in enumerate(transmat_cat):
            emission_probs_cat += state_prob * cat_hmm.emissionprob_[state_idx]
        if emission_probs_cat.sum() == 0:
            emission_probs_cat = np.ones(39) / 39
        else:
            emission_probs_cat /= emission_probs_cat.sum()
        crf_marginals = best_crf.predict_marginals_single(X_seqs[-1])
        last_marginal = crf_marginals[-1]
        emission_probs_crf = np.zeros(39)
        for label_str, prob_val in last_marginal.items():
            idx = int(label_str)
            if 0 <= idx < 39:
                emission_probs_crf[idx] = prob_val
        if meta_model is not None:
            final_input_vec = np.hstack([emission_probs_cat, emission_probs_crf]).reshape(1, -1)
            final_probs = meta_model.predict_proba(final_input_vec)[0]
            if len(final_probs) < 39:
                extended_probs = np.zeros(39)
                for i, p in enumerate(final_probs):
                    if i < 39:
                        extended_probs[i] = p
                final_probs = extended_probs
            if final_probs.sum() == 0:
                final_probs = np.ones(39) / 39
            else:
                final_probs /= final_probs.sum()
        else:
            final_probs = 0.5 * emission_probs_cat + 0.5 * emission_probs_crf
        final_probs = smooth_circular_probs(final_probs, 39, sigma=1.0)
        predicted_sector_index = np.argmax(final_probs)
        predicted_sector_label = get_sector_label(predicted_sector_index)
        confidence = final_probs[predicted_sector_index] * 100
        last_direction = previous_results[-1]['direction']
        predicted_direction = 'L' if last_direction == 'R' else 'R'
    except Exception as e:
        logging.error(f"Ошибка при работе HHMM: {e}")
        return None, None
    return {
        'direction': predicted_direction,
        'sector_label': predicted_sector_label,
        'predicted_sector_index': predicted_sector_index,
        'probabilities': final_probs
    }, confidence

# === STACKING_MODEL ===
final_lr_model = None
final_lr_scaler = None
probs_history = []

def train_final_lr_model(probs_history, current_direction):
    global final_lr_model, final_lr_scaler
    valid_entries = [entry for entry in probs_history if entry['actual_sector'] is not None]
    if len(valid_entries) <= min_data_threshold:
        return None
    X_list = []
    y_list = []
    dir_list = []
    for entry in valid_entries:
        X_list.append(entry['vector'])
        y_list.append(entry['actual_sector'])
        dir_list.append(entry['direction'])
    X_array = np.array(X_list)
    y_array = np.array(y_list)
    directions_array = np.array(dir_list)
    if np.isnan(X_array).any() or np.isinf(X_array).any():
        X_array = replace_with_mean(X_array)
    weights = np.where(directions_array == current_direction, 1.0, 0.5)
    logging.debug(
        f"Training final LR model with {len(valid_entries)} entries. "
        f"Direction weights: current_dir={current_direction}, "
        f"#1.0={np.sum(weights==1.0)}, #0.5={np.sum(weights==0.5)}"
    )
    if final_lr_scaler is None:
        final_lr_scaler = StandardScaler()
        X_scaled = final_lr_scaler.fit_transform(X_array)
    else:
        X_scaled = final_lr_scaler.transform(X_array)
    if final_lr_model is None:
        final_lr_model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
    try:
        final_lr_model.fit(X_scaled, y_array, sample_weight=weights)
    except Exception as e:
        logging.error(f"Ошибка обучения финальной LR с учетом направления: {e}")
        return None
    return final_lr_model

def predict_final_lr_model(probs_ml, probs_bamf, probs_hhmm):
    global final_lr_model, final_lr_scaler
    vector = np.concatenate([probs_ml, probs_bamf, probs_hhmm], axis=0)
    similarities = []
    for hist_idx, entry in enumerate(probs_history):
        if entry['actual_sector'] is not None:
            hist_vector = entry['vector']
            cosine_sim = np.dot(vector, hist_vector) / (np.linalg.norm(vector) * np.linalg.norm(hist_vector) + 1e-9)
            dist = circular_distance(entry['actual_sector'], np.argmax(vector[:39]), NUMBER_OF_SECTORS)
            accuracy_weight = 1.0 if dist <= BAMF_TOLERANCE else 0.5
            similarities.append((cosine_sim, accuracy_weight, hist_idx))
    if similarities:
        top_k = min(5, len(similarities))
        top_sim = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_k]
        sim_weights = np.array([sim[0] * sim[1] for sim in top_sim])
        hist_indices = [sim[2] for sim in top_sim]
        sim_weights /= sim_weights.sum() + 1e-9
        agg = np.zeros(39)
        for w, idx in zip(sim_weights, hist_indices):
            agg += probs_history[idx]['vector'][:39] * w
    else:
        sim_weights = None
        agg = np.zeros(39)
    if final_lr_model is None or final_lr_scaler is None:
        return None
    X_input = vector.reshape(1, -1)
    if np.isnan(X_input).any() or np.isinf(X_input).any():
        X_input = replace_with_last_known(X_input[0]).reshape(1, -1)
    try:
        X_scaled = final_lr_scaler.transform(X_input)
        preds = final_lr_model.predict_proba(X_scaled)[0]
        if np.isnan(preds).any():
            logging.warning("NaN в predict_proba final LR, fallback=равномерное распределение.")
            return np.ones(NUMBER_OF_SECTORS) / NUMBER_OF_SECTORS
        full_probs = np.zeros(NUMBER_OF_SECTORS)
        for cls, prob in zip(final_lr_model.classes_, preds):
            if cls < NUMBER_OF_SECTORS:
                full_probs[cls] = prob
        if sim_weights is not None:
            full_probs = full_probs * LR_WEIGHT + agg * MEM_WEIGHT
        return full_probs
    except Exception as e:
        logging.error(f"Ошибка при предсказании финальной LR: {e}")
        return None

# === MAIN ===
updates_since_last_training = 0
predictions_full_info = []
actual_results = []
predictions = []

def calculate_circular_accuracy(predictions, actual_results, total_sectors=39, tolerance=1):
    total = len(predictions)
    if total == 0:
        return 0
    correct = 0
    for (p_dir, p_sec), (a_dir, a_sec) in zip(predictions, actual_results):
        if p_dir == a_dir:
            if p_sec in sector_label_to_index and a_sec in sector_label_to_index:
                p_idx = sector_label_to_index[p_sec]
                a_idx = sector_label_to_index[a_sec]
                dist = min(abs(p_idx - a_idx), total_sectors - abs(p_idx - a_idx))
                if dist <= tolerance:
                    correct += 1
    return (correct / total) * 100

def mean_circular_error(predictions, actual_results, total_sectors=39):
    errors = []
    for p, a in zip(predictions, actual_results):
        if p[1] in sector_label_to_index and a[1] in sector_label_to_index:
            predicted_idx = sector_label_to_index[p[1]]
            actual_idx = sector_label_to_index[a[1]]
            diff = abs(predicted_idx - actual_idx)
            errors.append(min(diff, total_sectors - diff))
    if len(errors) == 0:
        return 0.0
    return np.mean(errors)

def clear_console():
    if sys.platform.startswith('win'):
        _ = os.system('cls')
    else:
        _ = os.system('clear')

def get_user_input():
    print("\033[2K\rВведите реальный результат: ", end='', flush=True)
    try:
        input_bytes = sys.stdin.buffer.readline().strip()
        return input_bytes
    except Exception as e:
        logging.error(f"Ошибка ввода: {e}")
        print(f"\nОшибка ввода: {e}")
        return None

def predict_next_sector_combined(previous_results, ml_stacker):
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_ml = executor.submit(predict_next_sector_ml, previous_results, ml_stacker)
        future_bamf = executor.submit(predict_next_sector_bamf, previous_results)
        future_hhmm = executor.submit(predict_next_sector_hhmm, previous_results)
        ml_prediction, ml_conf = future_ml.result()
        bamf_prediction, bamf_conf = future_bamf.result()
        hhmm_prediction, hhmm_conf = future_hhmm.result()
    if not ml_prediction or not bamf_prediction or not hhmm_prediction:
        logging.warning("Один из методов вернул None, прогноз невозможен.")
        return None, None
    probs_ml = ml_prediction['probabilities']
    probs_bamf = bamf_prediction['probabilities']
    probs_hhmm = hhmm_prediction['probabilities']
    last_direction = previous_results[-1]['direction']
    combined_vector = np.concatenate([probs_ml, probs_bamf, probs_hhmm])
    probs_history.append({
        'vector': combined_vector,
        'direction': last_direction,
        'actual_sector': None
    })
    if len(probs_history) > min_data_threshold:
        trained_lr = train_final_lr_model(probs_history, last_direction)
        if trained_lr:
            logging.info("Финальная логистическая регрессия обновлена в predict_next_sector_combined.")
    final_probs = None
    if final_lr_model is not None:
        final_probs = predict_final_lr_model(probs_ml, probs_bamf, probs_hhmm)
    if final_probs is None:
        final_probs = probs_ml * 0.33 + probs_bamf * 0.33 + probs_hhmm * 0.34
    final_probs = smooth_circular_probs(final_probs, total_sectors=39, sigma=1.0)
    predicted_sector_index = np.argmax(final_probs)
    predicted_sector_label = get_sector_label(predicted_sector_index)
    predicted_direction = 'L' if last_direction == 'R' else 'R'
    final_confidence = final_probs[predicted_sector_index] * 100
    logging.info(f"Прогноз: направление {predicted_direction}, сектор {predicted_sector_label}")
    logging.info(f"Надежность прогноза: {final_confidence:.2f}%")
    prediction_info = {
        'direction': predicted_direction,
        'sector_label': predicted_sector_label,
        'predicted_sector_index': predicted_sector_index,
        'final_probabilities': final_probs,
        'ml_prediction': ml_prediction,
        'bamf_prediction': bamf_prediction,
        'hhmm_prediction': hhmm_prediction
    }
    expected_direction = 'L' if last_direction == 'R' else 'R'
    if prediction_info['direction'] != expected_direction:
        logging.warning(
            f"Direction mismatch in final prediction: {prediction_info['direction']} vs {expected_direction}"
        )
    return prediction_info, final_confidence

def main():
    global ml_stacker_global, updates_since_last_training
    global predictions_full_info, actual_results, predictions
    global final_lr_model
    global probs_history
    previous_results = []
    if sys.version_info >= (3, 7):
        sys.stdin.reconfigure(encoding='utf-8')
        sys.stdout.reconfigure(encoding='utf-8')
    else:
        import codecs
        sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach())
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    print("Введите MASSIVE:")
    try:
        input_bytes = sys.stdin.buffer.readline()
        input_string = input_bytes.decode('utf-8').strip()
    except UnicodeDecodeError:
        print("Ошибка декодирования ввода. Пожалуйста, используйте кодировку UTF-8.")
        logging.error("Ошибка декодирования ввода. Используйте UTF-8.")
        return
    except Exception as e:
        print(f"Ошибка ввода: {e}")
        logging.error(f"Ошибка ввода: {e}")
        return
    data_list = input_string.strip().split(',')
    for data in data_list:
        result = parse_result(data)
        if result:
            previous_results.append(result)
        else:
            print(f"Результат '{data}' пропущен.")
            logging.warning(f"Результат '{data}' пропущен из-за ошибки парсинга.")
    print("Первичное обучение ML модели.")
    if len(previous_results) > 10:
        features, labels = build_features_labels(previous_results, max_order=7)
        if len(features) > 5:
            ml_stacker = MLStacker(cb_params, hgb_params, lgb_params, lr_params)
            metrics = ml_stacker.train(features, labels)
            ml_stacker_global = ml_stacker
            print("ML модель успешно обучена.")
            logging.info("ML модель успешно обучена.")
            logging.info(f"MLStacker metrics: precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}")
        else:
            print("Недостаточно данных для обучения модели.")
            logging.warning("Недостаточно данных для обучения ML модели.")
    else:
        print("Слишком мало результатов для первичного обучения ML.")
        logging.warning("Недостаточно данных для обучения ML модели.")
    train_bamf(previous_results)
    while True:
        clear_console()
        print(f"{gray_color}Предыдущие 100 результатов прогноза:{reset_color}")
        last_hundred_indices = range(max(0, len(predictions_full_info)-100), len(predictions_full_info))
        for idx_count, idx in enumerate(last_hundred_indices, start=1):
            if idx < len(predictions_full_info) and idx < len(predictions) and idx < len(actual_results):
                p_info, conf = predictions_full_info[idx]
                p_dir, p_sec = predictions[idx]
                a_dir, a_sec = actual_results[idx]
                print(gray_color + f"Результат #{idx_count}:" + reset_color)
                print(gray_color + "Прогнозы отдельных методов:" + reset_color)
                if p_info.get('ml_prediction'):
                    ml_probs = p_info['ml_prediction']['probabilities']
                    mi = p_info['ml_prediction']['predicted_sector_index']
                    ml_confidence = ml_probs[mi]*100 if ml_probs is not None else 0
                    ml_line_color = green_color if (p_info['ml_prediction']['direction'] == a_dir and p_info['ml_prediction']['sector_label'] == a_sec) else gray_color
                    print(ml_line_color + f"  ML: сектор {p_info['ml_prediction']['sector_label']} ({ml_confidence:.2f}%)" + reset_color)
                if p_info.get('bamf_prediction'):
                    bamf_probs = p_info['bamf_prediction']['probabilities']
                    bi = p_info['bamf_prediction']['predicted_sector_index']
                    bamf_confidence = bamf_probs[bi]*100 if bamf_probs is not None else 0
                    bamf_line_color = green_color if (p_info['bamf_prediction']['direction'] == a_dir and p_info['bamf_prediction']['sector_label'] == a_sec) else gray_color
                    print(bamf_line_color + f"  BAMF: сектор {p_info['bamf_prediction']['sector_label']} ({bamf_confidence:.2f}%)" + reset_color)
                if p_info.get('hhmm_prediction'):
                    hhmm_probs = p_info['hhmm_prediction']['probabilities']
                    hi = p_info['hhmm_prediction']['predicted_sector_index']
                    hhmm_confidence = hhmm_probs[hi]*100 if hhmm_probs is not None else 0
                    hhmm_line_color = green_color if (p_info['hhmm_prediction']['direction'] == a_dir and p_info['hhmm_prediction']['sector_label'] == a_sec) else gray_color
                    print(hhmm_line_color + f"  HHMM: сектор {p_info['hhmm_prediction']['sector_label']} ({hhmm_confidence:.2f}%)" + reset_color)
                print(gray_color + f"Прогноз финальный: направление {p_info['direction']}, сектор {p_info['sector_label']}" + reset_color)
                print(gray_color + f"Надежность прогноза: {conf:.2f}%" + reset_color)
                print(gray_color + f"Фактический результат: {a_dir}, сектор {a_sec}\n" + reset_color)
        try:
            accuracy = calculate_circular_accuracy(predictions, actual_results, 39, tolerance=1)
        except Exception as e:
            accuracy = 0
            logging.error(f"Ошибка при расчёте точности: {e}")
        mce = mean_circular_error(predictions, actual_results, 39)
        print(f"Общая результативность: {accuracy:.2f}%\n")
        if ml_stacker_global and len(previous_results) >= 10:
            prediction_info, confidence = predict_next_sector_combined(previous_results, ml_stacker_global)
            if prediction_info:
                print("Прогнозы отдельных методов:")
                if prediction_info.get('ml_prediction'):
                    ml_probs = prediction_info['ml_prediction']['probabilities']
                    mi = prediction_info['ml_prediction']['predicted_sector_index']
                    ml_confidence = ml_probs[mi]*100 if ml_probs is not None else 0
                    print(f"  {yellow_color}ML: сектор {prediction_info['ml_prediction']['sector_label']} {gray_color}({ml_confidence:.2f}%) {reset_color}")
                if prediction_info.get('bamf_prediction'):
                    bamf_probs = prediction_info['bamf_prediction']['probabilities']
                    bi = prediction_info['bamf_prediction']['predicted_sector_index']
                    bamf_confidence = bamf_probs[bi]*100 if bamf_probs is not None else 0
                    print(f"  {yellow_color}BAMF: сектор {prediction_info['bamf_prediction']['sector_label']} {gray_color}({bamf_confidence:.2f}%) {reset_color}")
                if prediction_info.get('hhmm_prediction'):
                    hhmm_probs = prediction_info['hhmm_prediction']['probabilities']
                    hi = prediction_info['hhmm_prediction']['predicted_sector_index']
                    hhmm_confidence = hhmm_probs[hi]*100 if hhmm_probs is not None else 0
                    print(f"  {yellow_color}HHMM: сектор {prediction_info['hhmm_prediction']['sector_label']} {gray_color}({hhmm_confidence:.2f}%) {reset_color}")
                print(f" {green_color}Финальный прогноз: направление {prediction_info['direction']}, сектор {prediction_info['sector_label']}{reset_color}")
                print(f"Надежность прогноза: {confidence:.2f}%")
            else:
                print("Не удалось сделать прогноз.")
                prediction_info = None
                confidence = 0
        else:
            prediction_info = None
            confidence = 0
            print("Недостаточно данных или модель ещё не обучена.")
        print("\n" + "="*50)
        input_bytes = get_user_input()
        if not input_bytes:
            print("Ошибка ввода: Пустая строка. Повторите ввод.")
            continue
        try:
            new_data = input_bytes.decode('utf-8').strip()
            logging.info(f"Получен ввод: '{new_data}'")
        except UnicodeDecodeError:
            print("Ошибка декодирования ввода. Пожалуйста, используйте кодировку UTF-8.")
            logging.error("Ошибка декодирования ввода. Используйте UTF-8.")
            continue
        if not new_data.strip():
            print("Ошибка ввода: Пустая строка. Повторите ввод.")
            continue
        new_result = parse_result(new_data)
        if not new_result:
            print("Результат не добавлен из-за ошибки ввода. Используйте формат: R/L + номер сектора (например, R12 или L5/6).")
            logging.warning(f"Неверный ввод: '{new_data}'")
            continue
        previous_results.append(new_result)
        actual_result = (
            new_result['direction'],
            get_sector_label(new_result['sector_index']) if get_sector_label(new_result['sector_index']) else "?"
        )
        actual_results.append(actual_result)
        logging.info(f"Добавлен фактический результат: {actual_result}")
        if probs_history:
            probs_history[-1]['actual_sector'] = new_result['sector_index']
        if prediction_info:
            predictions.append((prediction_info['direction'], prediction_info['sector_label']))
            predictions_full_info.append((prediction_info, confidence))
            bamf_pred = prediction_info.get('bamf_prediction')
            if bamf_pred:
                bamf_pred_idx = bamf_pred['predicted_sector_index']
                actual_idx = new_result['sector_index']
                last_direction = previous_results[-1].get('direction', 'R')
                bamf_prediction_history.append((bamf_pred_idx, actual_idx, last_direction))
        else:
            predictions.append(("?", "?"))
        updates_since_last_training += 1
        if updates_since_last_training >= TRAIN_INTERVAL:
            updates_since_last_training = 0
            features, labels = build_features_labels(previous_results, max_order=7)
            if len(features) > 5:
                ml_stacker = MLStacker(cb_params, hgb_params, lgb_params, lr_params)
                metrics = ml_stacker.train(features, labels)
                ml_stacker_global = ml_stacker
                print("ML модель переобучена.")
                logging.info("ML модель переобучена.")
                logging.info(f"MLStacker metrics: precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}")
            train_bamf(previous_results)
            directions = [r.get('direction', 'R') for r in previous_results[-len(bamf_prediction_history):]]
            accuracy_dict = calculate_bamf_accuracy(bamf_prediction_history, directions)
            logging.info(
                f"BAMF Accuracy: R={accuracy_dict['R']*100:.2f}%, "
                f"L={accuracy_dict['L']*100:.2f}%"
            )
            logging.info(
                f"Overall circular accuracy: "
                f"{calculate_circular_accuracy(predictions, actual_results, 39, tolerance=1):.2f}%"
            )
            logging.info(
                f"Mean circular error: "
                f"{mean_circular_error(predictions, actual_results, 39):.2f}"
            )
    total_accuracy = calculate_circular_accuracy(predictions, actual_results, 39, tolerance=1)
    print(f"Общая точность прогнозов (циклическая): {total_accuracy:.2f}%")
    logging.info(f"Общая точность прогнозов (циклическая): {total_accuracy:.2f}%")

if __name__ == "__main__":
    main()
