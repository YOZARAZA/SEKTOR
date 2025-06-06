
# -*- coding: utf-8 -*-

import math
import datetime
import sys
import numpy as np
import warnings
from functools import lru_cache
from joblib import Parallel, delayed
import random
import logging
import os
from collections import defaultdict

warnings.filterwarnings("ignore")

try:
    from numba import njit
except ModuleNotFoundError:
    def njit(func):
        return func

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

WHEEL_DIAMETER      = 1.8
WHEEL_CIRCUMFERENCE = math.pi * WHEEL_DIAMETER
NUMBER_OF_SECTORS   = 39
SECTOR_ANGLE        = 360 / NUMBER_OF_SECTORS

# Нормировочный коэффициент для итоговой дистанции
NORMALIZATION_COEFFICIENT = 0.8

# --------------------------
# Новый список моделей:
#  - Экспоненциальное трение
#  - INITIAL_SPEED_MIN/MAX
#  - Остальные поля для совместимости
# --------------------------
physics_params_list = [
    # Physics1 (~3.7 оборотов)
    {
        "INITIAL_SPEED_MIN": 1.33,
        "INITIAL_SPEED_MAX": 1.34,
        "FRICTION": 0.150,
        "AVERAGE_SPEED": 1.33,
        "SPEED_STD": 0.04,
        "ACCELERATION_TIME_MIN": 1.0,
        "ACCELERATION_TIME_MAX": 2.0,
        "TOTAL_DURATION_MIN": 26,
        "TOTAL_DURATION_MAX": 28
    },
    # Physics2 (~3.8 оборотов)
    {
        "INITIAL_SPEED_MIN": 1.30,
        "INITIAL_SPEED_MAX": 1.31,
        "FRICTION": 0.140,
        "AVERAGE_SPEED": 1.30,
        "SPEED_STD": 0.04,
        "ACCELERATION_TIME_MIN": 1.1,
        "ACCELERATION_TIME_MAX": 2.1,
        "TOTAL_DURATION_MIN": 27,
        "TOTAL_DURATION_MAX": 29
    },
    # Physics3 (~3.9 оборотов)
    {
        "INITIAL_SPEED_MIN": 1.27,
        "INITIAL_SPEED_MAX": 1.28,
        "FRICTION": 0.130,
        "AVERAGE_SPEED": 1.27,
        "SPEED_STD": 0.04,
        "ACCELERATION_TIME_MIN": 1.2,
        "ACCELERATION_TIME_MAX": 2.2,
        "TOTAL_DURATION_MIN": 26,
        "TOTAL_DURATION_MAX": 30
    },
    # Physics4 (~4.0 оборотов)
    {
        "INITIAL_SPEED_MIN": 1.24,
        "INITIAL_SPEED_MAX": 1.25,
        "FRICTION": 0.120,
        "AVERAGE_SPEED": 1.24,
        "SPEED_STD": 0.04,
        "ACCELERATION_TIME_MIN": 1.3,
        "ACCELERATION_TIME_MAX": 2.3,
        "TOTAL_DURATION_MIN": 29,
        "TOTAL_DURATION_MAX": 31
    },
    # Physics5 (~4.1 оборотов)
    {
        "INITIAL_SPEED_MIN": 1.21,
        "INITIAL_SPEED_MAX": 1.22,
        "FRICTION": 0.110,
        "AVERAGE_SPEED": 1.21,
        "SPEED_STD": 0.04,
        "ACCELERATION_TIME_MIN": 1.4,
        "ACCELERATION_TIME_MAX": 2.4,
        "TOTAL_DURATION_MIN": 30,
        "TOTAL_DURATION_MAX": 32
    },
    # Physics6 (~4.2 оборотов)
    {
        "INITIAL_SPEED_MIN": 1.16,
        "INITIAL_SPEED_MAX": 1.17,
        "FRICTION": 0.090,
        "AVERAGE_SPEED": 1.16,
        "SPEED_STD": 0.04,
        "ACCELERATION_TIME_MIN": 1.5,
        "ACCELERATION_TIME_MAX": 2.5,
        "TOTAL_DURATION_MIN": 32,
        "TOTAL_DURATION_MAX": 34
    },
    # Physics7 (~4.3 оборотов)
    {
        "INITIAL_SPEED_MIN": 1.10,
        "INITIAL_SPEED_MAX": 1.11,
        "FRICTION": 0.080,
        "AVERAGE_SPEED": 1.10,
        "SPEED_STD": 0.04,
        "ACCELERATION_TIME_MIN": 1.6,
        "ACCELERATION_TIME_MAX": 2.6,
        "TOTAL_DURATION_MIN": 33,
        "TOTAL_DURATION_MAX": 35
    },
    # Physics8 (~4.5 оборотов)
    {
        "INITIAL_SPEED_MIN": 1.05,
        "INITIAL_SPEED_MAX": 1.06,
        "FRICTION": 0.070,
        "AVERAGE_SPEED": 1.05,
        "SPEED_STD": 0.04,
        "ACCELERATION_TIME_MIN": 1.7,
        "ACCELERATION_TIME_MAX": 2.7,
        "TOTAL_DURATION_MIN": 34,
        "TOTAL_DURATION_MAX": 36
    },
    # Physics9 (~4.7 оборотов)
    {
        "INITIAL_SPEED_MIN": 1.00,
        "INITIAL_SPEED_MAX": 1.01,
        "FRICTION": 0.060,
        "AVERAGE_SPEED": 1.00,
        "SPEED_STD": 0.04,
        "ACCELERATION_TIME_MIN": 1.8,
        "ACCELERATION_TIME_MAX": 2.8,
        "TOTAL_DURATION_MIN": 36,
        "TOTAL_DURATION_MAX": 38
    },
    # Physics10 (~4.9 оборотов)
    {
        "INITIAL_SPEED_MIN": 0.95,
        "INITIAL_SPEED_MAX": 0.96,
        "FRICTION": 0.050,
        "AVERAGE_SPEED": 0.95,
        "SPEED_STD": 0.04,
        "ACCELERATION_TIME_MIN": 2.0,
        "ACCELERATION_TIME_MAX": 3.0,
        "TOTAL_DURATION_MIN": 38,
        "TOTAL_DURATION_MAX": 40
    }
]

model_errors = {
    f"Physics{i}": [] for i in range(1, 11)
}
model_errors_distance = {
    f"Physics{i}": [] for i in range(1, 11)
}

sectors = []
for i in range(1, 20):
    sectors.append(str(i))
    sectors.append(f"0({i}/{i+1})")
sectors.append('20')

sector_label_to_index = {label: index for index, label in enumerate(sectors)}
sector_index_to_label = {index: label for index, label in enumerate(sectors)}

@lru_cache(maxsize=1000)
def get_sector_label(sector_index):
    if 0 <= sector_index < len(sectors):
        return sectors[sector_index]
    else:
        return None

@njit
def circular_distance(a, b, total_sectors):
    diff = abs(a - b)
    return diff if diff <= total_sectors - diff else total_sectors - diff

def clear_console():
    if sys.platform.startswith('win'):
        os.system('cls')
    else:
        os.system('clear')

real_spin_statistics = {
    "speed_samples": [],
    "accel_time_samples": [],
    "decel_time_samples": []
}

error_window_size = 5

def cross_validate_models(data, k=5):
    """K‑fold оценка средней дистанции на валидационных фолдах."""
    if len(data) < k or k < 2:
        return {}
    fold_size = len(data) // k
    results = defaultdict(list)
    for i in range(k):
        test_data = data[i*fold_size:(i+1)*fold_size]
        for model_name in model_errors_distance.keys():
            dists = [dist for (mname, dist, _ts) in test_data if mname == model_name]
            if dists:
                results[model_name].append(np.mean(dists))
    return {m: np.mean(v) for m, v in results.items() if v}

def error_analysis_model():
    threshold = error_window_size
    adjustment_factor = 0.02
    distance_threshold = 5.0
    for i in range(1, 11):
        model_name = f"Physics{i}"
        errors = model_errors[model_name]
        if len(errors) >= threshold:
            recent_errors = errors[-threshold:]
            error_count = sum(recent_errors)
            error_rate = error_count / threshold
            if error_rate > 0.5:
                if i % 2 == 1:
                    adjustment = adjustment_factor * (error_rate - 0.5) * 2
                    physics_params_list[i-1]["AVERAGE_SPEED"] -= adjustment
                    physics_params_list[i-1]["AVERAGE_SPEED"] = max(
                        0.1, physics_params_list[i-1]["AVERAGE_SPEED"]
                    )
                else:
                    adjustment = adjustment_factor * (error_rate - 0.5) * 2
                    physics_params_list[i-1]["AVERAGE_SPEED"] += adjustment
                    physics_params_list[i-1]["AVERAGE_SPEED"] = min(
                        1.0, physics_params_list[i-1]["AVERAGE_SPEED"]
                    )
                if "SPEED_STD" in physics_params_list[i-1]:
                    if i % 2 == 1:
                        physics_params_list[i-1]["SPEED_STD"] += adjustment_factor * 0.5
                        physics_params_list[i-1]["SPEED_STD"] = min(
                            0.5, physics_params_list[i-1]["SPEED_STD"]
                        )
                    else:
                        physics_params_list[i-1]["SPEED_STD"] -= adjustment_factor * 0.5
                        physics_params_list[i-1]["SPEED_STD"] = max(
                            0.05, physics_params_list[i-1]["SPEED_STD"]
                        )
                logging.info(
                    f"Модель {model_name} скорректирована (1/0 ошибки): "
                    f"AVERAGE_SPEED={physics_params_list[i-1]['AVERAGE_SPEED']:.4f}, "
                    f"SPEED_STD={physics_params_list[i-1]['SPEED_STD']:.4f}"
                )

        dist_list = model_errors_distance[model_name]
        if len(dist_list) >= 3:
            recent_3 = [d[0] for d in dist_list[-3:]]
            avg_3_dist = np.mean(recent_3)
            if avg_3_dist > distance_threshold:
                dist_factor = (avg_3_dist - distance_threshold) / 10.0
                physics_params_list[i-1]["SPEED_STD"] = max(
                    0.05,
                    physics_params_list[i-1]["SPEED_STD"] - 0.01 * dist_factor
                )
                physics_params_list[i-1]["AVERAGE_SPEED"] = max(
                    0.1,
                    min(
                        1.0,
                        physics_params_list[i-1]["AVERAGE_SPEED"] - 0.01 * dist_factor
                    )
                )
                logging.info(
                    f"Модель {model_name} скорректирована (последние 3 дистанции): "
                    f"AVERAGE_SPEED={physics_params_list[i-1]['AVERAGE_SPEED']:.4f}, "
                    f"SPEED_STD={physics_params_list[i-1]['SPEED_STD']:.4f}"
                )

def retune_physics_params():
    if real_spin_statistics["speed_samples"]:
        real_avg_speed = np.mean(real_spin_statistics["speed_samples"])
        tuning_rate = 0.05
        for idx, params in enumerate(physics_params_list):
            current_avg_speed = params["AVERAGE_SPEED"]
            speed_difference = real_avg_speed - current_avg_speed
            adjustment = tuning_rate * speed_difference
            physics_params_list[idx]["AVERAGE_SPEED"] += adjustment
            physics_params_list[idx]["AVERAGE_SPEED"] = max(
                0.1, min(1.0, physics_params_list[idx]["AVERAGE_SPEED"])
            )
            logging.info(
                f"Модель Physics{idx+1} подстроена: "
                f"AVERAGE_SPEED={physics_params_list[idx]['AVERAGE_SPEED']:.4f}"
            )

def get_random_factor(model_name):
    # Используем узкий диапазон [0.95, 1.05]
    return np.random.uniform(0.95, 1.05)

# ---------- экспоненциальное трение + три фазы ----------
def calculate_new_sector_physics_improved(model_type, previous_result):
    if 'sector_index' not in previous_result or 'direction' not in previous_result:
        return random.randint(0, NUMBER_OF_SECTORS - 1)

    initial_sector_index = previous_result['sector_index']
    direction_code = 1 if previous_result['direction'] == 'R' else -1
    params = physics_params_list[model_type - 1]

    # Генерация начальной скорости:
    v0 = np.random.uniform(params["INITIAL_SPEED_MIN"], params["INITIAL_SPEED_MAX"])
    friction = params["FRICTION"]

    # Фазы времени:
    t_a = np.random.uniform(params["ACCELERATION_TIME_MIN"], params["ACCELERATION_TIME_MAX"])
    t_uniform = np.random.uniform(10, 17)
    t_total = np.random.uniform(params["TOTAL_DURATION_MIN"], params["TOTAL_DURATION_MAX"])
    # Корректировка суммирования фаз: гарантировать, что сумма фаз равна t_total
    t_uniform = min(t_uniform, t_total - t_a)
    t_d = t_total - t_a - t_uniform  # Добавлено определение t_d

    # Фаза 1: разгон (линейный 0..v0)
    distance_a = 0.5 * v0 * t_a

    # Фаза 2: равномерная (v0)
    distance_b = v0 * t_uniform

    # Фаза 3: экспоненциальное затухание
    # v(t) = v0 * exp(-friction * t)
    distance_c = 0.0
    if t_d > 0:
        if friction > 1e-9:
            distance_c = (v0 / friction) * (1 - math.exp(-friction * t_d))
        else:
            distance_c = v0 * t_d

    distance_total = distance_a + distance_b + distance_c

    # Применяем узкий диапазон случайного множителя: [0.95, 1.05]
    distance_total *= get_random_factor(f"Physics{model_type}")

    # Вводим коррекцию итоговой дистанции с нормировочным коэффициентом
    distance_total *= NORMALIZATION_COEFFICIENT

    # Перевод в градусы
    theta_total_degrees = (distance_total / WHEEL_CIRCUMFERENCE) * 360

    # Направление
    if direction_code == 1:
        theta_total_degrees = -theta_total_degrees

    new_sector_index = (initial_sector_index + theta_total_degrees / SECTOR_ANGLE) % NUMBER_OF_SECTORS
    return int(round(new_sector_index)) % NUMBER_OF_SECTORS

def adaptive_monte_carlo(model_type, predicted_result, max_samples=50000,
                         min_batch=5000, stability_threshold=0.005, max_iters=10):
    sector_counts = np.zeros(NUMBER_OF_SECTORS, dtype=int)
    total_samples = 0
    prev_distribution = None

    for i in range(max_iters):
        samples = Parallel(n_jobs=-2)(
            delayed(calculate_new_sector_physics_improved)(model_type, predicted_result)
            for _ in range(min_batch)
        )
        for s in samples:
            sector_counts[s] += 1
        total_samples += min_batch

        current_distribution = sector_counts / total_samples
        if prev_distribution is not None:
            diff = np.sum(np.abs(current_distribution - prev_distribution))
            if diff < stability_threshold:
                break
        prev_distribution = current_distribution.copy()
        if total_samples >= max_samples:
            break

    return sector_counts, total_samples

def predict_next_sector_physics_improved(model_type, previous_results):
    if not previous_results:
        return None, None
    last_result = previous_results[-1]
    predicted_direction = 'L' if last_result['direction'] == 'R' else 'R'
    predicted_result = {
        'direction': predicted_direction,
        'sector_index': last_result['sector_index']
    }
    max_simulations = random.randint(20000, 50000)
    try:
        sector_counts, total_samples = adaptive_monte_carlo(
            model_type,
            predicted_result,
            max_samples=max_simulations,
            min_batch=5000,
            stability_threshold=0.005,
            max_iters=10
        )
    except Exception as e:
        logging.error(f"Ошибка при выполнении адаптивных физических симуляций: {e}")
        return None, None

    physics_probs = sector_counts / total_samples
    best_idx = np.argmax(physics_probs)
    best_lbl = get_sector_label(best_idx)
    conf     = physics_probs[best_idx] * 100
    return {
        'direction': predicted_direction,
        'sector_label': best_lbl,
        'predicted_sector_index': best_idx,
        'probabilities': physics_probs
    }, conf

def ranked_voting_scoring(predictions, total_sectors):
    scores = defaultdict(int)
    for model_pred in predictions:
        for rank, sector in enumerate(model_pred):
            scores[sector] += (total_sectors - rank)
    most_probable_sector = max(scores, key=scores.get)
    highest_score        = scores[most_probable_sector]
    return most_probable_sector, highest_score

def predict_next_sector_ranked_voting(previous_results):
    predictions = []
    individual_predictions = {}
    for model_type in range(1, 11):
        prediction, _ = predict_next_sector_physics_improved(model_type, previous_results)
        if prediction:
            sorted_sectors = np.argsort(prediction['probabilities'])[::-1]
            predictions.append(sorted_sectors.tolist())
            individual_predictions[f"physics_prediction_{model_type}"] = prediction
    if predictions:
        predicted_sector_index, score = ranked_voting_scoring(predictions, NUMBER_OF_SECTORS)
        predicted_sector_label = get_sector_label(predicted_sector_index)
        last_direction = previous_results[-1]['direction']
        predicted_direction = 'L' if last_direction == 'R' else 'R'
        confidence = (score / (10 * NUMBER_OF_SECTORS)) * 100
        logging.info(f"Прогноз: направление {predicted_direction}, сектор {predicted_sector_label}")
        logging.info(f"Надёжность прогноза: {confidence:.2f}%")
        prediction_info = {
            'direction': predicted_direction,
            'sector_label': predicted_sector_label,
            'predicted_sector_index': predicted_sector_index,
            'scores': score,
            'predictions_list': predictions,
            'methods': [f"Physics{i}" for i in range(1, 11)]
        }
        for mk, val in individual_predictions.items():
            prediction_info[mk] = val
        return prediction_info, confidence
    return None, None

def calculate_accuracy(predictions, actual_results):
    total = min(len(predictions), len(actual_results))
    if total == 0:
        return 0
    total_score = 0.0
    for i in range(total):
        pred_sector_label = predictions[i][1]
        actual_sector_label = actual_results[i][1]
        pred_sector_index = sector_label_to_index.get(pred_sector_label, -1)
        actual_sector_index = sector_label_to_index.get(actual_sector_label, -1)
        d = circular_distance(pred_sector_index, actual_sector_index, NUMBER_OF_SECTORS)
        if d == 0:
            total_score += 1.0
        elif d == 1:
            total_score += 0.7
        elif d == 2:
            total_score += 0.5
        else:
            total_score += 0.0
    return (total_score / total) * 100

def get_user_input(prompt="Введите результат: "):
    print(f"\033[2K\r{prompt}", end='', flush=True)
    try:
        data = sys.stdin.buffer.readline().strip()
        if not data:
            return None
        return data.decode('utf-8', errors='ignore').strip()
    except Exception as e:
        logging.error(f"Ошибка ввода: {e}")
        print(f"\nОшибка ввода: {e}")
        return None

def analyze_errors(predictions, actual_results, prediction_info, previous_results):
    if not prediction_info or not previous_results:
        return
    real_idx = previous_results[-1]['sector_index']
    pred_idx = sector_label_to_index.get(prediction_info['sector_label'], -1)
    dist = circular_distance(pred_idx, real_idx, NUMBER_OF_SECTORS)
    model_list = prediction_info.get('methods', [])
    error_flag = 1 if pred_idx != real_idx else 0
    for m in model_list:
        model_errors[m].append(error_flag)
        model_errors_distance[m].append((dist, datetime.datetime.now()))

def loss_function(predicted_index, actual_index, total_sectors):
    d = circular_distance(predicted_index, actual_index, total_sectors)
    angle_radians = (2 * math.pi * d) / total_sectors
    chord = 2 * math.sin(angle_radians / 2)
    alpha = 0.5
    return alpha * d + (1 - alpha) * chord

def recalc_predicted_index(params, prev_result):
    if 'sector_index' not in prev_result or 'direction' not in prev_result:
        return random.randint(0, NUMBER_OF_SECTORS - 1)
    initial_sector_index = prev_result['sector_index']
    direction_code = 1 if prev_result['direction'] == 'R' else -1
    ACCELERATION_TIME_MIN = params["ACCELERATION_TIME_MIN"]
    ACCELERATION_TIME_MAX = params["ACCELERATION_TIME_MAX"]
    TOTAL_DURATION_MIN    = params["TOTAL_DURATION_MIN"]
    TOTAL_DURATION_MAX    = params["TOTAL_DURATION_MAX"]
    AVERAGE_SPEED         = params["AVERAGE_SPEED"]
    t_a = (ACCELERATION_TIME_MIN + ACCELERATION_TIME_MAX) / 2.0
    t_uniform = 13.5
    t_total = (TOTAL_DURATION_MIN + TOTAL_DURATION_MAX) / 2.0
    v = AVERAGE_SPEED
    if v < 0.1:
        v = 0.1
    distance = v * t_total
    theta_total_degrees = (distance / WHEEL_CIRCUMFERENCE) * 360
    if direction_code == 1:
        theta_total_degrees = -theta_total_degrees
    new_sector_index = (initial_sector_index + theta_total_degrees / SECTOR_ANGLE) % NUMBER_OF_SECTORS
    return int(round(new_sector_index)) % NUMBER_OF_SECTORS

def compute_gradient(param_ref, param_key, delta,
                     prev_result, actual_idx,
                     total_sectors):
    original_value = param_ref[param_key]
    param_ref[param_key] = original_value + delta
    predicted_index_up = recalc_predicted_index(param_ref, prev_result)
    loss_up = loss_function(predicted_index_up, actual_idx, total_sectors)
    param_ref[param_key] = original_value - delta
    predicted_index_down = recalc_predicted_index(param_ref, prev_result)
    loss_down = loss_function(predicted_index_down, actual_idx, total_sectors)
    param_ref[param_key] = original_value
    return (loss_up - loss_down) / (2 * delta)

def update_params(params, gradients, learning_rate=0.01):
    for key in gradients.keys():
        params[key] -= learning_rate * gradients[key]
        if key == "AVERAGE_SPEED":
            params[key] = max(0.1, min(1.0, params[key]))
        elif key == "SPEED_STD":
            params[key] = max(0.05, min(0.5, params[key]))

def apply_gradient_descent(prev_result, predicted_idx, actual_idx):
    if predicted_idx < 0 or actual_idx < 0:
        return
    current_loss = loss_function(predicted_idx, actual_idx, NUMBER_OF_SECTORS)
    for model_i, param_dict in enumerate(physics_params_list, start=1):
        grad_speed = compute_gradient(
            param_dict, 'AVERAGE_SPEED', 0.01,
            prev_result, actual_idx, NUMBER_OF_SECTORS
        )
        grad_std = compute_gradient(
            param_dict, 'SPEED_STD', 0.01,
            prev_result, actual_idx, NUMBER_OF_SECTORS
        )
        update_params(param_dict, {
            'AVERAGE_SPEED': grad_speed,
            'SPEED_STD': grad_std
        }, learning_rate=0.01)
        logging.info(
            f"Physics{model_i} параметры обновлены (градиентный спуск): "
            f"AVERAGE_SPEED={param_dict['AVERAGE_SPEED']:.4f}, "
            f"SPEED_STD={param_dict['SPEED_STD']:.4f}, "
            f"Текущая loss={current_loss:.4f}"
        )

def log_model_error_frequencies():
    for i in range(1, 11):
        model_name = f"Physics{i}"
        total_count = len(model_errors[model_name])
        if total_count > 0:
            error_count = sum(model_errors[model_name])
            freq = error_count / total_count
        else:
            freq = 0
            error_count = 0
        logging.info(
            f"Модель {model_name}: частота ошибок {freq:.2f} "
            f"(ошибок={error_count}, всего={total_count})"
        )

def main():
    if sys.version_info >= (3, 7):
        sys.stdin.reconfigure(encoding='utf-8')
        sys.stdout.reconfigure(encoding='utf-8')
    else:
        import codecs
        sys.stdin  = codecs.getreader("utf-8")(sys.stdin.detach())
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

    predictions_full_info = []
    previous_results      = []
    predictions           = []
    actual_results        = []
    prediction_info       = None
    confidence            = 0.0

    clear_console()
    init_input_str = get_user_input("Введите стартовый сектор: ")
    if init_input_str:
        data_list = [x.strip() for x in init_input_str.split(',')]
        for item in data_list:
            parsed = parse_result(item)
            if parsed:
                previous_results.append(parsed)
            else:
                logging.warning(f"Пропуск ввода: {item}")

    if previous_results:
        prediction_info, confidence = predict_next_sector_ranked_voting(previous_results)

    while True:
        clear_console()
        print(f"{gray_color}Предыдущие 100 результатов прогноза:{reset_color}")
        last_indices = range(max(0, len(predictions_full_info) - 100), len(predictions_full_info))
        for idx_count, idx in enumerate(last_indices, start=1):
            if (idx < len(predictions_full_info)
                    and idx < len(predictions)
                    and idx < len(actual_results)):
                p_info, conf_ = predictions_full_info[idx]
                p_dir, p_sec  = predictions[idx]
                a_dir, a_sec  = actual_results[idx]

                print(gray_color + f"Результат #{idx_count}:" + reset_color)
                print(gray_color + "Прогнозы отдельных методов:" + reset_color)
                for model_name in [f"physics_prediction_{i}" for i in range(1, 11)]:
                    if model_name in p_info:
                        m_pred = p_info[model_name]
                        conf_m = 0.0
                        mi     = m_pred.get('predicted_sector_index', -1)
                        probs  = m_pred.get('probabilities', [])
                        if 0 <= mi < len(probs):
                            conf_m = probs[mi] * 100
                        line_col = gray_color
                        if (m_pred['direction'] == a_dir
                                and m_pred['sector_label'] == a_sec):
                            line_col = green_color
                        short_n = model_name.replace("physics_prediction_", "Physics")
                        print(
                            line_col +
                            f"  {short_n}: сектор {m_pred['sector_label']} ({conf_m:.2f}%)" +
                            reset_color
                        )

                print(gray_color + f"Прогноз: направление {p_info['direction']}, "
                                  f"сектор {p_info['sector_label']}" + reset_color)
                print(gray_color + f"Надёжность прогноза: {conf_:.2f}%" + reset_color)
                print(gray_color + f"Фактический результат: {a_dir}, сектор {a_sec}\n" + reset_color)

        try:
            accuracy = calculate_accuracy(predictions, actual_results)
        except:
            accuracy = 0
        print(f"Общая результативность: {accuracy:.2f}%\n")

        if prediction_info:
            print("Прогнозы отдельных методов:")
            for i in range(1, 11):
                mk = f"physics_prediction_{i}"
                if mk in prediction_info:
                    pp = prediction_info[mk]
                    conf_i = 0.0
                    idx_pp = pp.get('predicted_sector_index', -1)
                    probs  = pp.get('probabilities', [])
                    if 0 <= idx_pp < len(probs):
                        conf_i = probs[idx_pp] * 100
                    match = (pp['direction'] == prediction_info['direction']
                             and pp['sector_label'] == prediction_info['sector_label'])
                    line_col = blue_color if match else yellow_color
                    print(f"  {line_col}Physics{i}: сектор {pp['sector_label']} "
                          f"{gray_color}({conf_i:.2f}%) {reset_color}")
            print(f"{green_color}Прогноз: направление {prediction_info['direction']}, "
                  f"сектор {prediction_info['sector_label']}{reset_color}")
            print(f"Надёжность прогноза: {confidence:.2f}%")
        else:
            print("Недостаточно данных для прогнозирования.")

        new_data = get_user_input("\nВведите фактический результат: ")
        if not new_data:
            print("Ошибка ввода или пустая строка. Попробуйте снова.")
            continue

        parsed_new = parse_result(new_data)
        if not parsed_new:
            print("Неверный формат ввода, попробуйте ещё раз.")
            continue

        previous_results.append(parsed_new)
        actual_dirsec = (
            parsed_new['direction'],
            get_sector_label(parsed_new['sector_index'])
            if get_sector_label(parsed_new['sector_index']) else "?"
        )
        actual_results.append(actual_dirsec)

        if 'AVERAGE_SPEED' in parsed_new:
            real_spin_statistics["speed_samples"].append(parsed_new['AVERAGE_SPEED'])

        retune_physics_params()
        error_analysis_model()

        if prediction_info:
            predictions.append((prediction_info['direction'],
                                prediction_info['sector_label']))
            real_idx = parsed_new['sector_index']
            pred_idx = sector_label_to_index.get(prediction_info['sector_label'], -1)
            match_flag = (real_idx == pred_idx)

            try:
                curr_acc = calculate_accuracy(predictions, actual_results)
            except:
                curr_acc = 0
            print(f"\nТекущая точность прогнозов: {curr_acc:.2f}%")
            logging.info(f"Текущая точность прогнозов: {curr_acc:.2f}%")

            predictions_full_info.append((prediction_info, confidence))
            analyze_errors(predictions, actual_results, prediction_info, previous_results)
            log_model_error_frequencies()

            if len(previous_results) >= 2:
                prev_result_for_gradient = previous_results[-2]
            else:
                prev_result_for_gradient = previous_results[-1]
            apply_gradient_descent(prev_result_for_gradient, pred_idx, real_idx)
        else:
            print("\nНет предыдущего прогноза для оценки.")

        new_pred_info, new_conf = predict_next_sector_ranked_voting(previous_results)
        if new_pred_info:
            prediction_info = new_pred_info
            confidence      = new_conf
        else:
            prediction_info = None
            confidence      = 0.0
            print("\nНе удалось сделать новый прогноз.")

        distData = []
        for mname, vals in model_errors_distance.items():
            for (dist, ts) in vals:
                distData.append((mname, dist, ts))
        cv_scores = {}
        if len(previous_results) % 20 == 0:
            cv_scores = cross_validate_models(distData, k=3)
        if cv_scores:
            print("\nРезультаты кросс-валидации (средняя дистанция по моделям):")
            for mn in sorted(cv_scores.keys()):
                print(f"  {mn}: {cv_scores[mn]:.2f}")

def parse_result(data_str):
    data_str = data_str.strip()
    if not data_str:
        return None
    direction = data_str[0].upper()
    if direction not in ('R', 'L'):
        print(f"Неверное направление в '{data_str}'")
        return None
    sector_str = data_str[1:].strip()
    if not sector_str:
        print(f"Не указан сектор в '{data_str}'")
        return None
    sector_str = sector_str.replace(' ', '')
    if '/' in sector_str:
        sector_label = f"0({sector_str})"
    else:
        sector_label = sector_str
    if sector_label not in sector_label_to_index:
        print(f"Неверный сектор '{sector_label}' в '{data_str}'")
        return None

    random_avg_speed = random.uniform(0.5, 0.8)
    return {
        'direction': direction,
        'sector_index': sector_label_to_index[sector_label],
        'timestamp': datetime.datetime.now(),
        'AVERAGE_SPEED': random_avg_speed
    }

if __name__ == "__main__":
    main()
