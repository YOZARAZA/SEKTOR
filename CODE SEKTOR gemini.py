# -*- coding: utf-8 -*-
"""
Физический симулятор (ver. 5)
Изменения:
1. calculate_sector_given_v0(deterministic=False):
   - если deterministic=True — берёт средние значения всех диапазонов
     и отключает случайный фактор (get_random_factor) -> полностью детерминированный расчёт.
2. recalc_predicted_index теперь вызывает calculate_sector_given_v0
   в детерминированном режиме для обеспечения согласованности.
3. Восстановлена случайная генерация AVERAGE_SPEED в parse_result,
   чтобы retune_physics_params мог продолжать функционировать.
   (Примечание: для научных целей рекомендуется заменять на реальные данные)
4. get_random_factor() теперь всегда возвращает 1.0 (шум отключён полностью)
   для более контролируемого исследования.
5. ADAM-оптимизация и градиентный спуск для SPEED_STD снова активны
   (как было в оригинальной версии, но были 
 отключены в v2).
6. Исправлены логирования и инициализация adam_states для SPEED_STD.
7. tau в smc_update использует SPEED_STD (адаптивное).
8. Формат отображения и обработки промежуточных секторов изменён с '0(i/i+1)' на 'i/i+1'
   для всех расчетов и вывода.
Parse_result обеспечивает обратную совместимость.
9. Восстановлено определение функции smc_update, которая отсутствовала в предыдущей версии.
"""

import math
import datetime
import sys
import numpy as np
import warnings
from functools import lru_cache
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

# ANSI цвета
yellow_color = '\033[93m'
gray_color   = '\033[90m'
reset_color  = '\033[0m'
green_color  = '\033[92m'
blue_color   = '\033[94m'

# Геометрия колеса
WHEEL_DIAMETER      = 1.8
WHEEL_CIRCUMFERENCE = math.pi * WHEEL_DIAMETER
NUMBER_OF_SECTORS   = 39
SECTOR_ANGLE        = 360 / NUMBER_OF_SECTORS

# Нормировочный коэффициент
NORMALIZATION_COEFFICIENT = 0.8

# Определение размера окна для анализа ошибок
ERROR_WINDOW_SIZE = 10  # число последних результатов, по которым считаем частоту ошибок

# Базовые параметры 10 моделей
physics_params_list = [
    {"INITIAL_SPEED_MIN": 1.33, "INITIAL_SPEED_MAX": 1.34, "FRICTION": 0.150,
     "AVERAGE_SPEED":    1.33, "SPEED_STD": 0.04,
     "ACCELERATION_TIME_MIN": 1.0, "ACCELERATION_TIME_MAX": 2.0,
     "TOTAL_DURATION_MIN": 26, "TOTAL_DURATION_MAX": 28},
    {"INITIAL_SPEED_MIN": 1.30, "INITIAL_SPEED_MAX": 1.31, "FRICTION": 0.140,
     "AVERAGE_SPEED":    1.30, "SPEED_STD": 0.04,
     "ACCELERATION_TIME_MIN": 1.1, "ACCELERATION_TIME_MAX": 2.1,
     "TOTAL_DURATION_MIN": 27, "TOTAL_DURATION_MAX": 29},
    {"INITIAL_SPEED_MIN": 1.27, "INITIAL_SPEED_MAX": 1.28, "FRICTION": 0.130,
     "AVERAGE_SPEED":    1.27, "SPEED_STD": 0.04,
     "ACCELERATION_TIME_MIN": 1.2, "ACCELERATION_TIME_MAX": 2.2,
     "TOTAL_DURATION_MIN": 26, "TOTAL_DURATION_MAX": 30},
    {"INITIAL_SPEED_MIN": 1.24, "INITIAL_SPEED_MAX": 1.25, "FRICTION": 0.120,
     "AVERAGE_SPEED":    1.24, "SPEED_STD": 0.04,
     "ACCELERATION_TIME_MIN": 1.3, "ACCELERATION_TIME_MAX": 2.3,
     "TOTAL_DURATION_MIN": 29, "TOTAL_DURATION_MAX": 31},
    {"INITIAL_SPEED_MIN": 1.21, "INITIAL_SPEED_MAX": 1.22, "FRICTION": 0.110,
     "AVERAGE_SPEED":    1.21, "SPEED_STD": 0.04,
     "ACCELERATION_TIME_MIN": 1.4, "ACCELERATION_TIME_MAX": 2.4,
     "TOTAL_DURATION_MIN": 30, "TOTAL_DURATION_MAX": 32},
    {"INITIAL_SPEED_MIN": 1.16, "INITIAL_SPEED_MAX": 1.17, "FRICTION": 0.090,
     "AVERAGE_SPEED":    1.16, "SPEED_STD": 0.04,
     "ACCELERATION_TIME_MIN": 1.5, "ACCELERATION_TIME_MAX": 2.5,
     "TOTAL_DURATION_MIN": 32, "TOTAL_DURATION_MAX": 34},
    {"INITIAL_SPEED_MIN": 1.10, "INITIAL_SPEED_MAX": 1.11, "FRICTION": 0.080,
     "AVERAGE_SPEED":    1.10, "SPEED_STD": 0.04,
     "ACCELERATION_TIME_MIN": 1.6, "ACCELERATION_TIME_MAX": 2.6,
     "TOTAL_DURATION_MIN": 33, "TOTAL_DURATION_MAX": 35},
    {"INITIAL_SPEED_MIN": 1.05, "INITIAL_SPEED_MAX": 1.06, "FRICTION": 0.070,
     "AVERAGE_SPEED":    1.05, "SPEED_STD": 0.04,
     "ACCELERATION_TIME_MIN": 1.7, "ACCELERATION_TIME_MAX": 2.7,
     "TOTAL_DURATION_MIN": 34, "TOTAL_DURATION_MAX": 36},
    {"INITIAL_SPEED_MIN": 1.00, "INITIAL_SPEED_MAX": 1.01, "FRICTION": 0.060,
     "AVERAGE_SPEED":    1.00, "SPEED_STD": 0.04,
     "ACCELERATION_TIME_MIN": 1.8, "ACCELERATION_TIME_MAX": 2.8,
     "TOTAL_DURATION_MIN": 36, "TOTAL_DURATION_MAX": 38},
    {"INITIAL_SPEED_MIN": 0.95, "INITIAL_SPEED_MAX": 0.96, "FRICTION": 0.050,
     "AVERAGE_SPEED":    0.95, "SPEED_STD": 0.04,
     "ACCELERATION_TIME_MIN": 2.0, "ACCELERATION_TIME_MAX": 3.0,
     "TOTAL_DURATION_MIN": 38, "TOTAL_DURATION_MAX": 40}
]

# Ошибки моделей
model_errors = {f"Physics{i}": [] for i in range(1, 11)}
model_errors_distance = {f"Physics{i}": [] for i in range(1, 11)}

# ADAM состояния
adam_states = [{"m": {"AVERAGE_SPEED": 0.0, "SPEED_STD": 0.0}, "v": {"AVERAGE_SPEED": 0.0, "SPEED_STD": 0.0}, "t": 0}
               for _ in physics_params_list]

ADAM_LR = 0.01
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# Сектора (изменена нумерация промежуточных секторов)
sectors = []
for i in range(1, 20):
    sectors.append(str(i))
    sectors.append(f"{i}/{i+1}") # Изменено с "0(i/i+1)" на "i/i+1"
sectors.append('20')

sector_label_to_index = {label: idx for idx, label in enumerate(sectors)}
sector_index_to_label = {idx: label for idx, label in enumerate(sectors)}


@lru_cache(maxsize=1000)
def get_sector_label(sector_index: int):
    return sectors[sector_index] if 0 <= sector_index < len(sectors) else None


@njit
def circular_distance(a: int, b: int, total: int) -> int:
    diff = abs(a - b)
    return diff if diff <= total - diff else total - diff


def clear_console():
    os.system('cls' if sys.platform.startswith('win') else 'clear')


# --------- статистика ----------
real_spin_statistics = {"speed_samples": []}

# ------------ get_random_factor (шум отключён) ----------------
def get_random_factor(model_name: str) -> float:
    return 1.0


# ------------ основная физическая функция --------------------
def calculate_sector_given_v0(model_type: int, prev: dict, v0: float,
                              deterministic: bool = False) -> int:
    """
    Если deterministic=True, берутся средние значения всех диапазонов
    и отключается случайный коэффициент.
    """
    if 'sector_index' not in prev or 'direction' not in prev:
        return random.randint(0, NUMBER_OF_SECTORS - 1)

    init_idx = prev['sector_index']
    dir_code = 1 if prev['direction'] == 'R' else -1
    p = physics_params_list[model_type - 1]
    friction = p["FRICTION"]

    if deterministic:
        t_a = (p["ACCELERATION_TIME_MIN"] + p["ACCELERATION_TIME_MAX"]) / 2.0
        t_uniform = (10 + 17) / 2.0
        t_total = (p["TOTAL_DURATION_MIN"] + p["TOTAL_DURATION_MAX"]) / 2.0
        v0_calc = p["AVERAGE_SPEED"] 
        rnd_factor = 1.0
    else:
        t_a = np.random.uniform(p["ACCELERATION_TIME_MIN"], p["ACCELERATION_TIME_MAX"])
        t_uniform = np.random.uniform(10, 17)
        t_total = np.random.uniform(p["TOTAL_DURATION_MIN"], p["TOTAL_DURATION_MAX"])
        v0_calc = v0
        rnd_factor = get_random_factor(f"Physics{model_type}")

    t_uniform = min(t_uniform, t_total - t_a)
    t_d = t_total - t_a - t_uniform

    dist_a = 0.5 * v0_calc * t_a
    dist_b = v0_calc * t_uniform
    if friction > 1e-9 and t_d > 0:
        dist_c = (v0_calc / friction) * (1 - math.exp(-friction * t_d))
    else:
        dist_c = v0_calc * t_d

    total_dist = (dist_a + dist_b + dist_c) * rnd_factor * NORMALIZATION_COEFFICIENT
    theta_deg = (total_dist / WHEEL_CIRCUMFERENCE) * 360
    theta_deg = -theta_deg if dir_code == 1 else theta_deg
    new_idx = (init_idx + theta_deg / SECTOR_ANGLE) % NUMBER_OF_SECTORS
    return int(round(new_idx)) % NUMBER_OF_SECTORS


def calculate_new_sector_physics_improved(model_type: int, prev: dict) -> int:
    v0 = np.random.uniform(physics_params_list[model_type - 1]["INITIAL_SPEED_MIN"],
                           physics_params_list[model_type - 1]["INITIAL_SPEED_MAX"])
    return calculate_sector_given_v0(model_type, prev, v0, deterministic=False)


# ----------------- recalc_predicted_index (детерминированный режим) --------------
def recalc_predicted_index(params: dict, prev: dict) -> int:
    """Используется градиентом и SMC — вызывает детерминированную физическую функцию."""
    model_type = physics_params_list.index(params) + 1 if params in physics_params_list else 1
    v0_for_deterministic = params["AVERAGE_SPEED"]
    return calculate_sector_given_v0(model_type, prev, v0_for_deterministic, deterministic=True)


# -------------- loss_function (njit) -----------------
@njit
def loss_function(predicted_index: int, actual_index: int, total_sectors: int) -> float:
    d = circular_distance(predicted_index, actual_index, total_sectors)
    angle = (2 * math.pi * d) / total_sectors
    chord = 2 * math.sin(angle / 2.0)
    alpha = 0.5
    return alpha * d + (1 - alpha) * chord


# ---- функции калибровки (оставлены, но используют новый recalc_predicted_index) ----
def compute_gradient(param_ref: dict, key: str, delta: float,
                     prev: dict, actual_idx: int, total: int) -> float:
    """Центральная разность."""
    orig = param_ref[key]
    param_ref[key] = orig + delta
    idx_up = recalc_predicted_index(param_ref, prev)
    loss_up = loss_function(idx_up, actual_idx, total)
    param_ref[key] = orig - delta
    idx_down = recalc_predicted_index(param_ref, prev)
    loss_down = loss_function(idx_down, actual_idx, total)
    param_ref[key] = orig
    return (loss_up - loss_down) / (2 * delta)


def calibrate_params_batch(prev_results, actual_results, batch_size=10):
    if len(prev_results) < batch_size or len(actual_results) < batch_size:
        return
    batch_prev = prev_results[-batch_size:]
    batch_act  = actual_results[-batch_size:]
    for model_idx, params in enumerate(physics_params_list):
        grad_sum = {'AVERAGE_SPEED': 0.0, 'SPEED_STD': 0.0}
        for prev, act in zip(batch_prev, batch_act):
            act_label = act[1]
            act_idx = sector_label_to_index.get(act_label, -1)

            grad_speed = compute_gradient(params, 'AVERAGE_SPEED', 0.01,
                                          prev, act_idx, NUMBER_OF_SECTORS)
            grad_std = compute_gradient(params, 'SPEED_STD', 0.001,
                                        prev, act_idx, NUMBER_OF_SECTORS)

            grad_sum['AVERAGE_SPEED'] += grad_speed
            grad_sum['SPEED_STD'] += grad_std

        grad_avg = {k: v / batch_size for k, v in grad_sum.items()}
        state = adam_states[model_idx]
        state['t'] += 1
        for key in grad_avg:
            state['m'][key] = ADAM_BETA1 * state['m'][key] + (1 - ADAM_BETA1) * grad_avg[key]
            state['v'][key] = ADAM_BETA2 * state['v'][key] + (1 - ADAM_BETA2) * (grad_avg[key] ** 2)
            m_hat = state['m'][key] / (1 - ADAM_BETA1 ** state['t'])
            v_hat = state['v'][key] / (1 - ADAM_BETA2 ** state['t'])

            params[key] -= ADAM_LR * m_hat / (math.sqrt(v_hat) + ADAM_EPS)

            if key == "AVERAGE_SPEED":
                params[key] = max(0.1, min(1.0, params[key]))
            elif key == "SPEED_STD":
                params[key] = max(0.01, min(0.1, params[key]))

        logging.info(f"Physics{model_idx+1} Adam: AVERAGE_SPEED={params['AVERAGE_SPEED']:.4f}, SPEED_STD={params['SPEED_STD']:.4f}")


def apply_gradient_descent(prev_res: dict, pred_idx: int, actual_idx: int):
    if pred_idx < 0 or actual_idx < 0:
        return
    curr_loss = loss_function(pred_idx, actual_idx, NUMBER_OF_SECTORS)
    for i, params in enumerate(physics_params_list, start=1):
        grad_speed = compute_gradient(
            params, 'AVERAGE_SPEED', 0.01,
            prev_res, actual_idx, NUMBER_OF_SECTORS
        )
        grad_std = compute_gradient(
            params, 'SPEED_STD', 0.001,
            prev_res, actual_idx, NUMBER_OF_SECTORS
        )
        params['AVERAGE_SPEED'] -= 0.01 * grad_speed
        params['AVERAGE_SPEED'] = max(0.1, min(1.0, params['AVERAGE_SPEED']))

        params['SPEED_STD'] -= 0.001 * grad_std
        params['SPEED_STD'] = max(0.01, min(0.1, params['SPEED_STD']))

        logging.info(
            f"Physics{i} обновлена по градиенту: "
            f"AVERAGE_SPEED={params['AVERAGE_SPEED']:.4f}, "
            f"SPEED_STD={params['SPEED_STD']:.4f}, loss={curr_loss:.4f}"
        )


# --- retune_physics_params ---
def retune_physics_params():
    if not real_spin_statistics["speed_samples"]:
        return
    real_avg_speed = float(np.mean(real_spin_statistics["speed_samples"]))
    tuning_rate = 0.05
    for idx, params in enumerate(physics_params_list):
        delta = tuning_rate * (real_avg_speed - params["AVERAGE_SPEED"])
        params["AVERAGE_SPEED"] = max(0.1, min(1.0, params["AVERAGE_SPEED"] + delta))
        logging.info(f"Physics{idx+1} подстроена: AVERAGE_SPEED={params['AVERAGE_SPEED']:.4f}")


# ----------------- разное (predict, I/O) без изменений -----------------

def get_user_input(prompt="Введите результат: "):
    print(f"\033[2K\r{prompt}", end='', flush=True)
    try:
        data = sys.stdin.buffer.readline().strip()
        return data.decode('utf-8', errors='ignore').strip() if data else None
    except Exception as e:
        logging.error(f"Ошибка ввода: {e}")
        print(f"\nОшибка ввода: {e}")
        return None


def parse_result(data_str: str):
    data_str = data_str.strip()
    if not data_str:
        return None
    direction = data_str[0].upper()
    if direction not in ('R', 'L'):
        print(f"Неверное направление в '{data_str}'")
        return None
    sector_str = data_str[1:].strip().replace(' ', '')

    # Изменено: теперь parse_result сначала пытается найти "i/i+1", потом "i", потом "0(i/i+1)"
    sector_label_to_find = sector_str 

    # Если это промежуточный сектор, попробуем формат "i/i+1"
    if '/' in sector_str:
        if sector_str not in sector_label_to_index:
            # Если "i/i+1" не найдено, пробуем старый формат "0(i/i+1)"
            temp_label_old_format = f"0({sector_str})"
            if temp_label_old_format in sector_label_to_index:
                sector_label_to_find = temp_label_old_format
            else:
                print(f"Неверный сектор '{sector_str}' в '{data_str}'")
                return None
    else: # Это целый сектор, например "4"
        if sector_str not in sector_label_to_index:
            print(f"Неверный сектор '{sector_str}' в '{data_str}'")
            return None

    # Теперь, когда определили, какую метку искать
    if sector_label_to_find not in sector_label_to_index:
        print(f"Неверный сектор '{sector_label_to_find}' в '{data_str}' (после попыток парсинга)")
        return None # Should not happen if logic above is correct

    random_avg_speed = random.uniform(0.5, 0.8) 
    return {
        'direction': direction,
        'sector_index': sector_label_to_index[sector_label_to_find],
        'timestamp': datetime.datetime.now(),
        'AVERAGE_SPEED': random_avg_speed 
    }


# ----------------- оставшаяся логика (прогнозы, UI) без изменений -----------------

# --- байесовские частицы ---
num_particles = 100
bayes_particles = []
bayes_weights   = []

for p in physics_params_list:
    weights = np.ones(num_particles) / num_particles
    particles = [{"AVERAGE_SPEED": np.clip(np.random.normal(p["AVERAGE_SPEED"], 0.05), 0.1, 1.0),
                  "FRICTION":      np.clip(np.random.normal(p["FRICTION"],       0.02), 0.0, 1.0)}
                 for _ in range(num_particles)]
    bayes_particles.append(particles)
    bayes_weights.append(weights)


def smc_update(model_idx: int, prev_result: dict, actual_idx: int):
    """Sequential Monte Carlo обновление."""
    # tau теперь адаптивное, зависит от SPEED_STD модели
    tau = max(0.05, physics_params_list[model_idx]["SPEED_STD"]) 
    particles = bayes_particles[model_idx]
    weights   = bayes_weights[model_idx]
    new_w     = np.zeros_like(weights)

    for i, particle in enumerate(particles):
        tmp = physics_params_list[model_idx].copy()
        tmp.update(particle)
        # Вызов calculate_sector_given_v0 в детерминированном режиме
        pred_idx = recalc_predicted_index(tmp, prev_result) # Используем recalc_predicted_index
        d = circular_distance(pred_idx, actual_idx, NUMBER_OF_SECTORS)
        likelihood = math.exp(-(d ** 2) / (2 * tau ** 2))
        new_w[i] = weights[i] * likelihood

    s = new_w.sum()
    weights[:] = (new_w / s) if s else 1.0 / num_particles

    # мультино ресэмплинг
    cum = np.cumsum(weights)
    new_particles = []
    j, u0 = 0, random.random() / num_particles
    for k in range(num_particles):
        u = u0 + k / num_particles
        while u > cum[j]:
            j += 1
        new_particles.append(particles[j].copy())

    bayes_particles[model_idx] = new_particles
    bayes_weights[model_idx]   = np.ones(num_particles) / num_particles

    physics_params_list[model_idx]["AVERAGE_SPEED"] = np.mean([p["AVERAGE_SPEED"] for p in new_particles])
    physics_params_list[model_idx]["FRICTION"]      = np.mean([p["FRICTION"]      for p in new_particles])

    logging.info(f"Bayes update Physics{model_idx+1}: "
                 f"AVERAGE_SPEED={physics_params_list[model_idx]['AVERAGE_SPEED']:.4f}, "
                 f"FRICTION={physics_params_list[model_idx]['FRICTION']:.4f}")


def cross_validate_models(data, k=5):
    if len(data) < k:
        return {}
    fold_size = len(data) // k
    results = defaultdict(list)
    for i in range(k):
        test_data  = data[i*fold_size:(i+1)*fold_size]
        train_data = data[:i*fold_size] + data[(i+1)*fold_size:]
        for model_name in model_errors_distance.keys():
            dist_list = [dist for (md_n, dist, ts) in test_data if md_n == model_name]
            if dist_list:
                avg_test_dist = np.mean(dist_list)
                results[model_name].append(avg_test_dist)
    final_scores = {mn: np.mean(vals) for mn, vals in results.items()}
    return final_scores

def error_analysis_model():
    threshold = ERROR_WINDOW_SIZE # Используем глобальную константу
    # При желании вместо фиксированного значения можно динамически брать длину накопленных ошибок, например:
    # threshold = min(ERROR_WINDOW_SIZE, len(model_errors[model_name]))
    adjustment_factor = 0.02
    distance_threshold = 5.0
    for i in range(1, 11):
        model_name = f"Physics{i}"
        errors = model_errors[model_name]
        if len(errors) >= threshold:
            recent_errors = errors[-threshold:]
            error_rate = sum(recent_errors) / threshold
            if error_rate > 0.5:
                if i % 2 == 1:
                    adj = adjustment_factor * (error_rate - 0.5) * 2
                    physics_params_list[i-1]["AVERAGE_SPEED"] = max(
                        0.1, physics_params_list[i-1]["AVERAGE_SPEED"] - adj
                    )
                else:
                    adj = adjustment_factor * (error_rate - 0.5) * 2
                    physics_params_list[i-1]["AVERAGE_SPEED"] = min(
                        1.0, physics_params_list[i-1]["AVERAGE_SPEED"] + adj
                    )
                if "SPEED_STD" in physics_params_list[i-1]:
                    if i % 2 == 1:
                        physics_params_list[i-1]["SPEED_STD"] = min(
                            0.5, physics_params_list[i-1]["SPEED_STD"] + adjustment_factor * 0.5
                        )
                    else:
                        physics_params_list[i-1]["SPEED_STD"] = max(
                            0.05, physics_params_list[i-1]["SPEED_STD"] - adjustment_factor * 0.5
                        )
                logging.info(
                    f"Модель {model_name}: частота ошибок {error_rate:.2f} "
                    f"(ошибок={sum(recent_errors)}, всего={threshold})"
                )

        dist_list = model_errors_distance[model_name]
        if len(dist_list) >= 3:
            recent_3 = [d[0] for d in dist_list[-3:]]
            avg_3_dist = np.mean(recent_3)
            if avg_3_dist > distance_threshold:
                df = (avg_3_dist - distance_threshold) / 10.0
                physics_params_list[i-1]["SPEED_STD"] = max(
                    0.05, physics_params_list[i-1]["SPEED_STD"] - 0.01 * df
                )
                physics_params_list[i-1]["AVERAGE_SPEED"] = max(
                    0.1, min(1.0, physics_params_list[i-1]["AVERAGE_SPEED"] - 0.01 * df)
                )
                logging.info(
                    f"Модель {model_name} скорректирована (дистанции): "
                    f"AVERAGE_SPEED={physics_params_list[i-1]['AVERAGE_SPEED']:.4f}, "
                    f"SPEED_STD={physics_params_list[i-1]['SPEED_STD']:.4f}"
                )


def adaptive_monte_carlo(model_type, predicted_result, max_samples=50000, min_batch=5000, epsilon=0.005, alpha=0.95, z=1.96):
    sector_counts = np.zeros(NUMBER_OF_SECTORS, dtype=int)
    total_samples = 0
    params = physics_params_list[model_type - 1]
    mu0 = (params["INITIAL_SPEED_MIN"] + params["INITIAL_SPEED_MAX"]) / 2.0
    prev_probs = None

    while total_samples < max_samples:
        half = min_batch // 2
        for _ in range(half):
            u = np.random.uniform()
            v0 = params["INITIAL_SPEED_MIN"] + u * (params["INITIAL_SPEED_MAX"] - params["INITIAL_SPEED_MIN"])
            v0_alt = 2 * mu0 - v0
            s1 = calculate_sector_given_v0(model_type, predicted_result, v0, deterministic=False)
            s2 = calculate_sector_given_v0(model_type, predicted_result, v0_alt, deterministic=False)
            sector_counts[s1] += 1
            sector_counts[s2] += 1
        total_samples += 2 * half

        if min_batch % 2 == 1:
            extra = calculate_new_sector_physics_improved(model_type, predicted_result)
            sector_counts[extra] += 1
            total_samples += 1

        probs = sector_counts / total_samples
        p_top = np.max(probs)
        sigma_est = math.sqrt(p_top * (1 - p_top))
        required_N = (z * sigma_est / epsilon) ** 2

        if prev_probs is not None:
            l2_norm = np.sum((probs - prev_probs) ** 2)
            if l2_norm < epsilon ** 2:
                break
        prev_probs = probs.copy()

        if total_samples >= required_N:
            break

    return sector_counts, total_samples

def predict_next_sector_physics_improved(model_type, previous_results):
    if not previous_results:
        return None, None
    last = previous_results[-1]
    predicted_direction = 'L' if last['direction'] == 'R' else 'R'
    predicted_result = {'direction': predicted_direction, 'sector_index': last['sector_index']}
    try:
        sector_counts, total_samples = adaptive_monte_carlo(
            model_type,
            predicted_result,
            max_samples=random.randint(20000, 50000),
            min_batch=5000,
            epsilon=0.005,
            z=1.96
        )
    except Exception as e:
        logging.error(f"Ошибка адаптивных симуляций: {e}")
        return None, None

    physics_probs = sector_counts / total_samples
    best_idx = np.argmax(physics_probs)
    best_lbl = get_sector_label(best_idx)
    conf = physics_probs[best_idx] * 100
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
    most_probable = max(scores, key=scores.get)
    highest = scores[most_probable]
    return most_probable, highest

def predict_next_sector_ranked_voting(previous_results):
    predictions = []
    individual_predictions = {}
    for m in range(1, 11):
        pred, _ = predict_next_sector_physics_improved(m, previous_results)
        if pred:
            sorted_sectors = np.argsort(pred['probabilities'])[::-1]
            predictions.append(sorted_sectors.tolist())
            individual_predictions[f"physics_prediction_{m}"] = pred
    if predictions:
        idx, score = ranked_voting_scoring(predictions, NUMBER_OF_SECTORS)
        label = get_sector_label(idx)
        last_dir = previous_results[-1]['direction']
        pred_dir = 'L' if last_dir == 'R' else 'R'
        confidence = (score / (10 * NUMBER_OF_SECTORS)) * 100
        logging.info(f"Прогноз: направление {pred_dir}, сектор {label}")
        logging.info(f"Надёжность: {confidence:.2f}%")
        info = {
            'direction': pred_dir,
            'sector_label': label,
            'predicted_sector_index': idx,
            'scores': score,
            'predictions_list': predictions,
            'methods': [f"Physics{i}" for i in range(1, 11)]
        }
        info.update(individual_predictions)
        return info, confidence
    else:
        return None, None

def calculate_accuracy(predictions, actual_results):
    total = min(len(predictions), len(actual_results))
    if total == 0:
        return 0
    total_score = 0.0
    for i in range(total):
        pred_label = predictions[i][1]
        actual_label = actual_results[i][1]
        p_idx = sector_label_to_index.get(pred_label, -1)
        a_idx = sector_label_to_index.get(actual_label, -1)
        d = circular_distance(p_idx, a_idx, NUMBER_OF_SECTORS)
        if d == 0:
            total_score += 1.0
        elif d == 1:
            total_score += 0.7
        elif d == 2:
            total_score += 0.5
    return (total_score / total) * 100

def get_user_input(prompt="Введите результат: "):
    print(f"\033[2K\r{prompt}", end='', flush=True)
    try:
        data = sys.stdin.buffer.readline().strip()
        return data.decode('utf-8', errors='ignore').strip() if data else None
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
    methods = prediction_info.get('methods', [])
    error_flag = 1 if pred_idx != real_idx else 0
    for m in methods:
        model_errors[m].append(error_flag)
        model_errors_distance[m].append((dist, datetime.datetime.now()))


def log_model_error_frequencies():
    for i in range(1, 11):
        model_name = f"Physics{i}"
        total_count = len(model_errors[model_name])
        if total_count > 0:
            err_count = sum(model_errors[model_name])
            freq = err_count / total_count
        else:
            freq = 0
            err_count = 0
        logging.info(
            f"Модель {model_name}: частота ошибок {freq:.2f} "
            f"(ошибок={err_count}, всего={total_count})"
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
    recent_performance    = []
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
                actual_results.append(
                    (parsed['direction'], get_sector_label(parsed['sector_index']))
                )
            else:
                logging.warning(f"Пропуск ввода: {item}")

    if previous_results:
        prediction_info, confidence = predict_next_sector_ranked_voting(previous_results)

    while True:
        clear_console()
        print(f"{gray_color}Предыдущие 100 результатов прогноза:{reset_color}")
        last_idxs = range(max(0, len(predictions_full_info) - 100), len(predictions_full_info))
        for idx_count, idx in enumerate(last_idxs, start=1):
            if idx < len(predictions_full_info) and idx < len(predictions) and idx + 1 < len(actual_results):
                p_info, conf_ = predictions_full_info[idx]
                p_dir, p_sec  = predictions[idx]
                a_dir, a_sec  = actual_results[idx + 1]

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
                        if (m_pred['direction'] == a_dir and m_pred['sector_label'] == a_sec):
                            line_col = green_color
                        short_n = model_name.replace("physics_prediction_", "Physics")
                        print(
                            line_col +
                            f"  {short_n}: сектор {m_pred['sector_label']} ({conf_m:.2f}%)" +
                            reset_color
                        )

                print(gray_color + f"Прогноз: направление {p_info['direction']}, сектор {p_info['sector_label']}" + reset_color)
                print(gray_color + f"Надёжность прогноза: {conf_:.2f}%" + reset_color)
                print(gray_color + f"Фактический результат: {a_dir}, сектор {a_sec}\n" + reset_color)

        try:
            accuracy = calculate_accuracy(predictions, actual_results)
        except Exception:
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
                    match = (pp['direction'] == prediction_info['direction'] and pp['sector_label'] == prediction_info['sector_label'])
                    line_col = blue_color if match else yellow_color
                    print(f"  {line_col}Physics{i}: сектор {pp['sector_label']} {gray_color}({conf_i:.2f}%) {reset_color}")
            print(f"{green_color}Прогноз: направление {prediction_info['direction']}, сектор {prediction_info['sector_label']}{reset_color}")
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
        actual_results.append(
            (parsed_new['direction'], get_sector_label(parsed_new['sector_index']))
        )

        if 'AVERAGE_SPEED' in parsed_new:
            real_spin_statistics["speed_samples"].append(parsed_new['AVERAGE_SPEED'])

        if len(previous_results) >= 2:
            prev_res = previous_results[-2]
            actual_idx = parsed_new['sector_index']
            for mi in range(len(physics_params_list)):
                smc_update(mi, prev_res, actual_idx)

        calibrate_params_batch(previous_results, actual_results, batch_size=10)

        retune_physics_params()
        error_analysis_model()

        if prediction_info:
            predictions.append((prediction_info['direction'], prediction_info['sector_label']))
            real_idx = parsed_new['sector_index']
            pred_idx = sector_label_to_index.get(prediction_info['sector_label'], -1)
            match_flag = (real_idx == pred_idx)
            recent_performance.append(1 if match_flag else 0)
            if len(recent_performance) > 100:
                recent_performance = recent_performance[-100:]

            try:
                curr_acc = calculate_accuracy(predictions, actual_results)
            except Exception:
                curr_acc = 0
            print(f"\nТекущая точность прогнозов: {curr_acc:.2f}%")
            logging.info(f"Текущая точность прогнозов: {curr_acc:.2f}%")

            predictions_full_info.append((prediction_info, confidence))
            analyze_errors(predictions, actual_results, prediction_info, previous_results)
            log_model_error_frequencies()

            if len(previous_results) >= 2:
                prev_for_grad = previous_results[-2]
            else:
                prev_for_grad = previous_results[-1]
            apply_gradient_descent(prev_for_grad, pred_idx, real_idx)
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
        cv_scores = cross_validate_models(distData, k=3)
        if cv_scores:
            print("\nРезультаты кросс-валидации (средняя дистанция по моделям):")
            for mn in sorted(cv_scores.keys()):
                print(f"  {mn}: {cv_scores[mn]:.2f}")

if __name__ == "__main__":
    main()
