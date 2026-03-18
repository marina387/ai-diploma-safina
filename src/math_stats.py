def mean(values: list[float]) -> float:
    """Среднее арифметическое. Требует непустой список."""
    if len(values) == 0: 
        raise ValueError("Нельзя вычислить среднее для пустого списка")
    return sum(values) / len(values)


print("mean =", mean(sample))


def median(values: list[float]) -> float:
    """Медиана. Требует непустой список."""
    if not values:
        raise ValueError("median: empty list")
    sorted_values = sorted(values)
    n = len(sorted_values)
    middle = n // 2
    if n % 2 == 1:
        # Для нечетного n: элемент посередине
        return float(sorted_values[middle])
    else:
       # Для четного n: среднее двух центральных
        return (sorted_values[middle - 1] + sorted_values[middle]) / 2
    

print("median =", median(sample))



def variance_sample(values: list[float]) -> float:
    """Выборочная дисперсия (деление на n-1)."""
    n = len(values)
    if n < 2:
        raise ValueError("Нужно минимум 2 элемента")
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (n - 1)

print("variance_sample =", variance_sample(sample))




def std_sample(values: list[float]) -> float:
    """Выборочное стандартное отклонение."""     
    var = variance_sample(values)
    return var ** 0.5

print("std_sample =", std_sample(sample))


def trimmed_mean(values: list[float], k: int = 1) -> float:
    """Усечённое среднее: убрать k минимальных и k максимальных."""
    n = len(values)
    if not values:
        raise ValueError("Список не может быть пустым")
    if 2 * k >= n:
        raise ValueError("k слишком большое")
    sorted_values = sorted(values)
    core = sorted_values[k:n-k] 
    return mean(core)

print("trimmed_mean(before) =", round(trimmed_mean(sample, k=1), 3))
print("trimmed_mean(after)  =", round(trimmed_mean(sample_out, k=1), 3))




def prob_event(count_A: int, n: int) -> float:
    """Вероятность события P(A) = count(A) / n"""
    if n <= 0:
        raise ValueError("prob_event: n must be > 0")
    if count_A < 0 or count_A > n:
        raise ValueError("prob_event: invalid count")
    return count_A / n


def prob_conditional(count_A_and_B: int, count_B: int) -> float:
    """Условная вероятность P(A|B) = count(A ∩ B) / count(B)"""
    if count_B <= 0:
        raise ValueError("prob_conditional: count_B must be > 0")
    if count_A_and_B < 0 or count_A_and_B > count_B:
        raise ValueError("prob_conditional: invalid intersection count")
    return count_A_and_B / count_B


def is_independent_by_counts(p_a: float, p_a_given_b: float, tol: float = 0.05) -> bool:
    """Проверка независимости: P(A|B) ≈ P(A)"""
    return abs(p_a_given_b - p_a) <= tol





# Функции из занятия 4: формула Байеса и наивный скоринг

def build_binary_counts(recs: list[dict], a_key: str, b_key: str) -> dict:
    """Подсчёт частот для двух бинарных признаков"""
    n = len(recs)
    count_A = 0
    count_B = 0
    count_A_and_B = 0

    for r in recs:
        a = int(r[a_key])
        b = int(r[b_key])
        if a not in (0, 1) or b not in (0, 1):
            raise ValueError("build_binary_counts: values must be 0/1")
        if a == 1:
            count_A += 1
        if b == 1:
            count_B += 1
        if a == 1 and b == 1:
            count_A_and_B += 1

    return {"n": n, "count_A": count_A, "count_B": count_B, "count_A_and_B": count_A_and_B}


def prob_from_counts(count: int, n: int) -> float:
    """Вероятность из частоты: P = count / n"""
    if n <= 0:
        raise ValueError("prob_from_counts: n must be > 0")
    if count < 0 or count > n:
        raise ValueError("prob_from_counts: invalid count")
    return count / n


def prob_conditional(count_A_and_B: int, count_A: int) -> float:
    """Условная вероятность P(B|A) = count(A∩B) / count(A)"""
    if count_A <= 0:
        raise ValueError("prob_conditional: condition count must be > 0")
    if count_A_and_B < 0 or count_A_and_B > count_A:
        raise ValueError("prob_conditional: invalid intersection count")
    return count_A_and_B / count_A



# Функции из занятия 5: доверительные интервалы и бутстрэп

import numpy as np
from typing import List, Tuple


def mean(values: list[float]) -> float:
    """Среднее арифметическое"""
    if len(values) == 0:
        raise ValueError("mean: empty list")
    return sum(values) / len(values)


def std_sample(values: list[float]) -> float:
    """Выборочное стандартное отклонение (деление на n-1)"""
    n = len(values)
    if n < 2:
        raise ValueError("std_sample: need at least 2 values")
    m = mean(values)
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return var ** 0.5


def ci_mean_normal_approx(values: list[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Приближённый доверительный интервал для среднего:
    mean ± z * (std / sqrt(n))
    
    confidence: уровень доверия (0.95 → z≈1.96)
    """
    if not values:
        raise ValueError("ci_mean_normal_approx: empty list")
    if confidence <= 0 or confidence >= 1:
        raise ValueError("ci_mean_normal_approx: confidence must be in (0,1)")
    
    n = len(values)
    m = mean(values)
    sem = std_sample(values) / (n ** 0.5)  # Standard Error of the Mean
    
    # z-значения для разных уровней доверия
    z_map = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }
    
    # Берём ближайшее значение или используем 1.96 по умолчанию
    z = z_map.get(confidence, 1.96)
    
    margin = z * sem
    return (m - margin, m + margin)


def bootstrap_ci_mean(
    values: list[float], 
    n_bootstrap: int = 1000, 
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Bootstrap доверительный интервал для среднего
    
    - n_bootstrap: количество псевдовыборок
    - confidence: уровень доверия (например, 0.95)
    
    Возвращает (lower_bound, upper_bound)
    """
    if not values:
        raise ValueError("bootstrap_ci_mean: empty list")
    if n_bootstrap < 100:
        raise ValueError("bootstrap_ci_mean: n_bootstrap too small")
    
    rng = np.random.default_rng(seed)
    n = len(values)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        # Выборка с возвращением того же размера
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means.append(mean(sample))
    
    # Квантили
    alpha = (1 - confidence) / 2
    lower = np.quantile(bootstrap_means, alpha)
    upper = np.quantile(bootstrap_means, 1 - alpha)
    
    return (float(lower), float(upper))
