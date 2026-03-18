# prob_event - расчет базовых вероятностей
def prob_event(count_A: int, n: int) -> float:
    """P(A) = count(A)/n"""
    if n <= 0:
        raise ValueError("prob_event: n must be > 0")
    if count_A < 0 or count_A > n:
        raise ValueError("prob_event: invalid count")
    return count_A / n

# prob_conditional - условные вероятности
def prob_conditional(count_A_and_B: int, count_B: int) -> float:
    """P(A|B) = count(A∩B)/count(B)"""
    if count_B <= 0:
        raise ValueError("prob_conditional: count_B must be > 0")
    if count_A_and_B < 0 or count_A_and_B > count_B:
        raise ValueError("prob_conditional: invalid intersection count")
    return count_A_and_B / count_B

# is_independent_by_counts - проверка независимости событий
def is_independent_by_counts(p_a: float, p_a_given_b: float, tol: float = 0.05) -> bool:
    """Проверка независимости по приближению |P(A|B)-P(A)| <= tol"""
    return abs(p_a_given_b - p_a) <= tol

# contingency_4x4 - таблица сопряженности
def contingency_4x4(recs: list[dict], a_key: str, b_key: str) -> list[list[int]]:
    """4x4 таблица частот для признаков a_key и b_key (значения 0/1/2/3)."""
    table = [[0, 0, 0, 0] for _ in range(4)]
    for r in recs:
        a = int(r[a_key])
        b = int(r[b_key])
        if a not in (0, 1, 2, 3) or b not in (0, 1, 2, 3):
            raise ValueError("contingency_4x4: values must be 0, 1, 2, or 3")
        table[a][b] += 1
    return table

# simulate_click_buy - Монте-Карло симуляция
def simulate_click_buy(n: int, p_click: float, p_buy_click0: float, p_buy_click1: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    clicked = rng.random(n) < p_click
    probs = np.where(clicked, p_buy_click1, p_buy_click0)
    bought = rng.random(n) < probs
    return clicked, bought

# estimate_p_buy_given_click1 - множественные оценки Монте-Карло
def estimate_p_buy_given_click1(n: int, seed: int) -> float:
    clicked_sim, bought_sim = simulate_click_buy(
        n=n, p_click=0.6, p_buy_click0=0.05, p_buy_click1=0.25, seed=seed
    )
    count_click1 = int(clicked_sim.sum())
    count_buy_and_click1 = int((bought_sim & clicked_sim).sum())
    return count_buy_and_click1 / count_click1

# Визуализация
import matplotlib.pyplot as plt

# Пример визуализации
for click in [0, 1, 2, 3]:
    total = sum(1 for r in records if r["clicked"] == click)
    bought = sum(1 for r in records if r["clicked"] == click and r["bought"] > 0)
    prob = bought / total if total > 0 else 0
    plt.bar(f"click={click}", prob)

plt.ylim(0, 1)
plt.title("Вероятность покупки при разном количестве кликов")
plt.show()