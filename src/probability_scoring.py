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


def bayes_posterior(prior: float, likelihood: float, evidence: float) -> float:
    """
    Формула Байеса: P(A|B) = P(B|A) * P(A) / P(B)
    
    prior = P(A)
    likelihood = P(B|A)
    evidence = P(B)
    """
    for name, p in [("prior", prior), ("likelihood", likelihood), ("evidence", evidence)]:
        if p < 0 or p > 1:
            raise ValueError(f"bayes_posterior: {name} must be in [0,1]")
    if evidence == 0:
        raise ValueError("bayes_posterior: evidence must be > 0")
    return (likelihood * prior) / evidence


def score_buy_probability(recs: list[dict], clicked_value: int) -> float:
    """
    Вероятность покупки при заданном значении clicked
    Для HR: вероятность успешного найма при наличии/отсутствии навыка
    """
    if clicked_value not in (0, 1):
        raise ValueError("clicked_value must be 0/1")
    subset = [r for r in recs if int(r["clicked"]) == clicked_value]
    if len(subset) == 0:
        raise ValueError("No records for clicked_value")
    bought_count = sum(1 for r in subset if int(r["bought"]) == 1)
    return bought_count / len(subset)


def laplace_smooth_prob(successes: int, trials: int) -> float:
    """
    Сглаживание Лапласа: P = (successes + 1) / (trials + 2)
    Чтобы избежать нулевых вероятностей для редких событий
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("laplace_smooth_prob: invalid counts")
    return (successes + 1) / (trials + 2)