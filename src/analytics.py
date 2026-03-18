def contingency_4x4(recs: list[dict], a_key: str, b_key: str) -> list[list[int]]:
    """Таблица сопряженности 4x4 для признаков со значениями 0,1,2,3"""
    table = [[0, 0, 0, 0] for _ in range(4)]
    for r in recs:
        a = int(r[a_key])
        b = int(r[b_key])
        if a not in (0, 1, 2, 3) or b not in (0, 1, 2, 3):
            raise ValueError("contingency_4x4: values must be 0, 1, 2, or 3")
        table[a][b] += 1
    return table


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
    clicked_value должен быть 0 или 1
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
    Чтобы избежать нулевых вероятностей
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("laplace_smooth_prob: invalid counts")
    return (successes + 1) / (trials + 2)

