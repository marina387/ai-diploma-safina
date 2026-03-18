def plot_bootstrap_hist(bootstrap_means: list[float], ci_lower: float, ci_upper: float):
    """
    Визуализация бутстрэп-распределения средних
    (для src/plots.py)
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_means, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(ci_lower, color='red', linestyle='--', linewidth=2, label=f'CI lower: {ci_lower:.3f}')
    plt.axvline(ci_upper, color='red', linestyle='--', linewidth=2, label=f'CI upper: {ci_upper:.3f}')
    plt.axvline(np.mean(bootstrap_means), color='green', linestyle='-', linewidth=2, label='mean of means')
    
    plt.title('Bootstrap распределение среднего')
    plt.xlabel('Среднее значение')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()