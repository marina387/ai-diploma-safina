import numpy as np
import matplotlib.pyplot as plt

# Генерируем 1000 случайных чисел (это как создание эмбеддингов)
data = np.random.normal(0, 1, 1000)

# Проверяем количество элементов
len(data)

# Среднее значение (для нормализации данных)
mean = np.mean(data)
std = np.std(data)

# Медиана (устойчива к выбросам)
median = np.median(data_out)  # data_out = np.append(data, 10)


# Процент попадания в интервал (для анализа оценок схожести)
lower = mean - std
upper = mean + std
within = np.sum((data >= lower) & (data <= upper))
within / len(data)  # ~68% для нормального распределения


# Гистограмма для анализа распределения (например, оценок схожести)
plt.hist(data, bins=30)
plt.title("Distribution")
plt.show()

# Сравнение распределений
plt.hist(uniform, bins=30)
plt.title("Uniform Distribution")
plt.show()

# Сравнение средних разных распределений
np.mean(data), np.mean(uniform)

# Проверки, которые пригодятся при отладке
assert len(data) == 1000
assert isinstance(mean, float)
assert isinstance(std, float)
print("✅ All tests passed")
