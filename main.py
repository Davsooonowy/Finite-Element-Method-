import numpy as np
import seaborn as sns
from scipy.integrate import quad as integration
import matplotlib.pyplot as plt


def E(x):
    return 3 if x <= 1 else 5


def e(n, i, x):
    h = 2 / n
    return max(0, 1 - abs((x / h - i)))


def e_prim(n, i, x):
    h = 2 / n
    if x <= (i - 1) * h or x >= (i + 1) * h:
        return 0
    else:
        return 1 / h if x <= i * h else -1 / h


def calculate_integral(n, i, j):
    start = 2 * max(max(i, j) - 1, 0) / n
    end = 2 * min(min(i, j) + 1, n) / n
    return integration(lambda x: E(x) * e_prim(n, i, x) * e_prim(n, j, x), start, end)[0] if abs(j - i) <= 1 else 0


def fill(n):
    B, L = np.zeros((n, n)), np.zeros(n)
    L[0] = -30 * e(n, 0, 0)
    for i in range(n):
        for j in range(n):
            integral = calculate_integral(n, i, j)
            B[i, j] = -3 * e(n, i, 0) * e(n, j, 0) + integral
    return B, L


def show_plot(solution, n):
    sns.lineplot(x=np.linspace(0, 2, n + 1), y=solution)
    plt.title('Elastic deformation plot')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.grid(True)
    plt.savefig('elastic_deformation_plot.png')
    plt.show()


if __name__ == '__main__':
    user_input = int(input("Input n: "))
    B, L = fill(user_input)
    show_plot(np.concatenate((np.linalg.solve(B, L), [0])), user_input)