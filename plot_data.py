import matplotlib.pyplot as plt
import numpy as np


def plot_recommendation_results():
    title = "Recommendation performance"
    x = [0, 0.2, 0.4, 0.6]
    y1 = [0.110, 0.0982, 0.0805, 0.0629]  # nDCG score for ex 1
    y2 = [0.110, 0.0936, 0.0777, 0.0672]  # nDCG score for ex 2
    y3 = [0.110, 0.0938, 0.0774, 0.0566]  # nDCG score for ex 3
    y4 = [0.110, 0.0970, 0.0820, 0.0675]  # nDCG score for ex 4
    y5 = [0.110, 0.0942, 0.0796, 0.0702]  # nDCG score for ex 5
    y6 = [0.110, 0.0942, 0.0786, 0.0609]  # nDCG score for ex 6
    y7 = [0.110, 0.100, 0.0888, 0.0739] # nDCG score for ex 7

    plt.plot(x, y1, label='Ex. 1')
    plt.plot(x, y2, label='Ex. 2')
    plt.plot(x, y3, label='Ex. 3')
    plt.plot(x, y4, label='Ex. 4')
    plt.plot(x, y5, label='Ex. 5')
    plt.plot(x, y6, label='Ex. 6')
    plt.plot(x, y7, label ='Ex. 7')
    plt.xticks(np.arange(min(x), max(x) + 0.1, 0.2))
    plt.xlabel("Obfuscation proportion")
    plt.ylabel("nDCG score")
    plt.legend()
    plt.title(title)
    plt.show()


def plot_attack_performance():
    title = "Attack performance"
    x = [0, 0.2, 0.4, 0.6]
    y1 = [0.7212, 0.6111, 0.5111, 0.4250]  # F1 score for ex 1
    y2 = [0.7212, 0.7071, 0.7789, 0.6988]  # F1 score for ex 2
    y3 = [0.7212, 0.7129, 0.6829, 0.4722]  # F1 score for ex 3
    y4 = [0.7212, 0.6538, 0.5053, 0.3232]  # F1 score for ex 4
    y5 = [0.7212, 0.7525, 0.7000, 0.6818]  # F1 score for ex 5
    y6 = [0.7212, 0.7255, 0.6875, 0.5238]  # F1 score for ex 6
    y7 = [0.7212, 0.6095, 0.4615, 0.3614] # F1 score for ex 7

    plt.plot(x, y1, label='Ex. 1')
    plt.plot(x, y2, label='Ex. 2')
    plt.plot(x, y3, label='Ex. 3')
    plt.plot(x, y4, label='Ex. 4')
    plt.plot(x, y5, label='Ex. 5')
    plt.plot(x, y6, label='Ex. 6')
    plt.plot(x, y7, label ='Ex. 7')

    plt.xticks(np.arange(min(x), max(x) + 0.1, 0.2))
    plt.xlabel("Obfuscation proportion")
    plt.ylabel("F1 score")
    plt.legend()
    plt.title(title)
    plt.show()

plot_recommendation_results()
plot_attack_performance()
