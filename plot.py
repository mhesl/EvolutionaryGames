from util import load_generation
import matplotlib.pyplot as plt


if __name__ == "__main__":
    SIZE = 30
    PATH_THRUST = 'checkpoint/thrust/'
    PATH_HELICOPTER = 'checkpoint/helicopter/'
    input_param = input("press 1 for helicopter or 2 for thrust")
    PATH = PATH_HELICOPTER if input_param == 1 else PATH_THRUST
    mins = []
    maxs = []
    avgs = []
    for i in range(2, SIZE + 1):
        players = load_generation(PATH + str(i), '')
        fitnesses = [player.fitness for player in players]
        mins.append(min(fitnesses))
        maxs.append(max(fitnesses))
        avgs.append(sum(fitnesses)/len(fitnesses))

    dmins = [mins[i] - mins[i - 1] for i in range(1, SIZE - 1)]
    dmaxs = [maxs[i] - maxs[i - 1] for i in range(1, SIZE - 1)]
    davgs = [avgs[i] - avgs[i - 1] for i in range(1, SIZE - 1)]

    x = range(2, SIZE + 1)
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title("generation")
    plt.plot(x, mins, label="min")
    plt.plot(x, maxs, label="max")
    plt.plot(x, avgs, label="avg")
    plt.plot(x[1:], dmins, label="delta min")
    plt.plot(x[1:], dmaxs, label="delta max")
    plt.plot(x[1:], davgs, label="delta avg")

    plt.legend()
    plt.grid()
    plt.show()

