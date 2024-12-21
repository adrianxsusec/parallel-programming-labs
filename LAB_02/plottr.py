import matplotlib.pyplot as plt

num_files = 3

all_measurements = []
for m in range(1, num_files + 1):
    with open(f"report/res_depth_{m}.txt") as f:
        meas = [float(line) for line in f.readlines()]
        all_measurements.append(meas)

for i, measurements in enumerate(all_measurements):
    speedup = [measurements[0] / m for m in measurements]
    efficiency = [s / p for s, p in zip(speedup, range(1, len(measurements) + 1))]

    plt.figure()
    plt.grid()
    plt.plot(range(2, len(measurements) + 2), speedup)
    plt.plot(range(2, len(measurements) + 2), range(1, len(measurements) + 1), label="Idealno ubrzanje", linestyle="--")
    plt.xlabel("Broj procesora")
    plt.ylabel("Ubrzanje")
    plt.title(f"Broj zadataka: {7 ** (i + 1)}, DFS dubina: {8 - (i + 1)}")
    plt.savefig(f"speedup_{i + 1}.png")

    plt.figure()
    plt.grid()
    plt.plot(range(2, len(measurements) + 2), efficiency, linestyle='--')
    plt.xlabel("Broj procesora")
    plt.ylabel("Ucinkovitost")
    plt.ylim(0, 1)
    plt.title(f"Broj zadataka: {7 ** (i + 1)}, DFS dubina: {8 - (i + 1)}")
    plt.savefig(f"efficiency_{i + 1}.png")

    plt.clf()
