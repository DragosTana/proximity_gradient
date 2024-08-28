import numpy as np
import matplotlib.pyplot as plt

times = np.load("times_features.npy")
my_times = np.load("my_times_features_cd.npy")

print(times)

#remove top 5 worst and best times
times = np.sort(times, axis=2)[:, :, 5:-5]
my_times = np.sort(my_times, axis=2)[:, :, 5:-5]


dataset_sizes = [100, 500, 1000, 5000, 10000, 100000, 500000]
lambda_ = [1, 5, 10]
features = [10, 20, 50, 100, 500]

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# plotting the average time for sklearn
average_times_sk = np.mean(times, axis=2)

for i, l in enumerate(features):
    ax[1].plot(dataset_sizes, average_times_sk[i], label=f"features={l}")
ax[1].set_xlabel("Dataset size")
ax[1].set_ylabel("Time (ns)")
ax[1].set_title("Average time for sklearn")
ax[1].grid()
ax[1].legend()

# plotting the average time for my_lr
average_times_mine = np.mean(my_times, axis=2)
for i, l in enumerate(features):
    ax[0].plot(dataset_sizes, average_times_mine[i], label=f"features={l}")
ax[0].set_xlabel("Dataset size")
ax[0].set_ylabel("Time (ns)")
ax[0].set_title("Average time for Coordinate Descent")
ax[0].grid()
ax[0].legend()

#speedup
speedup = average_times_sk / average_times_mine
for i, l in enumerate(features):
    ax[2].plot(dataset_sizes, speedup[i], label=f"features={l}")
ax[2].set_xlabel("Dataset size")
ax[2].set_ylabel("Speedup")
ax[2].set_title("Speedup")
ax[2].grid()
ax[2].legend()

plt.show()
