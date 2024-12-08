import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

r_values = [1, 2, 3, 4, 5]
x1_values = [0.9999999999999818, 2.0, 3.0, 3.999999999994655, 4.999999999971521]
x2_values = [1.0599999999999872, 2.12, 3.18, 4.239999999991208, 5.299999999956182]
S_values = [
    2.5299999999999656,
    10.120000000000001,
    22.77,
    40.479999999949705,
    63.24999999967259,
]

fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(r_values, x1_values, S_values, label="x1", color="blue", marker="o")
ax.scatter(r_values, x2_values, S_values, label="x2", color="green", marker="^")

ax.set_title("Visualização 3D de r, x1, x2 e S")
ax.set_xlabel("r")
ax.set_ylabel("x1 / x2")
ax.set_zlabel("S")
ax.legend()

plt.tight_layout()
plt.savefig("grafico_3D_r_x1_x2_S.png")
