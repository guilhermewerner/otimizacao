import matplotlib.pyplot as plt

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

plt.figure(figsize=(6.4, 4.8))
plt.plot(r_values, x1_values, label="x1", marker="o", linestyle="-")
plt.plot(r_values, x2_values, label="x2", marker="o", linestyle="-")
plt.title("Valores de x1 e x2 por r")
plt.xlabel("r")
plt.ylabel("Valores de x1 e x2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_x1_x2.png")

plt.figure(figsize=(6.4, 4.8))
plt.plot(r_values, S_values, label="S", marker="o", linestyle="-", color="red")
plt.title("Solução S por r")
plt.xlabel("r")
plt.ylabel("S")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_S.png")
