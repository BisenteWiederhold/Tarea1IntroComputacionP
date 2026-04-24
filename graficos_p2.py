import pandas as pd
import matplotlib.pyplot as plt

# Leer datos
df = pd.read_csv("resultados_speedup.csv")

# Obtener tamaños únicos
sizes = df["n"].unique()


# SPEEDUP
for n in sizes:
    data = df[df["n"] == n]

    plt.figure()
    plt.plot(data["threads"], data["speedup_bloques"], marker='o', label="Bloques")
    plt.plot(data["threads"], data["speedup_strassen"], marker='o', label="Strassen")

    plt.title(f"Speedup vs Threads (n={n})")
    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.legend()
    plt.grid()

    plt.savefig(f"speedup_n{n}.png")
    plt.close()


# EFICIENCIA

for n in sizes:
    data = df[df["n"] == n]

    plt.figure()
    plt.plot(data["threads"], data["eficiencia_bloques"], marker='o', label="Bloques")
    plt.plot(data["threads"], data["eficiencia_strassen"], marker='o', label="Strassen")

    plt.title(f"Eficiencia vs Threads (n={n})")
    plt.xlabel("Threads")
    plt.ylabel("Eficiencia")
    plt.legend()
    plt.grid()

    plt.savefig(f"eficiencia_n{n}.png")
    plt.close()

# TIEMPOS
for n in sizes:
    data = df[df["n"] == n]

    plt.figure()
    plt.plot(data["threads"], data["bloques_time"], marker='o', label="Bloques")
    plt.plot(data["threads"], data["strassen_time"], marker='o', label="Strassen")

    plt.title(f"Tiempo vs Threads (n={n})")
    plt.xlabel("Threads")
    plt.ylabel("Tiempo (s)")
    plt.legend()
    plt.grid()

    plt.savefig(f"tiempo_n{n}.png")
    plt.close()

print("Gráficos generados correctamente.")