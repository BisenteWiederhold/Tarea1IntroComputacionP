import pandas as pd
import matplotlib.pyplot as plt

# Leer datos
df = pd.read_csv("resultados.csv")

# Obtener tamaños únicos
sizes = sorted(df["n"].unique())
blocks = sorted(df["b"].unique())

# TIEMPO vs n (comparación algoritmos)
plt.figure()

# Promediamos por n (ignorando b para clásica y Strassen)
clasica = df.groupby("n")["clasica"].mean()
strassen = df.groupby("n")["strassen"].mean()

# Para bloques, tomamos el mejor b (mínimo tiempo por n)
bloques = df.loc[df.groupby("n")["bloques"].idxmin()].set_index("n")["bloques"]

plt.plot(clasica.index, clasica.values, marker='o', label="Clásica")
plt.plot(bloques.index, bloques.values, marker='o', label="Bloques (mejor b)")
plt.plot(strassen.index, strassen.values, marker='o', label="Strassen")

plt.title("Tiempo vs Tamaño de matriz")
plt.xlabel("n")
plt.ylabel("Tiempo (s)")
plt.legend()
plt.grid()

plt.savefig("parte1_tiempo_vs_n.png")
plt.close()

# TIEMPO vs b (solo bloques)
for n in sizes:
    data = df[df["n"] == n]

    plt.figure()
    plt.plot(data["b"], data["bloques"], marker='o')

    plt.title(f"Bloques: Tiempo vs tamaño de bloque (n={n})")
    plt.xlabel("Tamaño de bloque (b)")
    plt.ylabel("Tiempo (s)")
    plt.grid()

    plt.savefig(f"parte1_bloques_n{n}.png")
    plt.close()

print("Gráficos Parte 1 generados correctamente.")