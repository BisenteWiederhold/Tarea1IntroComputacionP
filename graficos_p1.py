import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("resultados.csv")

# Limpiar datos inválidos
df["clasica"] = df["clasica"].replace(-1, None)
df["strassen"] = df["strassen"].replace(-1, None)

sizes = sorted(df["n"].unique())

# TIEMPO vs n 
plt.figure()

# Clásica (tomar un solo valor por n)
clasica = df.dropna().drop_duplicates(subset=["n"])[["n", "clasica"]].set_index("n")

# Strassen
strassen = df.dropna().drop_duplicates(subset=["n"])[["n", "strassen"]].set_index("n")

# Bloques (mejor b)
bloques = df.loc[df.groupby("n")["bloques"].idxmin()].set_index("n")["bloques"]

plt.plot(clasica.index, clasica["clasica"], marker='o', label="Clásica")
plt.plot(bloques.index, bloques.values, marker='o', label="Bloques (mejor b)")
plt.plot(strassen.index, strassen["strassen"], marker='o', label="Strassen")

plt.xlabel("n")
plt.ylabel("Tiempo (s)")
plt.title("Tiempo vs Tamaño de matriz")
plt.legend()
plt.grid()

plt.savefig("parte1_tiempo_vs_n.png")
plt.close()

#BLOQUES vs b 
for n in sizes:
    data = df[df["n"] == n]

    plt.figure()
    plt.plot(data["b"], data["bloques"], marker='o')

    plt.title(f"Bloques: Tiempo vs b (n={n})")
    plt.xlabel("b")
    plt.ylabel("Tiempo (s)")
    plt.grid()

    plt.savefig(f"parte1_bloques_n{n}.png")
    plt.close()

print("Gráficos corregidos generados.")