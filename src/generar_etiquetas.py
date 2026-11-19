import os
import math
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_points(instances_dir: str, instance_id: str) -> list[tuple[float, float]]:
    """Carga las coordenadas (x, y) de una instancia TSP desde su CSV."""
    path = os.path.join(instances_dir, f"{instance_id}.csv")
    df = pd.read_csv(path)
    return list(zip(df["x"].tolist(), df["y"].tolist()))


def compute_dist_matrix(points: list[tuple[float, float]]) -> np.ndarray:
    """Precalcula la matriz de distancias euclidianas n x n."""
    pts = np.array(points, dtype=np.float64)
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def tour_length_mat(dmat: np.ndarray, route: list[int]) -> float:
    """Calcula la distancia total del ciclo cerrado usando la matriz de distancias."""
    n = len(route)
    return float(sum(dmat[route[i], route[(i + 1) % n]] for i in range(n)))


def nearest_neighbor_mat(dmat: np.ndarray, start_node: int = 0) -> list[int]:
    """Heurística constructiva del Vecino Más Cercano (Tarea 2, Ejercicio 3)."""
    n = dmat.shape[0]
    unvisited = set(range(n)) - {start_node}
    route = [start_node]
    curr = start_node
    while unvisited:
        nxt = min(unvisited, key=lambda city: dmat[curr, city])
        route.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
    return route


def two_opt_local_search_mat(
    dmat: np.ndarray, initial_route: list[int]
) -> tuple[list[int], float]:
    """Búsqueda local 2-opt Hill Climbing hasta alcanzar óptimo local (Tarea 2, Ej. 3)."""
    best_route = list(initial_route)
    n = len(best_route)
    if n < 4:
        return best_route, tour_length_mat(dmat, best_route)

    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            a = best_route[i - 1]
            b = best_route[i]
            dab = dmat[a, b]
            for k in range(i + 1, n):
                c = best_route[k]
                d = best_route[(k + 1) % n]
                delta = (dmat[a, c] + dmat[b, d]) - (dab + dmat[c, d])
                if delta < -1e-12:
                    best_route[i : k + 1] = best_route[i : k + 1][::-1]
                    improved = True
                    break
            if improved:
                break

    return best_route, tour_length_mat(dmat, best_route)


def mst_prim_mat(dmat: np.ndarray) -> dict[int, list[int]]:
    """Algoritmo de Prim desde el nodo 0 para construir el MST (Tarea 2, Ej. 5)."""
    n = dmat.shape[0]
    best = {v: (float(dmat[0, v]), 0) for v in range(1, n)}
    children = {v: [] for v in range(n)}
    while best:
        v = min(best, key=lambda u: best[u][0])
        _, parent = best.pop(v)
        children[parent].append(v)
        for u in best:
            d = float(dmat[v, u])
            if d < best[u][0]:
                best[u] = (d, v)
    return children


def preorder(children: dict[int, list[int]], root: int = 0) -> list[int]:
    """Recorrido en preorden del MST con atajos triangulares (Tarea 2, Ej. 5)."""
    return [root] + [c for child in children[root] for c in preorder(children, child)]


def simulated_annealing_mat(
    dmat: np.ndarray,
    initial_route: list[int],
    T0: float = 0.5,
    alpha: float = 0.995,
    seed: int = 42,
) -> float:
    """Recocido Simulado con operador de vecindad 2-opt (Tarea 2, Ej. 4)."""
    rng = random.Random(seed)
    n = dmat.shape[0]
    current = list(initial_route)
    current_cost = tour_length_mat(dmat, current)
    best_cost = current_cost
    if n < 4:
        return best_cost

    iters = max(800, 200 * n)
    T = T0
    for _ in range(iters):
        i, k = sorted(rng.sample(range(1, n), 2))
        a = current[i - 1]
        b = current[i]
        c = current[k]
        d = current[(k + 1) % n]
        delta = (dmat[a, c] + dmat[b, d]) - (dmat[a, b] + dmat[c, d])
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
            current[i : k + 1] = current[i : k + 1][::-1]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
        T *= alpha

    return best_cost


def evaluate_split(
    df: pd.DataFrame, instances_dir: str, seed_offset: int
) -> pd.DataFrame:
    """Corre los 3 algoritmos sobre cada instancia del split y determina el ganador."""
    c_2opt_list, c_sa_list, c_mst_list, winners = [], [], [], []

    for idx, row in df.iterrows():
        pts = load_points(instances_dir, row["instance_id"])
        dmat = compute_dist_matrix(pts)

        # 1. Búsqueda Local (2-opt Hill Climbing desde Vecino Más Cercano)
        nn_route = nearest_neighbor_mat(dmat, start_node=0)
        _, c_2opt = two_opt_local_search_mat(dmat, nn_route)

        # 2. Aproximación MST + 2-opt
        children = mst_prim_mat(dmat)
        mst_tour = preorder(children, root=0)
        _, c_mst = two_opt_local_search_mat(dmat, mst_tour)

        # 3. Recocido Simulado (SA + 2-opt)
        c_sa = simulated_annealing_mat(
            dmat, nn_route, T0=0.5, alpha=0.995, seed=seed_offset + int(idx)
        )

        # Criterio de desempate (tolerancia 1e-9): prioridad por velocidad computacional
        candidates = [
            ("2opt_hill_climbing", c_2opt),
            ("mst_2opt", c_mst),
            ("sa_2opt", c_sa),
        ]
        min_cost = min(c for _, c in candidates)
        winner = next(name for name, c in candidates if c <= min_cost + 1e-9)

        c_2opt_list.append(round(c_2opt, 6))
        c_sa_list.append(round(c_sa, 6))
        c_mst_list.append(round(c_mst, 6))
        winners.append(winner)

    labeled = df.copy()
    labeled["cost_2opt_hill_climbing"] = c_2opt_list
    labeled["cost_sa_2opt"] = c_sa_list
    labeled["cost_mst_2opt"] = c_mst_list
    labeled["best_algorithm"] = winners
    return labeled


def main() -> None:
    os.makedirs("results", exist_ok=True)
    data_dir = "data"
    instances_dir = os.path.join(data_dir, "instances")

    train_raw = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_raw = pd.read_csv(os.path.join(data_dir, "test.csv"))

    print("Ejecutando algoritmos de Tarea 2 sobre las 500 instancias TSP...")
    t0 = time.perf_counter()
    train_labeled = evaluate_split(train_raw, instances_dir, seed_offset=1000)
    test_labeled = evaluate_split(test_raw, instances_dir, seed_offset=2000)
    elapsed = time.perf_counter() - t0

    train_out = os.path.join(data_dir, "train_labeled.csv")
    test_out = os.path.join(data_dir, "test_labeled.csv")
    train_labeled.to_csv(train_out, index=False)
    test_labeled.to_csv(test_out, index=False)
    print(f"Etiquetado completado en {elapsed:.2f} s.")
    print(f"Archivos guardados: {train_out} y {test_out}\n")

    full_df = pd.concat(
        [train_labeled.assign(split="train"), test_labeled.assign(split="test")],
        ignore_index=True,
    )

    # Resumen de balance de clases
    counts = full_df["best_algorithm"].value_counts()
    pcts = full_df["best_algorithm"].value_counts(normalize=True) * 100
    print("=" * 68)
    print("BALANCE DE CLASES OBTENIDO (500 INSTANCIAS)")
    print("=" * 68)
    for alg in ["2opt_hill_climbing", "mst_2opt", "sa_2opt"]:
        print(f"  {alg:<22}: {counts.get(alg, 0):3d} instancias ({pcts.get(alg, 0.0):5.2f}%)")

    print("\nDistribución por conjunto (Train vs. Test):")
    print(pd.crosstab(full_df["split"], full_df["best_algorithm"]))

    print("\nPorcentaje ganador por distribución espacial:")
    print((pd.crosstab(full_df["distribution"], full_df["best_algorithm"], normalize="index") * 100).round(2))

    print("\nConteo ganador por tamaño de instancia (n):")
    print(pd.crosstab(full_df["n"], full_df["best_algorithm"]))

    # Gráfica 1: Balance de clases global y por partición
    alg_order = ["2opt_hill_climbing", "mst_2opt", "sa_2opt"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300)
    vals_total = [counts.get(a, 0) for a in alg_order]
    bars = ax1.bar(alg_order, vals_total, color=colors, width=0.55, edgecolor="black")
    for bar, v in zip(bars, vals_total):
        ax1.annotate(
            f"{v} ({v/5.0:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=9.5,
        )
    ax1.set_title("Balance de Clases Global (500 Instancias)")
    ax1.set_ylabel("Número de Instancias Ganadas")
    ax1.set_ylim(0, max(vals_total) * 1.16)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ct_split = pd.crosstab(full_df["split"], full_df["best_algorithm"])[alg_order]
    ct_split.plot(kind="bar", ax=ax2, color=colors, edgecolor="black", rot=0)
    ax2.set_title("Balance por Partición (Train = 351 | Test = 149)")
    ax2.set_xlabel("Conjunto")
    ax2.set_ylabel("Cantidad de Instancias")
    ax2.legend(title="Algoritmo Ganador")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("results/fig1_balance_clases.png", bbox_inches="tight")

    # Gráfica 2: Patrones por distribución espacial y tamaño n
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5), dpi=300)

    ct_dist = (
        pd.crosstab(full_df["distribution"], full_df["best_algorithm"], normalize="index")[alg_order]
        * 100
    )
    ct_dist.plot(kind="bar", stacked=True, ax=ax1, color=colors, edgecolor="black", rot=0)
    ax1.set_title("Tasa de Victoria (%) por Distribución Espacial")
    ax1.set_xlabel("Distribución de Puntos")
    ax1.set_ylabel("Porcentaje de Instancias (%)")
    ax1.legend(title="Algoritmo", loc="lower right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ct_size = (
        pd.crosstab(full_df["n"], full_df["best_algorithm"], normalize="index")[alg_order] * 100
    )
    for alg, col, mk in zip(alg_order, colors, ["o-", "s-", "^-"]):
        ax2.plot(ct_size.index, ct_size[alg], mk, color=col, linewidth=2.2, markersize=7, label=alg)
    ax2.set_title("Tasa de Victoria (%) según Tamaño del Problema (n)")
    ax2.set_xlabel("Número de Ciudades (n)")
    ax2.set_ylabel("Porcentaje de Instancias (%)")
    ax2.set_xticks(ct_size.index)
    ax2.legend(title="Algoritmo")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("results/fig2_patrones_distribucion_tamano.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()