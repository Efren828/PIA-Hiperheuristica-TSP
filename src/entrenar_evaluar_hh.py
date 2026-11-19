import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


def compute_regret_stats(
    preds: np.ndarray, df_eval: pd.DataFrame, cost_cols: dict[str, str]
) -> tuple[float, float, float, float]:
    """Calcula el Regret absoluto (medio, mediana, máximo) y relativo promedio (%)."""
    oracle_costs = df_eval[list(cost_cols.values())].min(axis=1).to_numpy()
    chosen_costs = np.array(
        [df_eval.iloc[i][cost_cols[preds[i]]] for i in range(len(preds))]
    )
    regrets = chosen_costs - oracle_costs
    rel_regrets = (regrets / oracle_costs) * 100.0
    return (
        float(np.mean(regrets)),
        float(np.median(regrets)),
        float(np.max(regrets)),
        float(np.mean(rel_regrets)),
    )


def main() -> None:
    os.makedirs("results", exist_ok=True)
    train_df = pd.read_csv("data/train_labeled.csv")
    test_df = pd.read_csv("data/test_labeled.csv")

    num_cols = ["n", "mean_pairwise_dist", "std_coords", "mean_centroid_dist", "mean_nn_dist"]
    cat_cols = ["distribution"]
    feature_cols = num_cols + cat_cols
    alg_order = ["2opt_hill_climbing", "mst_2opt", "sa_2opt"]

    cost_cols = {
        "2opt_hill_climbing": "cost_2opt_hill_climbing",
        "sa_2opt": "cost_sa_2opt",
        "mst_2opt": "cost_mst_2opt",
    }

    X_train = train_df[feature_cols]
    y_train = train_df["best_algorithm"]
    X_test = test_df[feature_cols]
    y_test = test_df["best_algorithm"]

    # 1. Baseline Trivial (siempre predecir la clase mayoritaria en train)
    majority_class = y_train.value_counts().idxmax()
    base_preds = np.full(len(y_test), majority_class)
    base_acc = accuracy_score(y_test, base_preds)
    base_reg = compute_regret_stats(base_preds, test_df, cost_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop=None), cat_cols),
        ]
    )

    models = {
        "KNN (k=15, euclidiana) [Principal]": KNeighborsClassifier(
            n_neighbors=15, weights="uniform", metric="euclidean"
        ),
        "Árbol de Decisión (max_depth=3)": DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=10, random_state=42
        ),
        "Random Forest (300 árboles, depth=4)": RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=6, random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("=" * 84)
    print("EVALUACIÓN DE LA HIPER-HEURÍSTICA EN CONJUNTO DE PRUEBA (149 INSTANCIAS)")
    print("=" * 84)
    print(
        f"{'Modelo':<38} | {'CV Acc':>7} | {'Test Acc':>8} | {'Regret Medio':>12} | {'Regret %':>8}"
    )
    print("-" * 84)
    print(
        f"{'Baseline Trivial (' + majority_class + ')':<38} | {'—':>7} | "
        f"{base_acc*100:7.2f}% | {base_reg[0]:12.5f} | {base_reg[3]:7.3f}%"
    )

    fitted_pipes = {}
    test_predictions = {"Baseline Trivial": base_preds}

    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        cv_score = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy").mean()
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        reg_mean, reg_med, reg_max, reg_pct = compute_regret_stats(preds, test_df, cost_cols)

        fitted_pipes[name] = pipe
        test_predictions[name] = preds
        print(
            f"{name:<38} | {cv_score*100:6.2f}% | {acc*100:7.2f}% | "
            f"{reg_mean:12.5f} | {reg_pct:7.3f}%"
        )

    # 2. Matrices de Confusión detalladas
    print("\n" + "=" * 84)
    print("MATRICES DE CONFUSIÓN EN TEST (Filas = Real, Columnas = Predicho)")
    print("=" * 84)
    for name in ["Baseline Trivial", "KNN (k=15, euclidiana) [Principal]", "Árbol de Decisión (max_depth=3)"]:
        cm = confusion_matrix(y_test, test_predictions[name], labels=alg_order)
        print(f"\n--- {name} ---")
        print(pd.DataFrame(cm, index=alg_order, columns=alg_order))

    # 3. Importancia de características (Permutation Importance en KNN + Gini en RF)
    knn_pipe = fitted_pipes["KNN (k=15, euclidiana) [Principal]"]
    perm = permutation_importance(
        knn_pipe, X_test, y_test, n_repeats=30, random_state=42, scoring="accuracy"
    )
    perm_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "perm_mean": perm.importances_mean,
            "perm_std": perm.importances_std,
        }
    ).sort_values("perm_mean", ascending=False)

    print("\n" + "=" * 84)
    print("IMPORTANCIA DE CARACTERÍSTICAS (Permutation Importance en KNN)")
    print("=" * 84)
    print(perm_df.to_string(index=False))

    dt_pipe = fitted_pipes["Árbol de Decisión (max_depth=3)"]
    ohe_names = list(
        dt_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols)
    )
    print("\nReglas interpretables del Árbol de Decisión (max_depth=3):")
    print(export_text(dt_pipe.named_steps["clf"], feature_names=num_cols + ohe_names))

    # 4. Gráfica 3: Matriz de Confusión de KNN vs. Regret comparativo
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    cm_knn = confusion_matrix(
        y_test, test_predictions["KNN (k=15, euclidiana) [Principal]"], labels=alg_order
    )
    im = ax1.imshow(cm_knn, cmap="Blues")
    ax1.set_xticks(range(len(alg_order)))
    ax1.set_yticks(range(len(alg_order)))
    ax1.set_xticklabels(["2opt_hc", "mst_2opt", "sa_2opt"])
    ax1.set_yticklabels(["2opt_hc", "mst_2opt", "sa_2opt"])
    ax1.set_xlabel("Algoritmo Predicho")
    ax1.set_ylabel("Algoritmo Óptimo Real")
    ax1.set_title("Matriz de Confusión: KNN (k=15) en Test (n=149)")
    for i in range(len(alg_order)):
        for j in range(len(alg_order)):
            color_txt = "white" if cm_knn[i, j] > cm_knn.max() / 2 else "black"
            ax1.text(j, i, str(cm_knn[i, j]), ha="center", va="center", color=color_txt, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    comp_labels = ["Baseline\nTrivial", "KNN\n(k=15)", "Árbol Decisión\n(depth=3)", "Random Forest\n(depth=4)"]
    comp_accs = [
        accuracy_score(y_test, test_predictions[k]) * 100
        for k in [
            "Baseline Trivial",
            "KNN (k=15, euclidiana) [Principal]",
            "Árbol de Decisión (max_depth=3)",
            "Random Forest (300 árboles, depth=4)",
        ]
    ]
    bar_colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#9467bd"]
    bars = ax2.bar(comp_labels, comp_accs, color=bar_colors, width=0.52, edgecolor="black")
    ax2.axhline(base_acc * 100, color="#d62728", linestyle="--", linewidth=1.6, label=f"Baseline ({base_acc*100:.2f}%)")
    for bar, v in zip(bars, comp_accs):
        ax2.annotate(
            f"{v:.2f}%",
            (bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=9.5,
        )
    ax2.set_title("Comparación de Accuracy en Conjunto de Prueba")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(50, 75)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("results/fig3_matriz_confusion_modelos.png", bbox_inches="tight")

    # 5. Gráfica 4: Importancia de Características
    perm_sorted = perm_df.sort_values("perm_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    ax.barh(
        perm_sorted["feature"],
        perm_sorted["perm_mean"],
        xerr=perm_sorted["perm_std"],
        color="#1f77b4",
        edgecolor="black",
        capsize=4,
    )
    ax.set_title("Importancia de Características (Permutation Importance en KNN)")
    ax.set_xlabel("Caída Media en Accuracy al Permutar la Variable")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("results/fig4_importancia_caracteristicas.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()