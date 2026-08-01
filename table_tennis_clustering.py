from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://data.scorenetwork.org/data/table_tennis-sept2022.csv"
DATA_PATH = Path("data/table_tennis-sept2022.csv")
OUTPUT_DIR = Path("outputs")
MIN_MATCHES = 10
SELECTED_K = 3
RANDOM_STATE = 42


def load_matches():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        matches = pd.read_csv(DATA_PATH)
    else:
        matches = pd.read_csv(DATA_URL)
        matches.to_csv(DATA_PATH, index=False)

    return matches


def validate_and_clean(matches):
    required = {
        "X", "Player1", "Player2", "Sets_P1", "Sets_P2", "HomeWinner",
        "P1_G1", "P2_G1", "P1_G2", "P2_G2", "P1_G3", "P2_G3",
        "P1_G4", "P2_G4", "P1_G5", "P2_G5"
    }
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    original_rows = len(matches)
    duplicate_rows = matches["X"].duplicated().sum()
    self_matches = (matches["Player1"] == matches["Player2"]).sum()
    invalid_winners = (~matches["HomeWinner"].isin([0, 1])).sum()
    winner_mismatches = (
        ((matches["HomeWinner"] == 1) & (matches["Sets_P1"] <= matches["Sets_P2"]))
        | ((matches["HomeWinner"] == 0) & (matches["Sets_P2"] <= matches["Sets_P1"]))
    ).sum()

    if invalid_winners or winner_mismatches:
        raise ValueError("Winner fields do not agree with the set scores.")

    matches = matches.drop_duplicates(subset="X").copy()
    matches = matches[matches["Player1"] != matches["Player2"]].copy()

    print("Validation summary")
    print(f"Original rows: {original_rows}")
    print(f"Duplicate match IDs removed: {duplicate_rows}")
    print(f"Self-matches removed: {self_matches}")
    print(f"Rows after cleaning: {len(matches)}")

    return matches


def build_network(matches):
    graph = nx.Graph()

    for row in matches.itertuples(index=False):
        if graph.has_edge(row.Player1, row.Player2):
            graph[row.Player1][row.Player2]["weight"] += 1
        else:
            graph.add_edge(row.Player1, row.Player2, weight=1)

    return graph


def build_player_features(matches, graph):
    p1_game_cols = [f"P1_G{i}" for i in range(1, 6)]
    p2_game_cols = [f"P2_G{i}" for i in range(1, 6)]

    matches["Points_P1"] = matches[p1_game_cols].sum(axis=1, skipna=True)
    matches["Points_P2"] = matches[p2_game_cols].sum(axis=1, skipna=True)

    player1_rows = pd.DataFrame({
        "player": matches["Player1"],
        "opponent": matches["Player2"],
        "win": matches["HomeWinner"].astype(int),
        "sets_won": matches["Sets_P1"],
        "sets_lost": matches["Sets_P2"],
        "points_won": matches["Points_P1"],
        "points_lost": matches["Points_P2"]
    })

    player2_rows = pd.DataFrame({
        "player": matches["Player2"],
        "opponent": matches["Player1"],
        "win": 1 - matches["HomeWinner"].astype(int),
        "sets_won": matches["Sets_P2"],
        "sets_lost": matches["Sets_P1"],
        "points_won": matches["Points_P2"],
        "points_lost": matches["Points_P1"]
    })

    player_matches = pd.concat([player1_rows, player2_rows], ignore_index=True)
    player_matches["set_diff"] = player_matches["sets_won"] - player_matches["sets_lost"]
    player_matches["point_diff"] = player_matches["points_won"] - player_matches["points_lost"]

    stats = player_matches.groupby("player").agg(
        wins=("win", "sum"),
        win_pct=("win", "mean"),
        avg_set_diff=("set_diff", "mean"),
        avg_point_diff=("point_diff", "mean")
    ).reset_index()

    stats["matches_played"] = stats["player"].map(dict(graph.degree(weight="weight")))
    stats["unique_opponents"] = stats["player"].map(dict(graph.degree()))

    win_pct_lookup = stats.set_index("player")["win_pct"]
    player_matches["opponent_win_pct"] = player_matches["opponent"].map(win_pct_lookup)
    opponent_strength = player_matches.groupby("player")["opponent_win_pct"].mean()
    stats["avg_opponent_win_pct"] = stats["player"].map(opponent_strength)

    if stats["wins"].sum() != len(matches):
        raise ValueError("Player win totals do not equal the number of matches.")

    stats = stats[stats["matches_played"] >= MIN_MATCHES].copy()
    return stats


def select_and_fit_clusters(stats):
    feature_columns = [
        "matches_played",
        "win_pct",
        "avg_set_diff",
        "avg_point_diff",
        "unique_opponents",
        "avg_opponent_win_pct"
    ]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(stats[feature_columns])

    results = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(scaled_features)
        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette_score": silhouette_score(scaled_features, labels)
        })

    k_results = pd.DataFrame(results)
    k_results.to_csv(OUTPUT_DIR / "k_selection_metrics.csv", index=False)

    model = KMeans(n_clusters=SELECTED_K, random_state=RANDOM_STATE, n_init=20)
    stats["cluster"] = model.fit_predict(scaled_features)
    stats["distance_to_centroid"] = np.linalg.norm(
        scaled_features - model.cluster_centers_[stats["cluster"]], axis=1
    )

    return stats, scaled_features, feature_columns, k_results


def name_clusters(stats):
    cluster_means = stats.groupby("cluster")[
        ["matches_played", "win_pct", "avg_set_diff", "avg_point_diff", "unique_opponents", "avg_opponent_win_pct"]
    ].mean()

    strongest = cluster_means["win_pct"].idxmax()
    weakest = cluster_means["win_pct"].idxmin()
    remaining = [cluster for cluster in cluster_means.index if cluster not in {strongest, weakest}]

    labels = {
        strongest: "Stronger performance",
        weakest: "Weaker performance"
    }
    if remaining:
        labels[remaining[0]] = "High-activity balanced"

    stats["cluster_description"] = stats["cluster"].map(labels)
    return stats


def save_tables(stats, feature_columns):
    summary = stats.groupby(["cluster", "cluster_description"])[feature_columns].mean().round(3)
    summary["players"] = stats.groupby(["cluster", "cluster_description"]).size()
    summary = summary.reset_index()

    examples = (
        stats.sort_values(["cluster", "distance_to_centroid"])
        .groupby("cluster")
        .head(2)[["cluster", "cluster_description", "player", *feature_columns]]
    )

    stats.sort_values(["cluster", "player"]).to_csv(OUTPUT_DIR / "player_clusters.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)
    examples.to_csv(OUTPUT_DIR / "cluster_examples.csv", index=False)

    print("\nCluster summary")
    print(summary.to_string(index=False))
    print("\nTwo example players from each cluster")
    print(examples.to_string(index=False))


def save_figures(stats, scaled_features, k_results, graph):
    plt.figure(figsize=(7, 5))
    plt.plot(k_results["k"], k_results["inertia"], marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Plot")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "elbow_plot.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(k_results["k"], k_results["silhouette_score"], marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Scores")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "silhouette_plot.png", dpi=300)
    plt.close()

    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(scaled_features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=stats["cluster"], alpha=0.7)
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.title("Table Tennis Player Clusters")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_pca.png", dpi=300)
    plt.close()

    top_players = stats.nlargest(40, "matches_played")["player"]
    sample_graph = graph.subgraph(top_players).copy()
    positions = nx.spring_layout(sample_graph, seed=RANDOM_STATE, weight="weight")
    cluster_lookup = stats.set_index("player")["cluster"]
    node_colors = [cluster_lookup.loc[player] for player in sample_graph.nodes()]
    node_sizes = [40 + 5 * graph.degree(player, weight="weight") for player in sample_graph.nodes()]
    edge_widths = [0.3 + 0.25 * sample_graph[u][v]["weight"] for u, v in sample_graph.edges()]

    plt.figure(figsize=(11, 9))
    nx.draw_networkx(
        sample_graph,
        pos=positions,
        node_color=node_colors,
        node_size=node_sizes,
        width=edge_widths,
        font_size=6,
        with_labels=True,
        cmap=plt.cm.viridis
    )
    plt.title("Opponent Network for the 40 Most Active Players")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "network_sample.png", dpi=300)
    plt.close()


def main():
    matches = validate_and_clean(load_matches())
    graph = build_network(matches)
    stats = build_player_features(matches, graph)
    stats, scaled_features, feature_columns, k_results = select_and_fit_clusters(stats)
    stats = name_clusters(stats)

    print(f"\nNetwork nodes: {graph.number_of_nodes()}")
    print(f"Network edges: {graph.number_of_edges()}")
    print(f"Players clustered after minimum-match filter: {len(stats)}")
    best_k = int(k_results.loc[k_results["silhouette_score"].idxmax(), "k"])
    print("\nK-selection results")
    print(k_results.round(3).to_string(index=False))
    print(f"Highest silhouette score: k={best_k}")
    print(f"Selected k: {SELECTED_K} for a more useful three-group interpretation")

    save_tables(stats, feature_columns)
    save_figures(stats, scaled_features, k_results, graph)
    print(f"\nFiles saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
