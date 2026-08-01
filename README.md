# Table Tennis Player Clustering

This project constructs a player-opponent network from Setka Cup table tennis match data and uses k-means clustering to group players based on performance and network features.

## Research question

Can table tennis players be grouped into meaningful types based on their performance and opponent-network patterns?

A suitable stakeholder is a table tennis tournament organizer, sports-data quality team, coaching organization, or sports media analytics team. The clusters can help the stakeholder distinguish stronger performers, weaker performers, and highly active players with balanced results.

## Dataset

The script downloads the Setka Table Tennis Dataset from the SCORE Sports Data Repository:

- 7,851 original match rows
- 18 original columns
- Player names
- Sets won
- Game-level point totals
- Match winner
- Match date and time

The code finds and removes one row in which the same player appears as both Player1 and Player2.

## Features used for clustering

Each row in the feature matrix represents one player.

- `matches_played`
- `win_pct`
- `avg_set_diff`
- `avg_point_diff`
- `unique_opponents`
- `avg_opponent_win_pct`

The features are standardized before k-means clustering.

## Selecting k

The script tests values from 2 through 8 and saves:

- An elbow plot
- A silhouette-score plot
- A CSV containing the k-selection metrics

The highest silhouette score is produced by k=2. The project uses k=3 because it remains close in quality while producing a more useful three-group interpretation:

1. Stronger performance
2. Weaker performance
3. High-activity balanced

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python table_tennis_clustering.py
```

The data is downloaded automatically and saved in the `data` folder.

## Generated files

The script creates an `outputs` folder containing:

- `elbow_plot.png`
- `silhouette_plot.png`
- `cluster_pca.png`
- `network_sample.png`
- `k_selection_metrics.csv`
- `player_clusters.csv`
- `cluster_summary.csv`
- `cluster_examples.csv`

## Validation

The code checks:

- Required columns
- Duplicate match IDs
- Invalid winner values
- Winner and set-score agreement
- Self-matches
- Whether total player wins equal total matches
- A minimum of 10 matches per clustered player

## Repository link

After uploading these files to GitHub, copy the repository URL into the final Medium post.
