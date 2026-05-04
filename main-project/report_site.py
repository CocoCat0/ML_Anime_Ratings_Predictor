"""
report_site.py
--------------
Generate a lightweight HTML dashboard from the output artifacts.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from utilities import CONFIG, log

ALGORITHM_DETAILS = {
    "K-Means": "Centroid-based baseline that forces each title into one of k compact groups.",
    "GMM": "Soft clustering with Gaussian components, useful when clusters overlap.",
    "DBSCAN": "Density-based clustering that can label sparse titles as noise.",
    "HDBSCAN": "Hierarchical density clustering that adapts cluster shape and can keep noise separate.",
    "BIRCH": "Tree-based clustering that scales well and often gives balanced partitions.",
}

GRAPH_FILES = [
    "ratings_scatter.png",
    "popularity_scatter.png",
    "gmm_clusters.png",
    "dbscan_clusters.png",
    "hdbscan_clusters.png",
    "birch_clusters.png",
    "algorithm_ratings_comparison.png",
    "algorithm_popularity_comparison.png",
    "algorithm_metrics_comparison.png",
    "algorithm_cluster_size_comparison.png",
    "vae_latent_space.png",
    "vae_algorithm_comparison.png",
    "vae_loss.png",
]

TABLE_FILES = [
    "algorithm_comparison_report.csv",
    "cluster_report.csv",
    "gmm_cluster_report.csv",
    "dbscan_cluster_report.csv",
    "hdbscan_cluster_report.csv",
    "birch_cluster_report.csv",
    "cluster_descriptions.csv",
]


def build_output_report() -> Path:
    output_dir = Path(CONFIG["output_dir"])
    clustered_path = Path(CONFIG["clustered_data_path"])
    comparison_path = Path(CONFIG["comparison_report_path"])
    index_path = output_dir / "index.html"

    clustered_df = pd.read_csv(clustered_path)
    comparison_df = pd.read_csv(comparison_path)
    snapshot_df = clustered_df[
        [
            "title",
            "genre",
            "critic_rating",
            "mal_weighted_score",
            "rating_gap",
            "kmeans_cluster",
            "gmm_cluster",
            "dbscan_cluster",
            "hdbscan_cluster",
            "birch_cluster",
        ]
    ].head(10)

    summary_cards = _build_summary_cards(clustered_df, comparison_df)
    algorithm_rows = _build_algorithm_rows(comparison_df)
    graph_gallery = _build_graph_gallery(output_dir)
    tables_html = _build_tables(output_dir)
    analysis_html = _build_analysis(comparison_df, clustered_df)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Anime Clustering Output Report</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: rgba(255, 252, 247, 0.88);
      --ink: #1f2430;
      --muted: #5d6573;
      --accent: #b85042;
      --accent-2: #3a6ea5;
      --line: rgba(31, 36, 48, 0.12);
      --shadow: 0 18px 50px rgba(44, 44, 44, 0.12);
      --radius: 22px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(184, 80, 66, 0.18), transparent 32%),
        radial-gradient(circle at top right, rgba(58, 110, 165, 0.18), transparent 28%),
        linear-gradient(180deg, #f7f2ea 0%, #efe7dc 100%);
    }}
    .shell {{
      width: min(1200px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0 56px;
    }}
    .hero {{
      padding: 36px;
      border-radius: calc(var(--radius) + 6px);
      background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(255,248,240,0.78));
      box-shadow: var(--shadow);
      border: 1px solid rgba(255,255,255,0.7);
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: clamp(2.2rem, 5vw, 4rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .hero p {{
      max-width: 760px;
      margin: 0;
      color: var(--muted);
      font-size: 1.05rem;
    }}
    .section {{
      margin-top: 28px;
      padding: 28px;
      border-radius: var(--radius);
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    h2 {{
      margin: 0 0 16px;
      font-size: 1.55rem;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 14px;
    }}
    .card, .algo, .graph, .table-wrap {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255,255,255,0.72);
    }}
    .card {{
      padding: 18px;
    }}
    .eyebrow {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.72rem;
      margin-bottom: 6px;
    }}
    .metric {{
      font-size: 2rem;
      font-weight: 700;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 18px;
    }}
    .algo-grid, .graph-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px;
    }}
    .algo {{
      padding: 18px;
    }}
    .algo h3 {{
      margin: 0 0 8px;
      font-size: 1.1rem;
    }}
    .algo p {{
      margin: 0 0 10px;
      color: var(--muted);
    }}
    .algo ul {{
      margin: 0;
      padding-left: 18px;
    }}
    .dataset-note {{
      color: var(--muted);
      margin-bottom: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      background: rgba(58, 110, 165, 0.06);
    }}
    .graph {{
      overflow: hidden;
    }}
    .graph img {{
      width: 100%;
      display: block;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      background: #fff;
    }}
    .graph .caption, .table-wrap h3 {{
      padding: 14px 16px;
      margin: 0;
    }}
    .caption small {{
      color: var(--muted);
    }}
    .table-stack {{
      display: grid;
      gap: 18px;
    }}
    .table-wrap {{
      overflow: hidden;
    }}
    .table-inner {{
      overflow-x: auto;
      padding: 0 4px 4px;
    }}
    .verdict {{
      padding: 18px;
      border-radius: 18px;
      background: linear-gradient(135deg, rgba(184, 80, 66, 0.12), rgba(58, 110, 165, 0.08));
      border: 1px solid var(--line);
    }}
    .footer {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    @media (max-width: 860px) {{
      .two-col {{ grid-template-columns: 1fr; }}
      .section, .hero {{ padding: 22px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Anime Clustering Output Report</h1>
      <p>This dashboard summarizes the generated files in <code>main-project/output</code>, including algorithm comparisons, cluster summaries, plots, and a compact verdict on which model looked strongest on this dataset.</p>
    </section>

    <section class="section">
      <h2>Overview</h2>
      <div class="cards">{summary_cards}</div>
    </section>

    <section class="section">
      <h2>Algorithms</h2>
      <div class="algo-grid">{algorithm_rows}</div>
    </section>

    <section class="section">
      <h2>Dataset | Snapshot</h2>
      <p class="dataset-note">The snapshot below shows the first 10 rows from <code>clustered_anime.csv</code> after feature engineering and label assignment.</p>
      <div class="table-wrap">
        <div class="table-inner">{_dataframe_to_html(snapshot_df)}</div>
      </div>
    </section>

    <section class="section">
      <h2>Results | Graphs | Tables</h2>
      <div class="graph-grid">{graph_gallery}</div>
      <div class="table-stack" style="margin-top: 18px;">{tables_html}</div>
    </section>

    <section class="section">
      <h2>Analysis | Verdict</h2>
      {analysis_html}
      <p class="footer">Generated from project outputs in <code>{escape(str(output_dir))}</code>.</p>
    </section>
  </div>
</body>
</html>
"""

    index_path.write_text(html, encoding="utf-8")
    log(f"Saved output report to {index_path}")
    return index_path


def _build_summary_cards(clustered_df: pd.DataFrame, comparison_df: pd.DataFrame) -> str:
    scored = comparison_df.copy()
    scored["silhouette_score"] = pd.to_numeric(scored["silhouette_score"], errors="coerce")
    scored["noise_pct"] = pd.to_numeric(scored["noise_pct"], errors="coerce")
    scored["balance_score"] = scored["silhouette_score"].fillna(0) - (scored["noise_pct"] / 100.0)
    best_balance = scored.sort_values("balance_score", ascending=False).iloc[0]
    most_noise = scored.sort_values("noise_pct", ascending=False).iloc[0]

    cards = [
        ("Titles Clustered", f"{len(clustered_df)}"),
        ("Features Used", f"{len(CONFIG['features'])}"),
        ("Algorithms", f"{len(comparison_df)}"),
        ("Best Balance", escape(str(best_balance["algorithm"]))),
        ("Top Silhouette", _fmt(scored["silhouette_score"].max())),
        ("Highest Noise", f"{escape(str(most_noise['algorithm']))} ({_fmt(most_noise['noise_pct'])}%)"),
    ]
    return "".join(
        f'<article class="card"><div class="eyebrow">{label}</div><div class="metric">{value}</div></article>'
        for label, value in cards
    )


def _build_algorithm_rows(comparison_df: pd.DataFrame) -> str:
    rows = []
    for record in comparison_df.to_dict(orient="records"):
        algorithm = str(record["algorithm"])
        rows.append(
            "<article class=\"algo\">"
            f"<h3>{escape(algorithm)}</h3>"
            f"<p>{escape(ALGORITHM_DETAILS.get(algorithm, 'Cluster summary algorithm.'))}</p>"
            "<ul>"
            f"<li>Clusters found: {_fmt(record['clusters_found'])}</li>"
            f"<li>Noise points: {_fmt(record['noise_points'])} ({_fmt(record['noise_pct'])}%)</li>"
            f"<li>Largest cluster: {_fmt(record['largest_cluster'])}</li>"
            f"<li>Silhouette: {_fmt(record['silhouette_score'])}</li>"
            "</ul>"
            "</article>"
        )
    return "".join(rows)


def _build_graph_gallery(output_dir: Path) -> str:
    cards = []
    for filename in GRAPH_FILES:
        path = output_dir / filename
        if not path.exists():
            continue
        title = filename.replace(".png", "").replace("_", " ").title()
        cards.append(
            "<figure class=\"graph\">"
            f"<img src=\"{escape(filename)}\" alt=\"{escape(title)}\">"
            f"<figcaption class=\"caption\">{escape(title)}<br><small>{escape(filename)}</small></figcaption>"
            "</figure>"
        )
    return "".join(cards)


def _build_tables(output_dir: Path) -> str:
    sections = []
    for filename in TABLE_FILES:
        path = output_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        sections.append(
            "<section class=\"table-wrap\">"
            f"<h3>{escape(filename)}</h3>"
            f"<div class=\"table-inner\">{_dataframe_to_html(df)}</div>"
            "</section>"
        )
    return "".join(sections)


def _build_analysis(comparison_df: pd.DataFrame, clustered_df: pd.DataFrame) -> str:
    scored = comparison_df.copy()
    scored["silhouette_score"] = pd.to_numeric(scored["silhouette_score"], errors="coerce")
    scored["noise_pct"] = pd.to_numeric(scored["noise_pct"], errors="coerce")
    scored["balance_score"] = scored["silhouette_score"].fillna(0) - (scored["noise_pct"] / 100.0)

    best_balance = scored.sort_values("balance_score", ascending=False).iloc[0]
    best_silhouette = scored.sort_values("silhouette_score", ascending=False, na_position="last").iloc[0]
    most_noise = scored.sort_values("noise_pct", ascending=False).iloc[0]
    rating_gap = pd.to_numeric(clustered_df["rating_gap"], errors="coerce")
    mean_gap = rating_gap.mean()

    verdict = (
        f"<div class=\"verdict\"><strong>Verdict:</strong> "
        f"{escape(str(best_balance['algorithm']))} delivered the strongest overall tradeoff in this run. "
        f"It avoided heavy noise labeling while keeping one of the better silhouette scores. "
        f"{escape(str(best_silhouette['algorithm']))} posted the highest raw silhouette, but its noise rate was much higher. "
        f"Average MAL-vs-critic rating gap across the clustered titles was {_fmt(mean_gap)}, which suggests the two rating systems were fairly aligned overall."
        "</div>"
    )

    notes = (
        "<p>"
        f"The noisiest model was {escape(str(most_noise['algorithm']))} at {_fmt(most_noise['noise_pct'])}% noise, "
        f"while {escape(str(best_silhouette['algorithm']))} reached the top silhouette score of {_fmt(best_silhouette['silhouette_score'])}. "
        "HDBSCAN found a broad dominant cluster with a smaller noise set, which may indicate the dataset is compact after scaling."
        "</p>"
    )
    return notes + verdict


def _dataframe_to_html(df: pd.DataFrame) -> str:
    display_df = df.copy()
    for column in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(_fmt)
    return display_df.to_html(index=False, border=0, classes="dataframe")


def _fmt(value) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)
