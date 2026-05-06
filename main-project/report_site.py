"""
report_site.py
--------------
Generate a narrative HTML report from the output artifacts.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from utilities import CONFIG, log

ALGORITHM_DETAILS = {
    "K-Means": "Partitions anime into a fixed number of compact groups by repeatedly moving each title toward the nearest centroid.",
    "GMM": "Models each group as a Gaussian distribution so overlapping clusters can still be separated probabilistically.",
    "DBSCAN": "Builds clusters from dense neighborhoods and marks isolated titles as noise when they do not belong to any dense region.",
    "HDBSCAN": "Extends density clustering hierarchically so it can adapt to uneven cluster shapes and keep uncertain titles separate.",
    "BIRCH": "Compresses the feature space into a clustering tree first, then assigns anime to scalable, relatively balanced partitions.",
}

SUMMARY_GRAPH_DETAILS = {
    "algorithm_ratings_comparison.png": (
        "Critic Rating vs. MAL Score by Algorithm",
        "Each panel clusters the same titles using critic ratings and MAL weighted scores, making it easier to see where agreement and disagreement between the two rating systems form natural groups.",
    ),
    "algorithm_popularity_comparison.png": (
        "Popularity Signals by Algorithm",
        "These scatterplots compare MAL votes and favourites across algorithms, showing whether highly engaged titles form their own communities or stay mixed with the rest of the dataset.",
    ),
    "algorithm_metrics_comparison.png": (
        "Algorithm Metric Comparison",
        "This bar-chart summary compares how many clusters each method found, how much noise it labeled, and how strong the resulting separation was through silhouette score.",
    ),
    "algorithm_cluster_size_comparison.png": (
        "Cluster Size Comparison",
        "This chart shows how evenly each algorithm distributed titles across clusters, which helps distinguish balanced segmentations from methods dominated by one large group.",
    ),
}


def build_output_report() -> Path:
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    clustered_df = pd.read_csv(CONFIG["clustered_data_path"])
    comparison_df = pd.read_csv(CONFIG["comparison_report_path"])
    cluster_desc_df = _read_csv_if_exists(output_dir / "cluster_descriptions.csv")
    kmeans_report_df = _read_csv_if_exists(output_dir / "cluster_report.csv")

    anime_df = pd.read_csv(CONFIG["anime_path"]).head(5)
    latest_mal_path = _latest_mal_dataset_path()
    mal_df = pd.read_csv(latest_mal_path).head(5)
    summary_cards = _build_summary_cards(clustered_df, comparison_df, latest_mal_path)
    algorithm_rows = _build_algorithm_rows(comparison_df)
    results_html = _build_results_gallery(output_dir)
    key_findings_html = _build_key_findings(cluster_desc_df, kmeans_report_df, comparison_df, clustered_df)
    conclusion_html = _build_conclusion(clustered_df, comparison_df)

    index_path = output_dir / "index.html"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Anime Ratings and Popularity Report</title>
  <style>
    :root {{
      --bg: #f2ede5;
      --panel: rgba(250, 245, 238, 0.88);
      --panel-strong: rgba(255, 250, 244, 0.93);
      --ink: #17212b;
      --muted: #4f5b66;
      --accent: #a44a3f;
      --accent-2: #305f86;
      --line: rgba(23, 33, 43, 0.12);
      --shadow: 0 20px 55px rgba(23, 33, 43, 0.16);
      --radius: 22px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background: var(--bg);
      position: relative;
      min-height: 100vh;
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      background:
        linear-gradient(rgba(17, 24, 32, 0.52), rgba(17, 24, 32, 0.62)),
        url('site_background.png') center / cover no-repeat;
      z-index: -2;
    }}
    body::after {{
      content: "";
      position: fixed;
      inset: 0;
      background:
        radial-gradient(circle at top left, rgba(212, 118, 90, 0.2), transparent 28%),
        radial-gradient(circle at top right, rgba(48, 95, 134, 0.18), transparent 30%),
        linear-gradient(180deg, rgba(242, 237, 229, 0.16), rgba(242, 237, 229, 0.34));
      z-index: -1;
    }}
    .shell {{
      width: min(1220px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 56px;
    }}
    .hero {{
      padding: 34px;
      border-radius: 28px;
      background: linear-gradient(135deg, rgba(255, 249, 241, 0.94), rgba(244, 232, 221, 0.82));
      border: 1px solid rgba(255, 255, 255, 0.48);
      box-shadow: var(--shadow);
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: clamp(2.2rem, 5vw, 4.2rem);
      line-height: 0.95;
      letter-spacing: -0.05em;
    }}
    .hero p {{
      margin: 0;
      max-width: 820px;
      color: var(--muted);
      font-size: 1.05rem;
    }}
    .section {{
      margin-top: 26px;
      padding: 28px;
      border-radius: var(--radius);
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 1.6rem;
    }}
    h3 {{
      margin: 0 0 10px;
      font-size: 1.08rem;
    }}
    p {{
      line-height: 1.6;
    }}
    .lede {{
      margin-top: 0;
      color: var(--muted);
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .card, .algo, .graph, .table-wrap, .finding, .fact-list {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255, 251, 246, 0.8);
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
      font-size: 1.9rem;
      font-weight: 700;
    }}
    .metric-note {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .table-grid, .algo-grid, .graph-grid, .finding-grid {{
      display: grid;
      gap: 16px;
    }}
    .table-grid {{
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    }}
    .algo-grid {{
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    }}
    .graph-grid {{
      grid-template-columns: 1fr;
      gap: 32px;
      padding: 10px 12px;
    }}
    .finding-grid {{
      grid-template-columns: minmax(0, 1.4fr) minmax(300px, 0.9fr);
      align-items: start;
    }}
    .algo, .finding, .fact-list {{
      padding: 18px;
    }}
    .algo p, .finding p {{
      margin: 0;
      color: var(--muted);
    }}
    .algo ul, .fact-list ul {{
      margin: 10px 0 0;
      padding-left: 18px;
    }}
    .dataset-note, .section-note {{
      color: var(--muted);
      margin: 0 0 14px;
    }}
    .table-wrap {{
      overflow: hidden;
    }}
    .table-head {{
      padding: 14px 16px 0;
    }}
    .table-inner {{
      overflow-x: auto;
      padding: 0 10px 10px;
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
      background: rgba(48, 95, 134, 0.08);
    }}
    .graph {{
      overflow: hidden;
    }}
    .graph img {{
      width: 100%;
      display: block;
      background: #fff;
      height: auto;
      object-fit: contain;
    }}
    .caption {{
      padding: 14px 16px 16px;
    }}
    .caption p {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .finding-grid .fact-list {{
      background: var(--panel-strong);
    }}
    .finding-meta {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .conclusion {{
      padding: 20px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(164, 74, 63, 0.1), rgba(48, 95, 134, 0.08));
    }}
    code {{
      font-family: "Courier New", monospace;
    }}
    @media (max-width: 900px) {{
      .finding-grid {{
        grid-template-columns: 1fr;
      }}
      .hero, .section {{
        padding: 22px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Unsupervised Clustering of Animated Series Based on Critic and Popularity Ratings</h1>
      <p>This report explores whether anime that are highly rated by critics are also the shows that attract the strongest MyAnimeList engagement, or whether popularity and critical opinion split into different clusters.</p>
      <div class="cards">{summary_cards}</div>
    </section>

    <section class="section">
      <h2>Introduction</h2>
      <p class="lede">The project is trying to measure the disconnect between two signals: curated critic ratings from <code>anime.csv</code> and community behavior from MyAnimeList snapshots. By clustering titles with both rating and popularity features, the site shows whether the same anime dominate both systems or whether fame, fan activity, and perceived quality separate into different groups.</p>
      <p class="lede">The clustering output is useful because a simple ranking does not explain why titles differ. Grouping titles by rating gap, engagement, release timing, and runtime reveals where mainstream hits, classics, divisive titles, and universally praised series land relative to one another.</p>
    </section>

    <section class="section">
      <h2>Setup</h2>
      <p class="lede">Clone the repository, move into <code>main-project</code>, install the Python dependencies used by the project, and run <code>python main.py</code> to generate the clustering outputs and rebuild the website. If you want the plots refreshed as well, run the same command without disabling visualization so the output images are recreated before the report page is written.</p>
      <p class="lede">A typical workflow is: <code>git clone ...</code>, <code>cd CS472-ML_Ratings_Predictor\\main-project</code>, install packages such as <code>pandas</code>, <code>scikit-learn</code>, <code>matplotlib</code>, and the clustering dependencies, then execute <code>python main.py</code>.</p>
    </section>

    <section class="section">
      <h2>GitHub</h2>
      <p class="lede">Repository link: <a href="https://github.com/CocoCat0/CS472-ML_Ratings_Predictor" target="_blank" rel="noreferrer">https://github.com/CocoCat0/CS472-ML_Ratings_Predictor</a></p>
    </section>

    <section class="section">
      <h2>Dataset</h2>
      <p class="dataset-note">The report uses critic data from <code>anime.csv</code> and MAL popularity snapshots from <code>{escape(latest_mal_path.name)}</code>. The tables below show the first 5 rows of each source.</p>
      <div class="table-grid">
        <section class="table-wrap">
          <div class="table-head">
            <h3>anime.csv Preview</h3>
          </div>
          <div class="table-inner">{_dataframe_to_html(anime_df)}</div>
        </section>
        <section class="table-wrap">
          <div class="table-head">
            <h3>MAL Snapshot Preview</h3>
          </div>
          <div class="table-inner">{_dataframe_to_html(mal_df)}</div>
        </section>
      </div>
    </section>

    <section class="section">
      <h2>Algorithms</h2>
      <p class="section-note">Each algorithm groups the same engineered feature set, but each one defines a "cluster" differently. That difference is what makes the result comparison meaningful.</p>
      <div class="algo-grid">{algorithm_rows}</div>
    </section>

    <section class="section">
      <h2>Results</h2>
      <p class="section-note">This section only shows the summary visuals that compare all algorithms side by side. Each figure is followed by a short interpretation of what it contributes to the analysis.</p>
      <div class="graph-grid">{results_html}</div>
    </section>

    <section class="section">
      <h2>Key Findings</h2>
      <p class="section-note">The cluster descriptions below summarize the K-Means groupings with representative titles, followed by notable facts from the overall algorithm comparison.</p>
      {key_findings_html}
    </section>

    <section class="section">
      <h2>Conclusion</h2>
      {conclusion_html}
    </section>
  </div>
</body>
</html>
"""

    index_path.write_text(html, encoding="utf-8")
    log(f"Saved output report to {index_path}")
    return index_path


def _build_summary_cards(clustered_df: pd.DataFrame, comparison_df: pd.DataFrame, latest_mal_path: Path) -> str:
    scored = _scored_comparison(comparison_df)
    best_balance = scored.sort_values("balance_score", ascending=False).iloc[0]
    top_silhouette = scored.sort_values("silhouette_score", ascending=False, na_position="last").iloc[0]
    mean_gap = pd.to_numeric(clustered_df["rating_gap"], errors="coerce").mean()
    cards = [
        ("Titles", str(len(clustered_df)), "Anime included after merging critic and MAL data."),
        ("MAL Snapshots", str(len(list(Path(CONFIG["popularity_dir"]).glob("*.csv")))), f"Latest preview uses {latest_mal_path.name}."),
        ("Best Overall Tradeoff", escape(str(best_balance["algorithm"])), "Best silhouette-minus-noise balance in this run."),
        ("Top Silhouette", _fmt(top_silhouette["silhouette_score"]), "Highest raw separation score among the algorithms."),
        ("Average Rating Gap", _fmt(mean_gap), "Positive means MAL scores are above critic scores on average."),
    ]
    return "".join(
        "<article class=\"card\">"
        f"<div class=\"eyebrow\">{label}</div>"
        f"<div class=\"metric\">{value}</div>"
        f"<div class=\"metric-note\">{note}</div>"
        "</article>"
        for label, value, note in cards
    )


def _build_algorithm_rows(comparison_df: pd.DataFrame) -> str:
    rows = []
    for record in comparison_df.to_dict(orient="records"):
        algorithm = str(record["algorithm"])
        rows.append(
            "<article class=\"algo\">"
            f"<h3>{escape(algorithm)}</h3>"
            f"<p>{escape(ALGORITHM_DETAILS.get(algorithm, 'Clustering algorithm used in this report.'))}</p>"
            "<ul>"
            f"<li>Clusters found: {_fmt(record['clusters_found'])}</li>"
            f"<li>Noise points: {_fmt(record['noise_points'])} ({_fmt(record['noise_pct'])}%)</li>"
            f"<li>Largest cluster: {_fmt(record['largest_cluster'])}</li>"
            f"<li>Silhouette score: {_fmt(record['silhouette_score'])}</li>"
            "</ul>"
            "</article>"
        )
    return "".join(rows)


def _build_results_gallery(output_dir: Path) -> str:
    cards = []
    for filename, (title, description) in SUMMARY_GRAPH_DETAILS.items():
        path = output_dir / filename
        if not path.exists():
            continue
        cards.append(
            "<figure class=\"graph\">"
            f"<img src=\"{escape(filename)}\" alt=\"{escape(title)}\">"
            "<figcaption class=\"caption\">"
            f"<h3>{escape(title)}</h3>"
            f"<p>{escape(description)}</p>"
            "</figcaption>"
            "</figure>"
        )
    return "".join(cards)


def _build_key_findings(
    cluster_desc_df: pd.DataFrame | None,
    kmeans_report_df: pd.DataFrame | None,
    comparison_df: pd.DataFrame,
    clustered_df: pd.DataFrame,
) -> str:
    finding_cards = _build_cluster_finding_cards(cluster_desc_df, kmeans_report_df)
    interesting_facts = _build_interesting_facts(comparison_df, clustered_df)

    left_column = f"<div class=\"algo-grid\">{finding_cards}</div>" if finding_cards else "<p>No cluster descriptions were available.</p>"
    right_column = (
        "<aside class=\"fact-list\">"
        "<h3>Interesting Facts</h3>"
        f"<ul>{interesting_facts}</ul>"
        "</aside>"
    )
    return f"<div class=\"finding-grid\"><div>{left_column}</div>{right_column}</div>"


def _build_cluster_finding_cards(cluster_desc_df: pd.DataFrame | None, kmeans_report_df: pd.DataFrame | None) -> str:
    if cluster_desc_df is None or cluster_desc_df.empty:
        return ""

    report_lookup: dict[int, dict[str, object]] = {}
    if kmeans_report_df is not None and not kmeans_report_df.empty:
        normalized = kmeans_report_df.copy()
        if "kmeans_cluster" in normalized.columns:
            normalized = normalized.rename(columns={"kmeans_cluster": "cluster_id"})
        if "cluster" in normalized.columns:
            normalized = normalized.rename(columns={"cluster": "cluster_id"})
        if "cluster_id" in normalized.columns:
            for record in normalized.to_dict(orient="records"):
                try:
                    report_lookup[int(record["cluster_id"])] = record
                except (TypeError, ValueError):
                    continue

    cards = []
    for record in cluster_desc_df.to_dict(orient="records"):
        cluster_id = int(record["cluster_id"])
        report_row = report_lookup.get(cluster_id, {})
        stats_html = ""
        if report_row:
            stats_html = (
                f"<div class=\"finding-meta\"><strong>Titles in cluster:</strong> {_fmt(report_row.get('titles'))} | "
                f"<strong>Avg. critic rating:</strong> {_fmt(report_row.get('critic_rating'))} | "
                f"<strong>Avg. MAL score:</strong> {_fmt(report_row.get('mal_weighted_score'))}</div>"
            )
        cards.append(
            "<article class=\"finding\">"
            f"<h3>Cluster {cluster_id}</h3>"
            f"<p>{escape(str(record.get('short_description', '')))}</p>"
            f"<div class=\"finding-meta\"><strong>Representative titles:</strong> {escape(str(record.get('representatives', '')))}</div>"
            f"{stats_html}"
            "</article>"
        )
    return "".join(cards)


def _build_interesting_facts(comparison_df: pd.DataFrame, clustered_df: pd.DataFrame) -> str:
    scored = _scored_comparison(comparison_df)
    best_balance = scored.sort_values("balance_score", ascending=False).iloc[0]
    highest_silhouette = scored.sort_values("silhouette_score", ascending=False, na_position="last").iloc[0]
    noisiest = scored.sort_values("noise_pct", ascending=False).iloc[0]

    rating_gap = pd.to_numeric(clustered_df["rating_gap"], errors="coerce")
    largest_gap_row = clustered_df.loc[rating_gap.abs().idxmax()]
    mal_higher = int((rating_gap > 0).sum())
    critic_higher = int((rating_gap < 0).sum())
    closest_match = clustered_df.loc[rating_gap.abs().idxmin()]

    facts = [
        f"<li><strong>{escape(str(best_balance['algorithm']))}</strong> was the most balanced overall because it combined the strongest silhouette-minus-noise tradeoff without discarding any titles as noise.</li>",
        f"<li><strong>{escape(str(highest_silhouette['algorithm']))}</strong> achieved the highest raw silhouette score at {_fmt(highest_silhouette['silhouette_score'])}, but it did so while labeling {_fmt(noisiest['noise_points'])} titles as noise in the noisiest case.</li>",
        f"<li>The largest disagreement between critic and MAL scoring was <strong>{escape(str(largest_gap_row['title']))}</strong>, where the MAL weighted score differed from the critic score by {_fmt(abs(float(largest_gap_row['rating_gap'])))} points.</li>",
        f"<li>MAL scored <strong>{mal_higher}</strong> titles above the critic rating and <strong>{critic_higher}</strong> below it, which suggests the disconnect exists in both directions rather than in only one consistent bias.</li>",
        f"<li><strong>{escape(str(closest_match['title']))}</strong> was the closest agreement case, with only a {_fmt(abs(float(closest_match['rating_gap'])))}-point gap between critic and MAL scoring.</li>",
    ]
    return "".join(facts)


def _build_conclusion(clustered_df: pd.DataFrame, comparison_df: pd.DataFrame) -> str:
    scored = _scored_comparison(comparison_df)
    best_balance = scored.sort_values("balance_score", ascending=False).iloc[0]
    mean_gap = pd.to_numeric(clustered_df["rating_gap"], errors="coerce").mean()
    mean_abs_gap = pd.to_numeric(clustered_df["rating_gap"], errors="coerce").abs().mean()

    return (
        "<div class=\"conclusion\">"
        "<p>The problem this project is trying to solve is whether popularity can be treated as a proxy for quality. The clustering results suggest the answer is only partially yes: some anime are strong on both critic score and MAL engagement, but other groups show that iconic franchises, divisive hits, and classics can attract very different levels of fan activity without matching critic behavior exactly.</p>"
        f"<p>Across this run, the average signed rating gap was {_fmt(mean_gap)} and the average absolute gap was {_fmt(mean_abs_gap)}, so the disconnect is not overwhelming in every case, but it is large enough to create distinct clusters. <strong>{escape(str(best_balance['algorithm']))}</strong> produced the clearest overall tradeoff for telling that story because it separated the dataset without relying on heavy noise labeling.</p>"
        "</div>"
    )
def _latest_mal_dataset_path() -> Path:
    popularity_dir = Path(CONFIG["popularity_dir"])
    return sorted(popularity_dir.glob("*.csv"))[-1]


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _scored_comparison(comparison_df: pd.DataFrame) -> pd.DataFrame:
    scored = comparison_df.copy()
    scored["silhouette_score"] = pd.to_numeric(scored["silhouette_score"], errors="coerce")
    scored["noise_pct"] = pd.to_numeric(scored["noise_pct"], errors="coerce")
    scored["balance_score"] = scored["silhouette_score"].fillna(0) - (scored["noise_pct"] / 100.0)
    return scored


def _dataframe_to_html(df: pd.DataFrame) -> str:
    display_df = df.copy()
    display_df.columns = [str(column).replace("_", " ") for column in display_df.columns]
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
