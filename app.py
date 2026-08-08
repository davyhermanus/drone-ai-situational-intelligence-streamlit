from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model_core import (
    ModelConfig,
    compute_anomaly_scores,
    evaluate_scores,
    generate_simulated_data,
    run_experiment_grid,
    scenario_detection,
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUTPUTS = BASE / "outputs"
OUTPUTS.mkdir(exist_ok=True)
SENSOR = DATA / "simulated_drone_sensing.csv"
FEATURES = DATA / "image_features.csv"
VALIDATIONS = OUTPUTS / "human_validation_log.csv"
DATA_DICTIONARY = BASE / "data_dictionary.csv"
APP_VERSION = "1.1.1"


def apply_high_contrast_style() -> None:
    """Improve readability without changing Streamlit's functional behavior."""
    st.markdown(
        """
        <style>
        :root {
            --text-main: #111827;
            --text-muted: #374151;
            --panel: #f8fafc;
            --border: #cbd5e1;
            --accent: #b91c1c;
        }
        html, body, [class*="css"] { color: var(--text-main); }
        .stApp { background: #ffffff; }
        h1, h2, h3, h4 { color: #0f172a !important; font-weight: 750 !important; }
        p, label, .stCaption, [data-testid="stMarkdownContainer"] {
            color: var(--text-main) !important;
        }
        [data-testid="stCaptionContainer"], .stCaption {
            color: var(--text-muted) !important;
            font-size: 0.95rem !important;
        }
        [data-testid="stSidebar"] {
            background: #eef2f7;
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #111827 !important;
        }
        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.75rem 0.9rem;
        }
        [data-testid="stMetricLabel"] { color: #334155 !important; font-weight: 650; }
        [data-testid="stMetricValue"] { color: #0f172a !important; font-weight: 750; }
        .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; border-bottom: 1px solid var(--border); }
        .stTabs [data-baseweb="tab"] {
            color: #334155 !important;
            font-weight: 650;
            padding-left: 0.85rem;
            padding-right: 0.85rem;
        }
        .stTabs [aria-selected="true"] {
            color: var(--accent) !important;
            border-bottom-color: var(--accent) !important;
        }
        [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }
        .stButton > button, .stDownloadButton > button {
            border: 1px solid #64748b;
            color: #111827;
            font-weight: 650;
            background: #ffffff;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: var(--accent);
            color: var(--accent);
        }
        .section-note {
            background: #f8fafc;
            border-left: 5px solid #475569;
            padding: 0.8rem 1rem;
            margin: 0.35rem 0 1rem 0;
            color: #111827;
        }
        .footer-note {
            margin-top: 2rem;
            padding-top: 0.8rem;
            border-top: 1px solid var(--border);
            color: #475569;
            font-size: 0.88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(SENSOR, parse_dates=["timestamp"]), pd.read_csv(FEATURES)


def save_validation(row: dict) -> None:
    new = pd.DataFrame([row])
    old = pd.read_csv(VALIDATIONS) if VALIDATIONS.exists() else pd.DataFrame()
    pd.concat([old, new], ignore_index=True).to_csv(VALIDATIONS, index=False)


def metric_cards(df: pd.DataFrame, metrics: dict) -> None:
    cols = st.columns(6)
    cols[0].metric("Frames", len(df))
    cols[1].metric("Known anomalies", int(df.ground_truth_anomaly.sum()))
    cols[2].metric("Detected frames", int(df.predicted_anomaly.sum()))
    cols[3].metric("Precision", f"{metrics['precision']:.3f}")
    cols[4].metric("Recall", f"{metrics['recall']:.3f}")
    cols[5].metric("F1", f"{metrics['f1']:.3f}")


def configuration_table(cfg: ModelConfig) -> pd.DataFrame:
    rows = [
        ("Isolation Forest contamination", cfg.contamination, "Both branches"),
        ("Pollutant branch weight (alpha)", cfg.alpha, "Ensemble fusion"),
        ("Image branch weight", 1 - cfg.alpha, "Ensemble fusion"),
        ("Watch threshold", cfg.watch, "Operational mapping"),
        ("Warning threshold", cfg.warning, "Operational mapping"),
        ("Critical threshold", cfg.critical, "Operational mapping"),
        ("Pollutant IF weight", cfg.pollutant_if_weight, "Within pollutant branch"),
        ("Image IF weight", cfg.image_if_weight, "Within image branch"),
        ("Pollutant random seed", cfg.pollutant_seed, "Reproducibility"),
        ("Image random seed", cfg.image_seed, "Reproducibility"),
        ("Isolation Forest estimators", cfg.n_estimators, "Both branches"),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value", "Role"])


def main() -> None:
    st.set_page_config(page_title=f"Drone-AI SI Prototype v{APP_VERSION}", layout="wide")
    apply_high_contrast_style()

    st.title(f"Drone-AI Situational Intelligence Prototype v{APP_VERSION}")
    st.caption(
        "Controlled synthetic scenarios with two-branch anomaly scoring, sensitivity analysis, "
        "ablation, robustness testing, and persistent human validation."
    )

    with st.sidebar:
        st.header("Simulation")
        seed = st.number_input("Simulation seed", 1, 9999, 42, 1)
        n = st.slider("Mission frames", 80, 240, 160, 20)
        if st.button("Regenerate and apply", use_container_width=True):
            generate_simulated_data(DATA, n=int(n), seed=int(seed))
            st.success("Data regenerated.")
            st.rerun()

        st.header("Model configuration")
        contamination = st.slider("Isolation Forest contamination", 0.03, 0.25, 0.10, 0.01)
        alpha = st.slider("Pollutant branch weight α", 0.0, 1.0, 0.60, 0.05)
        watch = st.slider("Watch threshold", 0.20, 0.75, 0.45, 0.01)
        warning = st.slider("Warning threshold", 0.30, 0.90, 0.60, 0.01)
        critical = st.slider("Critical threshold", 0.40, 0.98, 0.75, 0.01)

    if not (watch < warning < critical):
        st.error("Invalid thresholds. The required order is Watch < Warning < Critical.")
        st.stop()

    cfg = ModelConfig(
        contamination=contamination,
        alpha=alpha,
        watch=watch,
        warning=warning,
        critical=critical,
    )
    sensor, features = load_data()
    df = compute_anomaly_scores(sensor, features, cfg)
    metrics = evaluate_scores(df)
    alpha_df, ablation_df, noise_df = run_experiment_grid(sensor, features, cfg)
    scenario_df = scenario_detection(df)

    metric_cards(df, metrics)
    tabs = st.tabs(
        [
            "Mission dashboard",
            "Quantitative evaluation",
            "Sensitivity and ablation",
            "Robustness",
            "Human validation",
            "Data and reproducibility",
        ]
    )

    with tabs[0]:
        left, right = st.columns([1.2, 1])
        with left:
            fig = px.scatter_mapbox(
                df,
                lat="latitude",
                lon="longitude",
                color="status",
                size="ensemble_score",
                hover_data=["mission_minute", "simulated_event", "pm25", "pm10"],
                zoom=12,
                height=420,
                mapbox_style="open-street-map",
            )
            fig.update_layout(font=dict(size=13), legend_title_text="Status")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.dataframe(
                df[
                    [
                        "mission_minute",
                        "status",
                        "ensemble_score",
                        "pollutant_score",
                        "image_score",
                        "simulated_event",
                        "recommendation",
                    ]
                ]
                .sort_values("ensemble_score", ascending=False)
                .head(15),
                use_container_width=True,
                hide_index=True,
            )

        fig = go.Figure()
        for col, name in [
            ("pollutant_score", "Pollutant-weather"),
            ("image_score", "Image-feature"),
            ("ensemble_score", "Ensemble"),
        ]:
            fig.add_trace(go.Scatter(x=df.mission_minute, y=df[col], mode="lines", name=name))
        for y, name in [(watch, "Watch"), (warning, "Warning"), (critical, "Critical")]:
            fig.add_hline(y=y, annotation_text=name, line_width=1.4)
        fig.update_layout(
            title="Branch and ensemble anomaly scores",
            xaxis_title="Mission minute",
            yaxis_title="Normalized score",
            font=dict(size=14),
            legend_title_text="Score",
            height=470,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Agreement with known synthetic event labels")
        st.warning(
            "These metrics quantify agreement with controlled synthetic labels; "
            "they are not field-validation results."
        )
        metric_table = pd.DataFrame(
            {
                "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
                "Value": [
                    metrics["accuracy"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                    metrics["roc_auc"],
                    metrics["pr_auc"],
                ],
            }
        )
        st.dataframe(metric_table.style.format({"Value": "{:.4f}"}), use_container_width=True, hide_index=True)
        cm = pd.DataFrame(
            [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
            index=["Actual normal", "Actual anomaly"],
            columns=["Predicted normal", "Predicted anomaly"],
        )
        st.subheader("Confusion matrix")
        st.dataframe(cm, use_container_width=True)
        st.subheader("Scenario-level detection")
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("Alpha sensitivity")
        st.dataframe(alpha_df, use_container_width=True, hide_index=True)
        alpha_fig = px.line(
            alpha_df,
            x="alpha",
            y=["precision", "recall", "f1", "roc_auc"],
            markers=True,
            labels={"value": "Metric value", "variable": "Metric"},
        )
        alpha_fig.update_layout(font=dict(size=14), height=430)
        st.plotly_chart(alpha_fig, use_container_width=True)

        st.subheader("Modality ablation")
        st.dataframe(ablation_df, use_container_width=True, hide_index=True)
        ablation_fig = px.bar(
            ablation_df,
            x="configuration",
            y=["precision", "recall", "f1"],
            barmode="group",
            labels={"value": "Metric value", "variable": "Metric", "configuration": "Configuration"},
        )
        ablation_fig.update_layout(font=dict(size=14), height=430)
        st.plotly_chart(ablation_fig, use_container_width=True)

    with tabs[3]:
        st.subheader("Synthetic noise robustness")
        st.markdown(
            '<div class="section-note"><b>Interpretation:</b> binary anomaly detection and '
            "four-level operational status agreement are reported separately.</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(noise_df, use_container_width=True, hide_index=True)
        noise_fig = px.line(
            noise_df,
            x="noise_level",
            y=["f1", "roc_auc", "status_agreement"],
            markers=True,
            labels={"value": "Metric value", "variable": "Metric", "noise_level": "Noise level"},
        )
        noise_fig.update_layout(font=dict(size=14), height=430)
        st.plotly_chart(noise_fig, use_container_width=True)

    with tabs[4]:
        minute = st.selectbox(
            "Mission minute",
            df.sort_values("ensemble_score", ascending=False).mission_minute.astype(int).tolist(),
        )
        row = df[df.mission_minute == minute].iloc[0]
        c1, c2 = st.columns(2)
        with c1:
            st.image(str(BASE / row.image_file), caption=f"Minute {minute}: {row.simulated_event}")
        with c2:
            st.write(f"Model status: **{row.status}** | ensemble = **{row.ensemble_score:.3f}**")
            operator = st.text_input("Operator ID or initials", value="operator-01")
            decision = st.selectbox("Operator decision", ["Approve", "Correct", "Override", "Need field check"])
            corrected = st.selectbox(
                "Corrected status",
                ["Normal", "Watch", "Warning", "Critical"],
                index=["Normal", "Watch", "Warning", "Critical"].index(row.status),
            )
            notes = st.text_area(
                "Reason or notes",
                placeholder="Example: Reviewed the pollutant and image evidence; the Critical status is approved.",
            )
            if st.button("Save persistent validation"):
                save_validation(
                    {
                        "validation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "mission_minute": int(minute),
                        "operator_id": operator,
                        "simulated_event": row.simulated_event,
                        "model_status": row.status,
                        "corrected_status": corrected,
                        "operator_decision": decision,
                        "notes": notes,
                        "ensemble_score": row.ensemble_score,
                        "pollutant_score": row.pollutant_score,
                        "image_score": row.image_score,
                        "alpha": alpha,
                        "contamination": contamination,
                        "watch": watch,
                        "warning": warning,
                        "critical": critical,
                    }
                )
                st.success("Validation saved to outputs/human_validation_log.csv")

        if VALIDATIONS.exists():
            log = pd.read_csv(VALIDATIONS)
            st.subheader("Persistent validation log")
            st.dataframe(log, use_container_width=True, hide_index=True)
            st.download_button(
                "Download validation log",
                log.to_csv(index=False).encode(),
                file_name="human_validation_log.csv",
                mime="text/csv",
            )

    with tabs[5]:
        st.subheader("Model and experiment configuration")
        st.dataframe(configuration_table(cfg), use_container_width=True, hide_index=True)

        st.subheader("Data dictionary")
        if DATA_DICTIONARY.exists():
            dictionary_df = pd.read_csv(DATA_DICTIONARY)
            st.dataframe(dictionary_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download data dictionary CSV",
                dictionary_df.to_csv(index=False).encode(),
                file_name="data_dictionary.csv",
                mime="text/csv",
            )
        else:
            st.warning("data_dictionary.csv was not found.")

        st.subheader("Scored mission preview")
        st.dataframe(df.head(30), use_container_width=True, hide_index=True)

        st.subheader("Reproducibility exports")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                "Download scored mission CSV",
                df.to_csv(index=False).encode(),
                file_name="baseline_scored_mission.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download alpha sensitivity CSV",
                alpha_df.to_csv(index=False).encode(),
                file_name="alpha_sensitivity.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "Download modality ablation CSV",
                ablation_df.to_csv(index=False).encode(),
                file_name="modality_ablation.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download noise robustness CSV",
                noise_df.to_csv(index=False).encode(),
                file_name="noise_robustness.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d3:
            st.download_button(
                "Download scenario detection CSV",
                scenario_df.to_csv(index=False).encode(),
                file_name="scenario_detection.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if VALIDATIONS.exists():
                validation_df = pd.read_csv(VALIDATIONS)
                st.download_button(
                    "Download validation log CSV",
                    validation_df.to_csv(index=False).encode(),
                    file_name="human_validation_log.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        st.info(
            "All inputs are synthetic. Digital Twin, data lakehouse, and distributed edge-cloud operation "
            "remain architectural-readiness components rather than fully implemented services."
        )

    st.markdown(
        f'<div class="footer-note">Prototype version {APP_VERSION} · Controlled synthetic evaluation · '
        "Generated outputs are intended for reproducibility and framework-feasibility analysis.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
