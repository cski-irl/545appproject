"""
Streamlit app for the next-session injury-risk model from 545_Model__Project_.ipynb.

The model is the notebook's tuned pipeline (median imputation + scaling for numeric
columns, most-frequent imputation + one-hot for categorical columns, then a
200-tree RandomForest), exported to `model_export.npz` as plain NumPy arrays.
Predictions are computed here with NumPy, so the app does not depend on any
particular scikit-learn or Python version.

Run:  streamlit run streamlit_app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_FILE = APP_DIR / "model_export.npz"

# Same engineered features as the notebook (cell 3).
ROLL_COLS = ["recovery_score", "sleep_quality", "fatigue_index", "training_load"]


# --------------------------------------------------------------------------------------
# Model: NumPy re-implementation of the fitted sklearn pipeline
# --------------------------------------------------------------------------------------
class InjuryRiskModel:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        self.numeric_features = meta["numeric_features"]
        self.categorical_features = meta["categorical_features"]
        self.categories = meta["categories"]
        self.cat_fill = meta["cat_fill"]
        self.medians, self.mean, self.scale = d["medians"], d["mean"], d["scale"]
        self.offsets = d["offsets"]
        self.left, self.right = d["left"], d["right"]
        self.feature, self.threshold = d["feature"], d["threshold"]
        self.leaf_proba = d["proba"]

    @property
    def features(self):
        return self.numeric_features + self.categorical_features

    def _transform(self, df):
        # Numeric: median-impute, then standard-scale.
        Xn = df[self.numeric_features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        Xn = np.where(np.isnan(Xn), self.medians, Xn)
        Xn = (Xn - self.mean) / self.scale
        # Categorical: most-frequent impute, then one-hot (unknown values -> all zeros).
        blocks = [Xn]
        for col, cats, fill in zip(self.categorical_features, self.categories, self.cat_fill):
            vals = df[col].where(df[col].notna(), fill).tolist()
            oh = np.zeros((len(df), len(cats)))
            for i, v in enumerate(vals):
                for j, c in enumerate(cats):
                    if v == c or str(v) == str(c):
                        oh[i, j] = 1.0
                        break
            blocks.append(oh)
        # sklearn trees compare in float32.
        return np.hstack(blocks).astype(np.float32)

    def predict_proba(self, df):
        X = self._transform(df)
        n = len(X)
        total = np.zeros((n, 2))
        rows = np.arange(n)
        for t in range(len(self.offsets) - 1):
            base = self.offsets[t]
            node = np.zeros(n, dtype=np.int64)
            while True:
                g = base + node
                is_split = self.left[g] != -1
                if not is_split.any():
                    break
                f = self.feature[g]
                go_left = X[rows, np.where(is_split, f, 0)] <= self.threshold[g].astype(np.float32)
                nxt = np.where(go_left, self.left[g], self.right[g])
                node = np.where(is_split, nxt, node)
            total += self.leaf_proba[base + node]
        return total / (len(self.offsets) - 1)

    def predict(self, df):
        return (self.predict_proba(df)[:, 1] > 0.5).astype(int)


@st.cache_resource
def load_model(path):
    return InjuryRiskModel(path)


if not MODEL_FILE.exists():
    st.error(f"`{MODEL_FILE.name}` not found. Put it in the same folder as this script.")
    st.stop()

model = load_model(str(MODEL_FILE))
numeric_features = model.numeric_features
categorical_features = model.categorical_features
FEATURES = model.features
medians = dict(zip(numeric_features, model.medians))
stds = dict(zip(numeric_features, model.scale))
cat_options = dict(zip(categorical_features, model.categories))

base_numeric = [c for c in numeric_features if not c.endswith(("_roll5", "_trend3"))]
INT_COLS = {"step_count", "age"}


def pretty(col):
    return col.replace("_", " ").title()


def add_history_features(hist):
    """Recreate the notebook's rolling means and trends for one athlete's history."""
    hist = hist.copy()
    if "session_id" in hist.columns:
        hist = hist.sort_values("session_id")
    hist = hist.reset_index(drop=True)
    for col in ROLL_COLS:
        hist[f"{col}_roll5"] = hist[col].rolling(5, min_periods=1).mean()
        hist[f"{col}_trend3"] = (hist[col] - hist[col].shift(3)).fillna(0)
    return hist


def show_prediction(row):
    proba = float(model.predict_proba(row[FEATURES])[0, 1])
    pred = int(proba > 0.5)
    st.subheader("Prediction for the next session")
    c1, c2 = st.columns(2)
    c1.metric("Probability of high-risk next session", f"{proba:.1%}")
    c2.metric("Predicted class", "HIGH RISK" if pred == 1 else "Not high risk")
    st.progress(min(max(proba, 0.0), 1.0))
    st.caption(
        "High risk = the athlete's next session has injury_occurred == 2. "
        "Class uses the model's default 0.5 cut-off (the model was trained with "
        "balanced class weights, so probabilities run higher than the ~15% base rate)."
    )


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
st.title("Athlete Next-Session Injury Risk Predictor")
st.write(
    """
    Predicts whether an athlete's **next** training session will be **high risk**,
    using a tuned Random Forest trained on the Multimodal Sports Injury dataset.
    """
)

mode = st.radio(
    "Input method",
    ["Enter one session manually", "Upload an athlete's session history (CSV)"],
    horizontal=True,
)

# ---- Mode 1: manual entry ------------------------------------------------------------
if mode.startswith("Enter"):
    st.write("Defaults are the training-set medians.")
    values = {}

    st.markdown("#### Athlete & session context")
    cols = st.columns(len(categorical_features))
    for col, box in zip(categorical_features, cols):
        values[col] = box.selectbox(pretty(col), cat_options[col])

    st.markdown("#### Current session measurements")
    grid = st.columns(3)
    for i, col in enumerate(base_numeric):
        default = float(medians[col])
        step = float(stds[col]) / 10 if stds[col] > 0 else 0.1
        with grid[i % 3]:
            if col in INT_COLS:
                values[col] = st.number_input(pretty(col), value=int(round(default)), step=1)
            else:
                values[col] = st.number_input(
                    pretty(col), value=round(default, 3), step=round(step, 3) or 0.001,
                    format="%.3f",
                )

    with st.expander("Recent history (rolling averages and 3-session trends)"):
        st.write(
            "By default these assume no history: each 5-session average equals the "
            "current value and each trend is 0 (same as an athlete's first session in "
            "the notebook). Override if you know the athlete's recent sessions."
        )
        override = st.checkbox("Enter history values manually")
        for col in ROLL_COLS:
            r, t = f"{col}_roll5", f"{col}_trend3"
            if override:
                a, b = st.columns(2)
                values[r] = a.number_input(f"{pretty(col)} — 5-session avg",
                                           value=float(values[col]), format="%.3f")
                values[t] = b.number_input(f"{pretty(col)} — change vs 3 sessions ago",
                                           value=0.0, format="%.3f")
            else:
                values[r] = float(values[col])
                values[t] = 0.0

    row = pd.DataFrame([values])
    show_prediction(row)

# ---- Mode 2: CSV history -------------------------------------------------------------
else:
    st.write(
        "Upload a CSV of **one athlete's** sessions with the original dataset columns "
        f"({', '.join(base_numeric + categorical_features)}), plus `session_id` for ordering. "
        "Rolling averages and trends are computed as in the notebook, and the prediction "
        "is for the session after the last row."
    )
    file = st.file_uploader("Session history CSV", type="csv")
    if file is not None:
        hist = pd.read_csv(file)
        needed = base_numeric + categorical_features
        missing = [c for c in needed if c not in hist.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()
        if "athlete_id" in hist.columns and hist["athlete_id"].nunique() > 1:
            athlete = st.selectbox("Athlete", sorted(hist["athlete_id"].unique()))
            hist = hist[hist["athlete_id"] == athlete]

        hist = add_history_features(hist)
        st.dataframe(hist.tail(5), use_container_width=True)
        show_prediction(hist.tail(1))
