from __future__ import annotations

import io
import json
import os
import base64
import traceback
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from flask import (render_template, request, jsonify, redirect,
                   url_for, flash)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from neurix.datalab import datalab

# ── Config ────────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"csv"}
MAX_ROWS = 50_000       # safety cap
MAX_COLS = 50

PALETTE = {
    "primary":  "#1a2535",
    "accent":   "#b84c27",
    "green":    "#27a35a",
    "bg":       "#f7f5f0",
    "grid":     "#e5e0d8",
}

sns.set_theme(style="whitegrid", palette="muted")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _parse_csv(file_storage) -> pd.DataFrame:
    df = pd.read_csv(file_storage, nrows=MAX_ROWS)
    if len(df.columns) > MAX_COLS:
        df = df.iloc[:, :MAX_COLS]
    return df


def _df_summary(df: pd.DataFrame) -> Dict[str, Any]:
    num_cols   = df.select_dtypes(include="number").columns.tolist()
    cat_cols   = df.select_dtypes(exclude="number").columns.tolist()
    missing    = int(df.isnull().sum().sum())
    return {
        "rows":       len(df),
        "cols":       len(df.columns),
        "num_cols":   num_cols,
        "cat_cols":   cat_cols,
        "missing":    missing,
        "columns":    df.columns.tolist(),
        "dtypes":     {c: str(t) for c, t in df.dtypes.items()},
        "head":       df.head(8).fillna("").to_dict(orient="records"),
    }


# ── Chart generators ──────────────────────────────────────────────────────────

def _correlation_heatmap(df: pd.DataFrame) -> str | None:
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return None

    corr = num.corr()
    n = len(corr)
    size = max(5, min(n * 0.9, 12))

    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    cmap = sns.diverging_palette(220, 15, as_cmap=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
        annot=True, fmt=".2f", annot_kws={"size": max(7, 10 - n // 4)},
        linewidths=.5, linecolor=PALETTE["grid"],
        ax=ax, cbar_kws={"shrink": .8}
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13,
                 fontweight="bold", color=PALETTE["primary"], pad=12)
    ax.tick_params(colors=PALETTE["primary"], labelsize=9)
    return _fig_to_b64(fig)


def _distribution_plots(df: pd.DataFrame) -> str | None:
    num = df.select_dtypes(include="number")
    if num.empty:
        return None

    cols  = num.columns.tolist()[:12]   # cap at 12
    ncols = min(3, len(cols))
    nrows = (len(cols) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.2, nrows * 3.2))
    fig.patch.set_facecolor(PALETTE["bg"])
    axes_flat = np.array(axes).flatten() if hasattr(axes, "__len__") else [axes]

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        ax.set_facecolor("#fff")
        data = num[col].dropna()
        ax.hist(data, bins=30, color=PALETTE["accent"],
                alpha=.75, edgecolor="white", linewidth=.4)
        ax.set_title(col, fontsize=9, color=PALETTE["primary"], fontweight="bold")
        ax.set_xlabel("", fontsize=7)
        ax.tick_params(labelsize=7, colors="#6b7280")
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["grid"])

        # Overlay KDE
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            xs  = np.linspace(data.min(), data.max(), 200)
            ax2 = ax.twinx()
            ax2.plot(xs, kde(xs), color=PALETTE["primary"], lw=1.5, alpha=.7)
            ax2.set_yticks([])
            ax2.set_facecolor("none")
        except Exception:
            pass

    for j in range(len(cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=13,
                 fontweight="bold", color=PALETTE["primary"], y=1.01)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _feature_importance_plot(feature_names, importances, title="Feature Importances") -> str:
    idx   = np.argsort(importances)[::-1][:20]
    names = [feature_names[i] for i in idx]
    vals  = importances[idx]

    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.45)))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    colors = [PALETTE["accent"] if v == vals.max() else PALETTE["primary"] for v in vals]
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], alpha=.85, height=.6)
    ax.set_xlabel("Importance", fontsize=9, color="#6b7280")
    ax.set_title(title, fontsize=12, fontweight="bold", color=PALETTE["primary"])
    ax.tick_params(labelsize=8, colors=PALETTE["primary"])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_edgecolor(PALETTE["grid"])
    ax.spines["bottom"].set_edgecolor(PALETTE["grid"])

    fig.tight_layout()
    return _fig_to_b64(fig)


def _confusion_matrix_plot(cm, class_labels) -> str:
    fig, ax = plt.subplots(figsize=(max(4, len(class_labels)), max(3.5, len(class_labels) * .8)))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels,
                linewidths=.5, linecolor=PALETTE["grid"], ax=ax,
                annot_kws={"size": 10})
    ax.set_xlabel("Predicted", fontsize=10, color=PALETTE["primary"])
    ax.set_ylabel("Actual", fontsize=10, color=PALETTE["primary"])
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold", color=PALETTE["primary"])
    ax.tick_params(labelsize=8, colors=PALETTE["primary"])
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── Routes ────────────────────────────────────────────────────────────────────

@datalab.route("/datalab")
@login_required
def index():
    return render_template("datalab/index.html", title="Dataset Lab")


@datalab.route("/datalab/upload", methods=["POST"])
@login_required
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename or not _allowed(f.filename):
        return jsonify({"error": "Please upload a .csv file"}), 400

    try:
        df = _parse_csv(f)
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {e}"}), 400

    if df.empty:
        return jsonify({"error": "The uploaded file is empty"}), 400

    summary = _df_summary(df)

    # Store serialised CSV in session (small DFs only; large ones → temp file)
    csv_b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()

    # Generate EDA charts
    corr_img  = _correlation_heatmap(df)
    dist_img  = _distribution_plots(df)

    return jsonify({
        "summary":   summary,
        "csv_b64":   csv_b64,
        "corr_img":  corr_img,
        "dist_img":  dist_img,
    })


@datalab.route("/datalab/train", methods=["POST"])
@login_required
def train():
    data = request.json or {}

    csv_b64    = data.get("csv_b64", "")
    target_col = data.get("target", "")
    model_type = data.get("model", "auto")
    feature_cols = data.get("features", [])  # empty = use all numeric

    if not csv_b64 or not target_col:
        return jsonify({"error": "Missing data or target column"}), 400

    try:
        csv_bytes = base64.b64decode(csv_b64)
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as e:
        return jsonify({"error": f"Could not decode dataset: {e}"}), 400

    if target_col not in df.columns:
        return jsonify({"error": f"Column '{target_col}' not found"}), 400

    # ── Prepare features ──────────────────────────────────────────────────────
    y = df[target_col].dropna()
    df = df.loc[y.index]

    if feature_cols:
        X_raw = df[feature_cols]
    else:
        X_raw = df.drop(columns=[target_col]).select_dtypes(include="number")

    if X_raw.empty:
        return jsonify({"error": "No numeric feature columns found"}), 400

    X = X_raw.fillna(X_raw.median(numeric_only=True))
    y = df[target_col]

    # ── Detect task type ──────────────────────────────────────────────────────
    n_unique = y.nunique()
    is_classification = (
        y.dtype == object
        or n_unique <= 20
        or model_type in ("logistic", "random_forest_clf", "knn")
    )
    if model_type in ("linear", "random_forest_reg"):
        is_classification = False

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.metrics import (
            accuracy_score, classification_report,
            confusion_matrix, mean_squared_error, r2_score
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        le = None
        if is_classification and y.dtype == object:
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test  = le.transform(y_test)

        # ── Select model ──────────────────────────────────────────────────────
        def _pick_model():
            if is_classification:
                if model_type == "logistic" or model_type == "auto":
                    from sklearn.linear_model import LogisticRegression
                    return LogisticRegression(max_iter=500, random_state=42), "Logistic Regression"
                elif model_type == "random_forest_clf":
                    from sklearn.ensemble import RandomForestClassifier
                    return RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest (Classifier)"
                elif model_type == "knn":
                    from sklearn.neighbors import KNeighborsClassifier
                    return KNeighborsClassifier(n_neighbors=5), "K-Nearest Neighbours"
                elif model_type == "svm":
                    from sklearn.svm import SVC
                    return SVC(probability=True, random_state=42), "Support Vector Machine"
                else:
                    from sklearn.linear_model import LogisticRegression
                    return LogisticRegression(max_iter=500, random_state=42), "Logistic Regression"
            else:
                if model_type == "linear" or model_type == "auto":
                    from sklearn.linear_model import LinearRegression
                    return LinearRegression(), "Linear Regression"
                elif model_type == "random_forest_reg":
                    from sklearn.ensemble import RandomForestRegressor
                    return RandomForestRegressor(n_estimators=100, random_state=42), "Random Forest (Regressor)"
                elif model_type == "ridge":
                    from sklearn.linear_model import Ridge
                    return Ridge(alpha=1.0), "Ridge Regression"
                else:
                    from sklearn.linear_model import LinearRegression
                    return LinearRegression(), "Linear Regression"

        model, model_name = _pick_model()
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        # ── Metrics & charts ──────────────────────────────────────────────────
        result: Dict[str, Any] = {
            "model_name":       model_name,
            "task":             "classification" if is_classification else "regression",
            "n_train":          len(X_train),
            "n_test":           len(X_test),
            "feature_count":    X.shape[1],
            "feature_names":    X.columns.tolist(),
        }

        importance_img = None
        conf_img       = None

        if is_classification:
            acc = accuracy_score(y_test, y_pred)
            result["accuracy"] = round(float(acc), 4)

            labels = le.classes_.tolist() if le else sorted(y.unique().tolist())
            cm = confusion_matrix(y_test, y_pred)
            conf_img = _confusion_matrix_plot(cm, [str(l) for l in labels])
            result["confusion_matrix_img"] = conf_img

            # Classification report as dict
            cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            result["classification_report"] = cr

        else:
            mse_val = mean_squared_error(y_test, y_pred)
            r2_val  = r2_score(y_test, y_pred)
            result["mse"]  = round(float(mse_val), 4)
            result["rmse"] = round(float(np.sqrt(mse_val)), 4)
            result["r2"]   = round(float(r2_val), 4)

            # Actual vs Predicted scatter
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            fig.patch.set_facecolor(PALETTE["bg"])
            ax.set_facecolor(PALETTE["bg"])
            ax.scatter(y_test, y_pred, alpha=.55, color=PALETTE["accent"],
                       edgecolors="white", linewidth=.3, s=35)
            mn = min(float(y_test.min()), float(y_pred.min()))
            mx = max(float(y_test.max()), float(y_pred.max()))
            ax.plot([mn, mx], [mn, mx], "--", color=PALETTE["primary"], lw=1.5, alpha=.7)
            ax.set_xlabel("Actual", fontsize=9, color="#6b7280")
            ax.set_ylabel("Predicted", fontsize=9, color="#6b7280")
            ax.set_title("Actual vs Predicted", fontsize=11,
                         fontweight="bold", color=PALETTE["primary"])
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            fig.tight_layout()
            result["actual_vs_pred_img"] = _fig_to_b64(fig)

        # Feature importance
        if hasattr(model, "feature_importances_"):
            importance_img = _feature_importance_plot(
                X.columns.tolist(), model.feature_importances_
            )
        elif hasattr(model, "coef_"):
            coefs = model.coef_
            if coefs.ndim > 1:
                coefs = np.abs(coefs).mean(axis=0)
            importance_img = _feature_importance_plot(
                X.columns.tolist(), np.abs(coefs),
                title="Coefficient Magnitudes (Feature Importance)"
            )

        result["importance_img"] = importance_img
        return jsonify(result)

    except Exception:
        tb = traceback.format_exc()
        return jsonify({"error": f"Training failed:\n{tb}"}), 500
