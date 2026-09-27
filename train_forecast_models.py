"""
Lag-based 1-day and 7-day forecasting of PM2.5 and PM10.

Models: persistence baseline, Random Forest, XGBoost, Support Vector Regression.
Station currently available: GM Office, Brajrajnagar (OSPCB), Jharsuguda coal belt, Odisha.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "cleaned_data.xlsx"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"

LAGS = [0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 30]
ROLL_WINDOWS = [3, 7, 14]
HORIZONS = [1, 7]
POLLUTANTS = {"pm25": "PM2.5", "pm10": "PM10"}
TEST_START = pd.Timestamp("2024-01-01")
VAL_START = pd.Timestamp("2023-01-01")
RANDOM_STATE = 42
MAX_INTERP_GAP = 3
PM25_CAP = 500.0
PM10_CAP = 800.0

REGION_LABEL = "Brajrajnagar"


def _match_column(columns: list[str], *needles: str) -> str | None:
    for col in columns:
        key = col.lower().replace(" ", "").replace("_", "")
        if all(n.replace("_", "") in key for n in needles):
            return col
    return None


def load_cleaned(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Cleaned_Data")
    cols = list(df.columns)
    ts_col = _match_column(cols, "timestamp") or cols[0]
    pm25_col = _match_column(cols, "pm2", "5") or cols[1]
    pm10_col = _match_column(cols, "pm10") or cols[2]

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[ts_col], errors="coerce"),
            "pm25": pd.to_numeric(df[pm25_col], errors="coerce"),
            "pm10": pd.to_numeric(df[pm10_col], errors="coerce"),
        }
    )
    out["region"] = REGION_LABEL
    out = out.dropna(subset=["timestamp"])
    out = out.sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    return out.reset_index(drop=True)


def quality_control(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["pm25"] < 0, "pm25"] = np.nan
    df.loc[df["pm10"] < 0, "pm10"] = np.nan
    df.loc[df["pm25"] > PM25_CAP, "pm25"] = np.nan
    df.loc[df["pm10"] > PM10_CAP, "pm10"] = np.nan
    # Daily PM2.5 cannot exceed PM10; treat clear inversions as missing PM2.5.
    inversion = df["pm25"].notna() & df["pm10"].notna() & (df["pm25"] > df["pm10"] * 1.05)
    df.loc[inversion, "pm25"] = np.nan
    return df


def to_daily_series(df: pd.DataFrame, region: str) -> pd.DataFrame:
    g = df.loc[df["region"] == region, ["timestamp", "pm25", "pm10"]].copy()
    g = g.set_index("timestamp").sort_index()
    g = g[~g.index.duplicated(keep="last")].asfreq("D")
    for col in ("pm25", "pm10"):
        g[col] = g[col].interpolate(method="time", limit=MAX_INTERP_GAP)
    g["region"] = region
    return g


def add_lag_features(daily: pd.DataFrame) -> pd.DataFrame:
    feat = daily.copy()
    doy = feat.index.dayofyear
    feat["month"] = feat.index.month
    feat["weekday"] = feat.index.weekday
    feat["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feat["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    for col in ("pm25", "pm10"):
        for lag in LAGS:
            feat[f"{col}_lag{lag}"] = feat[col].shift(lag)
        for win in ROLL_WINDOWS:
            feat[f"{col}_rollmean{win}"] = feat[col].shift(1).rolling(win, min_periods=win).mean()
            feat[f"{col}_rollstd{win}"] = feat[col].shift(1).rolling(win, min_periods=win).std()
        feat[f"{col}_delta1"] = feat[col] - feat[col].shift(1)

    for h in HORIZONS:
        feat[f"pm25_target_h{h}"] = feat["pm25"].shift(-h)
        feat[f"pm10_target_h{h}"] = feat["pm10"].shift(-h)
    return feat


def feature_columns(frame: pd.DataFrame) -> list[str]:
    drop = {"pm25", "pm10", "region"}
    targets = [c for c in frame.columns if "_target_h" in c]
    return [c for c in frame.columns if c not in drop and c not in targets]


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mean_obs = float(np.mean(y_true)) if len(y_true) else np.nan
    return {
        "n": int(len(y_true)),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "nrmse": round(rmse / mean_obs, 4) if mean_obs else np.nan,
        "nmae": round(mae / mean_obs, 4) if mean_obs else np.nan,
        "mean_obs": round(mean_obs, 4),
    }


def split_xy(feat: pd.DataFrame, pollutant: str, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    target = f"{pollutant}_target_h{horizon}"
    cols = feature_columns(feat)
    data = feat[cols + [target]].dropna()
    return data[cols], data[target]


def time_masks(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = index < VAL_START
    val = (index >= VAL_START) & (index < TEST_START)
    test = index >= TEST_START
    return train, val, test


def build_models(n_features: int) -> dict:
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb = XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=4,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=40,
        verbosity=0,
    )
    svr = TransformedTargetRegressor(
        regressor=Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svr",
                    SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.08, cache_size=500),
                ),
            ]
        ),
        transformer=StandardScaler(),
    )
    return {"RandomForest": rf, "XGBoost": xgb, "SVR": svr}


def fit_model(name: str, model, X_train, y_train, X_val, y_val):
    if name == "XGBoost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        X_fit = pd.concat([X_train, X_val])
        y_fit = pd.concat([y_train, y_val])
        model.fit(X_fit, y_fit)
    return model


def native_importance(name: str, model, columns: list[str]) -> pd.DataFrame | None:
    if name == "RandomForest":
        values = model.feature_importances_
    elif name == "XGBoost":
        values = model.feature_importances_
    else:
        return None
    return pd.DataFrame({"feature": columns, "importance": values, "source": "native"})


def plot_pred_vs_obs(dates, y_true, y_pred, title: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [1.4, 1]})
    axes[0].plot(dates, y_true, label="Observed", color="#1f4e79", linewidth=1.1)
    axes[0].plot(dates, y_pred, label="Forecast", color="#c45911", linewidth=1.1, alpha=0.9)
    axes[0].set_ylabel("Concentration (µg/m³)")
    axes[0].set_title(title)
    axes[0].legend(frameon=False)
    axes[1].scatter(y_true, y_pred, s=12, alpha=0.55, color="#1f4e79")
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    axes[1].plot([lo, hi], [lo, hi], color="#c45911", linewidth=1)
    axes[1].set_xlabel("Observed (µg/m³)")
    axes[1].set_ylabel("Forecast (µg/m³)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_metrics(metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, pollutant in zip(axes, ["pm25", "pm10"]):
        sub = metrics[(metrics["pollutant"] == pollutant) & (metrics["model"] != "Persistence")]
        pivot = sub.pivot_table(index="model", columns="horizon", values="rmse")
        pivot.plot(kind="bar", ax=ax, color=["#1f4e79", "#c45911"], rot=0)
        ax.set_title(f"{POLLUTANTS[pollutant]} test RMSE (µg/m³)")
        ax.set_xlabel("")
        ax.set_ylabel("RMSE")
        ax.legend(title="Horizon (days)", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, path: Path) -> None:
    top = (
        imp.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#1f4e79")
    ax.set_xlabel("Mean importance (tree models, all tasks)")
    ax.set_title("Most influential historical predictors")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_region(daily: pd.DataFrame, region: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feat = add_lag_features(daily)
    metric_rows = []
    importance_rows = []
    prediction_rows = []

    for pollutant in POLLUTANTS:
        for horizon in HORIZONS:
            X, y = split_xy(feat, pollutant, horizon)
            train_m, val_m, test_m = time_masks(X.index)
            if train_m.sum() < 200 or test_m.sum() < 50:
                print(f"  skip {pollutant} h={horizon}: too few samples")
                continue

            X_train, y_train = X.loc[train_m], y.loc[train_m]
            X_val, y_val = X.loc[val_m], y.loc[val_m]
            X_test, y_test = X.loc[test_m], y.loc[test_m]

            persist = X_test[f"{pollutant}_lag0"].to_numpy()
            persist_metrics = metrics_dict(y_test.to_numpy(), persist)
            metric_rows.append(
                {
                    "region": region,
                    "pollutant": pollutant,
                    "horizon": horizon,
                    "model": "Persistence",
                    **persist_metrics,
                }
            )

            models = build_models(X_train.shape[1])
            for name, model in models.items():
                print(f"  {region} | {POLLUTANTS[pollutant]} | h={horizon} | {name}")
                fitted = fit_model(name, model, X_train, y_train, X_val, y_val)
                pred = fitted.predict(X_test)
                scores = metrics_dict(y_test.to_numpy(), pred)
                metric_rows.append(
                    {
                        "region": region,
                        "pollutant": pollutant,
                        "horizon": horizon,
                        "model": name,
                        **scores,
                    }
                )

                model_path = MODEL_DIR / f"{region}_{pollutant}_h{horizon}_{name}.joblib"
                joblib.dump({"model": fitted, "features": list(X.columns)}, model_path)

                nat = native_importance(name, fitted, list(X.columns))
                if nat is not None:
                    nat["region"] = region
                    nat["pollutant"] = pollutant
                    nat["horizon"] = horizon
                    nat["model"] = name
                    importance_rows.append(nat)

                perm = permutation_importance(
                    fitted,
                    X_test,
                    y_test,
                    n_repeats=6,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    scoring="neg_root_mean_squared_error",
                )
                perm_df = pd.DataFrame(
                    {
                        "feature": X.columns,
                        "importance": perm.importances_mean,
                        "source": "permutation_test",
                        "region": region,
                        "pollutant": pollutant,
                        "horizon": horizon,
                        "model": name,
                    }
                )
                importance_rows.append(perm_df)

                pred_df = pd.DataFrame(
                    {
                        "timestamp": X_test.index,
                        "region": region,
                        "pollutant": pollutant,
                        "horizon": horizon,
                        "model": name,
                        "observed": y_test.to_numpy(),
                        "predicted": pred,
                    }
                )
                prediction_rows.append(pred_df)

                plot_pred_vs_obs(
                    X_test.index,
                    y_test.to_numpy(),
                    pred,
                    f"{region} {POLLUTANTS[pollutant]} {horizon}-day {name}",
                    FIG_DIR / f"{region}_{pollutant}_h{horizon}_{name}.png",
                )

    metrics = pd.DataFrame(metric_rows)
    importance = pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame()
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    return metrics, importance, predictions


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_cleaned(DATA_PATH)
    raw = quality_control(raw)
    regions = sorted(raw["region"].dropna().unique())

    all_metrics = []
    all_importance = []
    all_predictions = []
    coverage = []

    for region in regions:
        daily = to_daily_series(raw, region)
        valid = daily[["pm25", "pm10"]].notna().all(axis=1).sum()
        coverage.append(
            {
                "region": region,
                "start": str(daily.index.min().date()),
                "end": str(daily.index.max().date()),
                "calendar_days": int(len(daily)),
                "complete_pm_days": int(valid),
                "pm25_mean": round(float(daily["pm25"].mean()), 3),
                "pm10_mean": round(float(daily["pm10"].mean()), 3),
            }
        )
        print(f"\n=== {region}: {daily.index.min().date()} to {daily.index.max().date()} ===")
        metrics, importance, predictions = run_region(daily, region)
        all_metrics.append(metrics)
        all_importance.append(importance)
        all_predictions.append(predictions)

    metrics = pd.concat(all_metrics, ignore_index=True)
    importance = pd.concat([d for d in all_importance if len(d)], ignore_index=True)
    predictions = pd.concat([d for d in all_predictions if len(d)], ignore_index=True)

    metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
    importance.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    predictions.to_csv(OUT_DIR / "predictions_test.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT_DIR / "station_coverage.csv", index=False)

    plot_metrics(metrics, FIG_DIR / "rmse_by_model.png")
    tree_imp = importance[(importance["source"] == "native") & (importance["model"] != "SVR")]
    if len(tree_imp):
        plot_importance(tree_imp, FIG_DIR / "feature_importance_top15.png")

    ranking = (
        metrics[metrics["model"] != "Persistence"]
        .sort_values(["pollutant", "horizon", "rmse"])
        .groupby(["region", "pollutant", "horizon"], as_index=False)
        .first()
    )
    ranking.to_csv(OUT_DIR / "best_model_by_task.csv", index=False)

    summary = {
        "objectives": {
            "lag_features": LAGS,
            "rolling_windows": ROLL_WINDOWS,
            "horizons_days": HORIZONS,
            "models": ["Persistence", "RandomForest", "XGBoost", "SVR"],
            "test_start": str(TEST_START.date()),
            "regions_in_file": regions,
        },
        "note": (
            "cleaned_data.xlsx currently contains a single OSPCB station "
            "(GM Office, Brajrajnagar). Additional mining-region folders under "
            "data/ can be merged into the same schema for multi-region comparison."
        ),
        "best_models": ranking.to_dict(orient="records"),
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nWrote results to", OUT_DIR)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
