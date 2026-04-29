# Theory of Machine Learning - Spring 2026
# Final Project - Perry, Penner, 

import argparse
import math
import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader, TensorDataset


# Seeding for reproducibility
# -----------------------------
def set_seed(seed: int = 37) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Data loading / cleaning
# -----------------------------
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    target_col = "Chance of Admit"
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' not found. Columns: {df.columns.tolist()}")

    drop_cols = [c for c in df.columns if c.lower().startswith("serial")]
    if drop_cols:
        print(f"[INFO] Dropping non-predictive column(s): {drop_cols}")
        df = df.drop(columns=drop_cols)

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(float)

    print("[INFO] Dataset loaded successfully.")
    print(f"[INFO] Samples: {len(df)}, Features: {X.shape[1]}")
    print("[INFO] Feature columns:")
    for c in X.columns:
        print(f"  - {c}")
    print(f"[INFO] Target range: min = {y.min():.3f}, max = {y.max():.3f}, mean = {y.mean():.3f}")
    print()
    return X, y


# Metrics
# -----------------------------
@dataclass
class ModelResults:
    name: str
    mae: float
    rmse: float
    r2: float
    mape: float
    y_pred: np.ndarray
    extra: dict


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Safe Mean Absolute Percentage Error for very small regression targets.
    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0
    return mae, rmse, r2, mape


def print_metrics_block(title: str, y_true, y_pred):
    mae, rmse, r2, mape = compute_metrics(y_true, y_pred)
    print(f"[{title}] MAE  = {mae:.5f}")
    print(f"[{title}] RMSE = {rmse:.5f}")
    print(f"[{title}] R^2  = {r2:.5f}")
    print(f"[{title}] MAPE = {mape:.3f}%")
    return mae, rmse, r2, mape


# SVR Model
# -----------------------------
def build_svr_pipeline(features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipeline, features)],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("svr", SVR(kernel="rbf", C=30.0, epsilon=0.03, gamma="scale")),
        ]
    )
    return model


# Random Forest Regressor Model
# -----------------------------
def build_rf_pipeline(features):
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipeline, features)],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=350,
                    max_depth=10,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    random_state=37,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return model


def evaluate_sklearn_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\n---------- Training {name} ----------")

    cv = KFold(n_splits=5, shuffle=True, random_state=37)
    cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=None)
    cv_rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error", n_jobs=None))

    print(f"[{name}] 5-Fold CV R^2 scores : {np.round(cv_r2, 4)}")
    print(f"[{name}] 5-Fold CV R^2 mean +/- std: {cv_r2.mean():.5f} +/- {cv_r2.std():.5f}")
    print(f"[{name}] 5-Fold CV RMSE scores: {np.round(cv_rmse, 4)}")
    print(f"[{name}] 5-Fold CV RMSE mean +/- std: {cv_rmse.mean():.5f} +/- {cv_rmse.std():.5f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae, rmse, r2, mape = print_metrics_block(name, y_test, y_pred)

    extra = {
        "cv_r2_mean": cv_r2.mean(),
        "cv_r2_std": cv_r2.std(),
        "cv_rmse_mean": cv_rmse.mean(),
        "cv_rmse_std": cv_rmse.std(),
    }

    if name == "Random Forest Regressor":
        rf = model.named_steps["rf"]
        feature_names = X_train.columns.tolist()
        feature_importances = sorted(
            zip(feature_names, rf.feature_importances_), key=lambda x: x[1], reverse=True
        )
        extra["feature_importances"] = feature_importances
        print(f"[{name}] Feature importances:")
        for fname, score in feature_importances:
            print(f"  - {fname:<20s}: {score:.5f}")

    return ModelResults(name=name, mae=mae, rmse=rmse, r2=r2, mape=mape, y_pred=y_pred, extra=extra)


# PyTorch FFNN
# -----------------------------
class FeedForwardRegressor(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_ffnn(X_train, X_test, y_train, y_test, output_dir, epochs=220, batch_size=32, lr=1e-3, dropout=0.25):
    print("\n---------- Training FFNN with Dropout ----------")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create validation split from training set.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train.values, test_size=0.15, random_state=37
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FFNN] Using device: {device}")

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr.reshape(-1, 1), dtype=torch.float32),
    )
    val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(device)
    test_x = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = FeedForwardRegressor(input_dim=X_train.shape[1], dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses = []
    val_losses = []
    best_state = None
    best_val_loss = float("inf")
    patience = 30
    retries = 0

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        avg_train_loss = float(np.mean(batch_losses))
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_preds = model(val_x)
            val_loss = criterion(val_preds, val_y).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            retries = 0
        else:
            retries += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[FFNN] Epoch {epoch:03d}/{epochs} | "
                f"Train MSE: {avg_train_loss:.6f} | Val MSE: {val_loss:.6f}"
            )

        if retries >= patience:
            print(f"[FFNN] Early stopping triggered at epoch {epoch}!")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred = model(test_x).cpu().numpy().ravel()

    mae, rmse, r2, mape = print_metrics_block("FFNN with Dropout", y_test, y_pred)

    # Plot training curve in output folder
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses, label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("FFNN Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ffnn_training_curve.png"), dpi=200)
    plt.close()

    extra = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "epochs_ran": len(train_losses),
    }

    return ModelResults(name="FFNN with Dropout", mae=mae, rmse=rmse, r2=r2, mape=mape, y_pred=y_pred, extra=extra)


# Plotting
# -----------------------------
def save_metric_comparison_plot(results, output_dir):
    names = [r.name for r in results]
    maes = [r.mae for r in results]
    rmses = [r.rmse for r in results]
    r2s = [r.r2 for r in results]

    x = np.arange(len(names))
    width = 0.18

    plt.figure(figsize=(12, 7))
    plt.bar(x - 1 * width, maes, width, label="MAE")
    plt.bar(x - 0.0 * width, rmses, width, label="RMSE")
    plt.bar(x + 1 * width, r2s, width, label="R²")

    plt.xticks(x, names, rotation=12)
    plt.ylabel("Metric value")
    plt.title("Model Performance Comparison")
    plt.legend()

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_metric_comparison.png"), dpi=200)
    plt.close()


def save_actual_vs_predicted_plot(results, y_test, output_dir):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    y_min = min(float(np.min(y_test)), *(float(np.min(r.y_pred)) for r in results))
    y_max = max(float(np.max(y_test)), *(float(np.max(r.y_pred)) for r in results))

    for ax, res in zip(axes, results):
        ax.scatter(y_test, res.y_pred, alpha=0.75)
        ax.plot([y_min, y_max], [y_min, y_max], linestyle="--")
        ax.set_title(f"{res.name}\nR²={res.r2:.3f}, RMSE={res.rmse:.3f}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Actual vs Predicted by Model", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=200)
    plt.close()


def save_residual_plot(results, y_test, output_dir):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        residuals = y_test - res.y_pred
        ax.scatter(res.y_pred, residuals, alpha=0.75)
        ax.axhline(0.0, linestyle="--")
        ax.set_title(res.name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Residual Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residual_analysis.png"), dpi=200)
    plt.close()


def save_random_forest_importance_plot(results, output_dir):
    rf_results = next((r for r in results if r.name == "Random Forest Regressor"), None)
    if rf_results is None or "feature_importances" not in rf_results.extra:
        return

    feat_imp = rf_results.extra["feature_importances"]
    features = [f for f, _ in feat_imp]
    scores = [s for _, s in feat_imp]

    plt.figure(figsize=(10, 6))
    plt.barh(features[::-1], scores[::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importances")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rf_feature_importances.png"), dpi=200)
    plt.close()


def save_summary_csv(results, output_dir):
    summary_df = pd.DataFrame(
        {
            "Model": [r.name for r in results],
            "MAE": [r.mae for r in results],
            "RMSE": [r.rmse for r in results],
            "R2": [r.r2 for r in results],
            "MAPE_percent": [r.mape for r in results],
        }
    ).sort_values(by="R2", ascending=False)
    out_path = os.path.join(output_dir, "model_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\n[INFO] Saved summary metrics to: {out_path}")
    return summary_df


# Main
# -----------------------------
def main():
    # Ingest args to run the demonstration
    parser = argparse.ArgumentParser(description="Compare SVR, Random Forest, and FFNN for university admission prediction.")
    parser.add_argument("--data", type=str, required=True, help="Path to input dataset")
    parser.add_argument("--output_dir", type=str, default="admission_outputs", help="FOlder for plots and reports")
    parser.add_argument("--test_size", type=float, default=0.20, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=37, help="Random seed")
    parser.add_argument("--ffnn_epochs", type=int, default=250, help="Max FFNN epochs")
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Make output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and split input data
    X, y = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    print(f"[INFO] Train size: {len(X_train)} | Test size: {len(X_test)}")

    # Collect features for models
    features = X.columns.tolist()

    # Build pipelines for SVR and Random Forest models w/ our features
    svr_model = build_svr_pipeline(features)
    rf_model = build_rf_pipeline(features)

    results = []

    # Evaluate the SVR and Random Forest models
    results.append(evaluate_sklearn_model("SVR", svr_model, X_train, X_test, y_train, y_test))
    results.append(evaluate_sklearn_model("Random Forest Regressor", rf_model, X_train, X_test, y_train, y_test))

    # Train our FFNN model and evaluate
    results.append(train_ffnn(X_train, X_test, y_train, y_test, args.output_dir, epochs=args.ffnn_epochs))

    print("\n---------- Final Ranked Results ----------")
    # Rank our results by the r^2 metric
    ranked = sorted(results, key=lambda r: r.r2, reverse=True)
    for idx, r in enumerate(ranked, start=1):
        print(
            f"{idx}. {r.name:<24s} | R²={r.r2:.5f} | RMSE={r.rmse:.5f} | MAE={r.mae:.5f} | MAPE={r.mape:.3f}%"
        )

    # Complete relevant plots for further analysis
    save_metric_comparison_plot(results, args.output_dir)
    save_actual_vs_predicted_plot(results, y_test.values, args.output_dir)
    save_residual_plot(results, y_test.values, args.output_dir)
    save_random_forest_importance_plot(results, args.output_dir)
    summary_df = save_summary_csv(results, args.output_dir)

    print("\n---------- Summary Table ----------")
    print(summary_df.to_string(index=False))
    print(f"\n[INFO] All plots and reports saved in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
