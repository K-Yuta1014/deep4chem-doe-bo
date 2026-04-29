#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3. Regression.ipynb の GPR 部分をバッチ実行用にしたスクリプト（Random Forest は含まない）。

配置: プロジェクト直下の src/run_regression_gpr.py
データ・記述子 CSV は notebook/outputs および data/ を参照する。

実行例（プロジェクトルートで）::

    cd /path/to/260411_GPR
    conda activate chem
    python src/run_regression_gpr.py 2>&1 | tee gpr_run.log

夜間実行（tmux）::

    tmux new -s gpr
    cd /path/to/260411_GPR && python src/run_regression_gpr.py 2>&1 | tee gpr_run.log

図は notebook/outputs/regression_gpr_yyplot.png に保存（表示用の show は行わない）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    Matern,
    RBF,
    WhiteKernel,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

# src/ の親 = プロジェクトルート、記述子・図は notebook/ 配下
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebook"
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH = OUTPUT_DIR / "regression_gpr_yyplot.png"


def yyplot(train_df, test_df, save_path: Path | None = None) -> None:
    """実測 vs 予測プロット（notebook 版と同じ指標・スタイル）。"""
    rmse_tr = np.sqrt(mean_squared_error(train_df["true"], train_df["pred"]))
    r2_tr = r2_score(train_df["true"], train_df["pred"])
    mae_tr = mean_absolute_error(train_df["true"], train_df["pred"])

    rmse_te = np.sqrt(mean_squared_error(test_df["true"], test_df["pred"]))
    r2_te = r2_score(test_df["true"], test_df["pred"])
    mae_te = mean_absolute_error(test_df["true"], test_df["pred"])

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111)

    ax.scatter(train_df["true"], train_df["pred"], c="red", label="Train", alpha=0.6)
    ax.scatter(test_df["true"], test_df["pred"], c="blue", label="Val", alpha=0.6)

    all_true = pd.concat([train_df["true"], test_df["true"]])
    min_val = all_true.min()
    max_val = all_true.max()
    ax.plot([min_val, max_val], [min_val, max_val], "k--")

    ax.set_xlabel("True Value")
    ax.set_ylabel("Predicted Value")

    margin = (max_val - min_val) * 0.05
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)

    ax.text(
        0.05,
        0.95,
        f"Train RMSE = {rmse_tr:.2f}\nTrain MAE  = {mae_tr:.2f}\nTrain R²    = {r2_tr:.2f}",
        transform=ax.transAxes,
        fontsize=11,
        color="red",
        verticalalignment="top",
    )
    ax.text(
        0.05,
        0.80,
        f"Val  RMSE = {rmse_te:.2f}\nVal  MAE  = {mae_te:.2f}\nVal  R²    = {r2_te:.2f}",
        transform=ax.transAxes,
        fontsize=11,
        color="blue",
        verticalalignment="top",
    )

    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"図を保存しました: {save_path}", flush=True)
    plt.close()


def main() -> None:
    # ----- パラメータ（notebook と揃える） -----
    descriptor_type = "rdkit"
    MAX_SAMPLES = 5000
    RANDOM_STATE_SAMPLE = 1234
    test_size = 0.3
    random_state_split = 1234
    fold_number = 10

    # ----- データ読み込み -----
    dataset = pd.read_csv(PROJECT_ROOT / "data" / "dataset_train.csv", index_col=0)
    target = dataset[["Stokes shift"]]

    if descriptor_type == "rdkit":
        des = pd.read_csv(
            NOTEBOOK_DIR / "outputs/descriptors/rdkit/X_train_rdkit_processed.csv",
            index_col=0,
        )
    elif descriptor_type == "mordred_2d":
        des = pd.read_csv(
            NOTEBOOK_DIR
            / "outputs/descriptors/mordred_2d/X_train_mordred_2d_processed.csv",
            index_col=0,
        )
    elif descriptor_type == "mordred_3d":
        des = pd.read_csv(
            NOTEBOOK_DIR
            / "outputs/descriptors/mordred_3d/X_train_mordred_3d_processed.csv",
            index_col=0,
        )
    elif descriptor_type == "fp":
        des = pd.read_csv(
            NOTEBOOK_DIR / "outputs/descriptors/fp/X_train_fp.csv",
            index_col=0,
        )
    else:
        raise ValueError(f"未知の descriptor_type: {descriptor_type}")

    dataset_train = target.join(des, how="inner")
    print(target.shape, des.shape, dataset_train.shape, flush=True)

    if len(dataset_train) > MAX_SAMPLES:
        dataset_train = dataset_train.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE_SAMPLE)
        print(f"dataset_train を {MAX_SAMPLES} 件に間引き: {dataset_train.shape}", flush=True)
    else:
        print(f"件数 ≤ {MAX_SAMPLES} のためそのまま利用: {dataset_train.shape}", flush=True)

    y = dataset_train["Stokes shift"]
    X = dataset_train.drop("Stokes shift", axis=1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state_split
    )

    zero_std_cols = X_train.columns[X_train.std() == 0]
    if len(zero_std_cols) > 0:
        X_train = X_train.drop(zero_std_cols, axis=1)
        X_val = X_val.drop(zero_std_cols, axis=1)

    print("X_train:", X_train.shape, "X_val:", X_val.shape, flush=True)

    X_scaler = StandardScaler()
    autoscaled_X_train = pd.DataFrame(
        X_scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    autoscaled_X_val = pd.DataFrame(
        X_scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )

    y_scaler = StandardScaler()
    autoscaled_y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    autoscaled_y_val = y_scaler.transform(y_val.values.reshape(-1, 1))

    autoscaled_y_train = pd.DataFrame(
        autoscaled_y_train, index=y_train.index, columns=["y"]
    )
    autoscaled_y_val = pd.DataFrame(
        autoscaled_y_val, index=y_val.index, columns=["y"]
    )

    # ----- GPR カーネル選択 -----
    n_features = autoscaled_X_train.shape[1]

    kernels = [
        ConstantKernel() * DotProduct() + WhiteKernel(),
        ConstantKernel() * RBF() + WhiteKernel(),
        ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
        ConstantKernel() * RBF(np.ones(n_features)) + WhiteKernel(),
        ConstantKernel() * RBF(np.ones(n_features))
        + WhiteKernel()
        + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=1.5)
        + WhiteKernel()
        + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=0.5)
        + WhiteKernel()
        + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=2.5)
        + WhiteKernel()
        + ConstantKernel() * DotProduct(),
    ]

    cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True)

    X_tr = autoscaled_X_train.values
    y_tr_scaled = autoscaled_y_train["y"].values

    r2cvs = []
    for index, kernel in enumerate(kernels):
        print(index + 1, "/", len(kernels), flush=True)
        model_cv = GaussianProcessRegressor(alpha=1e-10, kernel=kernel, random_state=0)
        estimated_y_in_cv = np.ndarray.flatten(
            cross_val_predict(model_cv, X_tr, y_tr_scaled, cv=cross_validation)
        )
        estimated_y_in_cv = y_scaler.inverse_transform(
            estimated_y_in_cv.reshape(-1, 1)
        ).ravel()
        r2cvs.append(r2_score(y_train, estimated_y_in_cv))

    optimal_kernel_number = int(np.argmax(r2cvs))
    optimal_kernel = kernels[optimal_kernel_number]
    print("クロスバリデーションで選択されたカーネル関数の番号 :", optimal_kernel_number, flush=True)
    print("クロスバリデーションで選択されたカーネル関数 :", optimal_kernel, flush=True)

    model = GaussianProcessRegressor(alpha=1e-10, kernel=optimal_kernel, random_state=0)
    model.fit(X_tr, y_tr_scaled)

    # ----- 予測と yyplot -----
    pred_train_s, _ = model.predict(autoscaled_X_train.values, return_std=True)
    pred_val_s, _ = model.predict(autoscaled_X_val.values, return_std=True)

    pred_train = y_scaler.inverse_transform(pred_train_s.reshape(-1, 1)).ravel()
    pred_val = y_scaler.inverse_transform(pred_val_s.reshape(-1, 1)).ravel()

    train_df = pd.DataFrame(
        {"true": y_train.values, "pred": pred_train}, index=y_train.index
    )
    test_df = pd.DataFrame({"true": y_val.values, "pred": pred_val}, index=y_val.index)

    yyplot(train_df, test_df, save_path=FIG_PATH)
    print("完了。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("エラーが発生しました。", file=sys.stderr)
        raise
