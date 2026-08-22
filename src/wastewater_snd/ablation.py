from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from wastewater_snd.model_v4 import make_group_kfold
from wastewater_snd.schema import (
    AERATION_COL,
    A_LIVE_COL,
    A_MAX_COL,
    COD_IN_COL,
    DATE_COL,
    H_LIVE_COL,
    H_MAX_COL,
    N_LIVE_COL,
    N_MAX_COL,
    REMOVAL_COL,
    SND_COL,
    TEMP_COL,
    TN_IN_COL,
)
from wastewater_snd.sources import model_row_audit, read_model_csv


def _feature_table(data: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    return pd.DataFrame(
        {
            "H_max": data[H_MAX_COL],
            "AOB_max": data[A_MAX_COL],
            "NOB_max": data[N_MAX_COL],
            "temperature": data[TEMP_COL],
            "aeration": data[AERATION_COL],
            "TN_in": data[TN_IN_COL],
            "COD_in": data[COD_IN_COL],
            "C_N": data[COD_IN_COL] / data[TN_IN_COL],
            "H_live": data[H_LIVE_COL],
            "AOB_live": data[A_LIVE_COL],
            "NOB_live": data[N_LIVE_COL],
            "AOB_per_H": data[A_MAX_COL] / (data[H_MAX_COL] + eps),
            "NOB_per_AOB": data[N_MAX_COL] / (data[A_MAX_COL] + eps),
        }
    )


FEATURE_SETS = {
    "最大OUR": [
        "H_max",
        "AOB_max",
        "NOB_max",
        "temperature",
        "aeration",
        "TN_in",
        "COD_in",
        "C_N",
    ],
    "最大OUR+异养菌实时OUR": [
        "H_max",
        "AOB_max",
        "NOB_max",
        "temperature",
        "aeration",
        "TN_in",
        "COD_in",
        "C_N",
        "H_live",
    ],
    "最大OUR+三类实时OUR": [
        "H_max",
        "AOB_max",
        "NOB_max",
        "temperature",
        "aeration",
        "TN_in",
        "COD_in",
        "C_N",
        "H_live",
        "AOB_live",
        "NOB_live",
    ],
    "三类实时OUR_不含最大OUR": [
        "temperature",
        "aeration",
        "TN_in",
        "COD_in",
        "C_N",
        "H_live",
        "AOB_live",
        "NOB_live",
    ],
    "V4精简机理特征": [
        "temperature",
        "aeration",
        "TN_in",
        "COD_in",
        "C_N",
        "H_live",
        "AOB_per_H",
        "NOB_per_AOB",
    ],
    "V4精简机理特征+三类实时OUR": [
        "temperature",
        "aeration",
        "TN_in",
        "COD_in",
        "C_N",
        "H_live",
        "AOB_per_H",
        "NOB_per_AOB",
        "AOB_live",
        "NOB_live",
    ],
}


def _make_ridge() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))])


def _make_rbf(target: str) -> Pipeline:
    gamma = 0.01 if target == REMOVAL_COL else 0.003
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", SVR(kernel="rbf", C=1.0, gamma=gamma, epsilon=0.03)),
        ]
    )


def _oof_metrics(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    splitter,
    factory: Callable[[], Pipeline],
) -> dict[str, float]:
    prediction = np.full(len(y), np.nan)
    for train_index, valid_index in splitter.split(x, y, groups=groups):
        model = factory()
        model.fit(x.iloc[train_index], y[train_index])
        prediction[valid_index] = model.predict(x.iloc[valid_index])
    return {
        "r2": float(r2_score(y, prediction)),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
    }


def evaluate_realtime_our_ablation(
    data_path: Path,
    output_path: Path,
    repeats: int = 10,
) -> pd.DataFrame:
    """Compare realtime-OUR feature sets under identical record-level models."""
    raw = read_model_csv(data_path)
    missing_live = [column for column in [A_LIVE_COL, N_LIVE_COL] if column not in raw]
    if missing_live:
        raise ValueError("实时 OUR 消融缺少字段：" + "、".join(missing_live))
    normalized, audit = model_row_audit(raw)
    included = audit["训练状态"].eq("纳入").to_numpy()
    data = normalized.loc[included].reset_index(drop=True)
    if data[[A_LIVE_COL, N_LIVE_COL]].isna().any().any():
        raise ValueError("纳入训练的记录仍有 AOB/NOB 实时 OUR 缺失。")
    if data[DATE_COL].nunique() < 5:
        raise ValueError("有效日期少于 5 个，无法执行按日期 5 折消融。")

    features = _feature_table(data)
    groups = data[DATE_COL].astype(str)
    rows: list[dict[str, object]] = []
    for target in [REMOVAL_COL, SND_COL]:
        y = data[target].to_numpy(dtype=float)
        factories: dict[str, Callable[[], Pipeline]] = {
            "Ridge10": _make_ridge,
            "RBF-SVR": lambda target=target: _make_rbf(target),
        }
        for feature_name, columns in FEATURE_SETS.items():
            x = features[columns]
            for model_name, factory in factories.items():
                fixed = _oof_metrics(x, y, groups, GroupKFold(n_splits=5), factory)
                logo = _oof_metrics(x, y, groups, LeaveOneGroupOut(), factory)
                repeated_r2: list[float] = []
                repeated_mae: list[float] = []
                for repeat in range(repeats):
                    repeated = _oof_metrics(
                        x,
                        y,
                        groups,
                        make_group_kfold(
                            n_splits=5,
                            shuffle=True,
                            random_state=42 + repeat,
                        ),
                        factory,
                    )
                    repeated_r2.append(repeated["r2"])
                    repeated_mae.append(repeated["mae"])
                repeated_mean = float(np.mean(repeated_r2))
                repeated_std = float(np.std(repeated_r2, ddof=1))
                rows.append(
                    {
                        "目标": target,
                        "模型": model_name,
                        "特征组": feature_name,
                        "特征数": len(columns),
                        "固定5折_R2": fixed["r2"],
                        "固定5折_MAE": fixed["mae"],
                        "固定5折_RMSE": fixed["rmse"],
                        "重复分组_R2均值": repeated_mean,
                        "重复分组_R2标准差": repeated_std,
                        "重复分组_MAE均值": float(np.mean(repeated_mae)),
                        "留一日期_R2": logo["r2"],
                        "留一日期_MAE": logo["mae"],
                        "稳定得分": repeated_mean - 0.25 * repeated_std,
                    }
                )

    result = pd.DataFrame.from_records(rows).sort_values(
        ["目标", "模型", "稳定得分"], ascending=[True, True, False]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result

