"""Reusable runtime helpers for the public synthetic-data dashboard.

The production CLI intentionally performs an exhaustive date-grouped model
comparison.  A web demo should start faster and must not silently retrain on
untrusted uploads, so this module fits the two documented baseline models on
the repository's synthetic dataset and exposes structured prediction helpers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import model_v4
from .synthetic import demo_frame

DEMO_MODEL_NAMES = {
    model_v4.REMOVAL_COL: "PLS2_原工程特征_记录级",
    model_v4.SND_COL: "Ridge10_原基础特征_记录级",
}


@dataclass
class DemoRuntime:
    """In-memory models and evidence needed by the interactive demo."""

    data: pd.DataFrame
    info: dict[str, object]
    selected: dict[str, model_v4.ModelSpec]
    trained: dict[str, object]
    error_q90: dict[str, float]
    support: dict[str, dict[str, object]]
    gate: dict[str, object]
    metrics: pd.DataFrame


def _documented_specs() -> dict[str, model_v4.ModelSpec]:
    selected: dict[str, model_v4.ModelSpec] = {}
    for target, expected_name in DEMO_MODEL_NAMES.items():
        matches = [spec for spec in model_v4.candidate_models(target) if spec.name == expected_name]
        if len(matches) != 1:
            raise RuntimeError(f"找不到演示模型：{target} / {expected_name}")
        selected[target] = matches[0]
    return selected


def build_demo_runtime(
    data_source: Path | pd.DataFrame | None = None,
) -> DemoRuntime:
    """Fit demo baselines from generated data, a frame, or an explicit CSV."""

    if data_source is None:
        data, info = model_v4.clean_model_frame(demo_frame(), encoding="synthetic-generator")
    elif isinstance(data_source, pd.DataFrame):
        data, info = model_v4.clean_model_frame(data_source, encoding="in-memory-local-import")
    else:
        data, info = model_v4.load_and_clean_data(data_source)
    selected = _documented_specs()
    oof_store: dict[tuple[str, str, str], model_v4.OOFResult] = {}
    metric_rows: list[dict[str, object]] = []

    for target, spec in selected.items():
        oof = model_v4.oof_predict(
            spec,
            data,
            model_v4.make_group_kfold(
                n_splits=min(model_v4.MAX_FOLDS, data[model_v4.DATE_COL].nunique())
            ),
        )
        oof_store[(target, spec.name, "固定日期5折")] = oof
        metric_rows.append(
            {
                "目标": target,
                "模型": spec.name,
                "日期分组R²": oof.metrics["r2"],
                "MAE": oof.metrics["mae"],
                "RMSE": oof.metrics["rmse"],
            }
        )

    trained, error_q90, support = model_v4.train_selected_models(data, selected, oof_store)
    gate = model_v4.aeration_optimization_gate(model_v4.aeration_effect_diagnostic(data))
    return DemoRuntime(
        data=data,
        info=info,
        selected=selected,
        trained=trained,
        error_q90=error_q90,
        support=support,
        gate=gate,
        metrics=pd.DataFrame.from_records(metric_rows),
    )


def make_condition(values: Mapping[str, object]) -> pd.DataFrame:
    """Normalize and validate one prediction condition."""

    missing = [column for column in model_v4.RAW_MODEL_INPUTS if column not in values]
    if missing:
        raise ValueError("工况缺少字段：" + "、".join(missing))

    record: dict[str, object] = {model_v4.DATE_COL: str(values.get(model_v4.DATE_COL, "web-demo"))}
    for column in model_v4.RAW_MODEL_INPUTS:
        try:
            value = float(values[column])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} 必须是数字。") from exc
        if not np.isfinite(value):
            raise ValueError(f"{column} 必须是有限数字。")
        record[column] = value

    nonnegative = [
        model_v4.H_MAX_COL,
        model_v4.A_MAX_COL,
        model_v4.N_MAX_COL,
        model_v4.H_LIVE_COL,
        model_v4.COD_IN_COL,
    ]
    if any(float(record[column]) < 0 for column in nonnegative):
        raise ValueError("OUR 与进水 COD 不能为负数。")
    if float(record[model_v4.TN_IN_COL]) <= 0:
        raise ValueError("进水 TN 必须大于 0。")
    if float(record[model_v4.AERATION_COL]) <= 0:
        raise ValueError("曝气量必须大于 0。")
    return pd.DataFrame([record])


def predict_condition(
    runtime: DemoRuntime,
    condition: pd.DataFrame,
    tn_standard: float = model_v4.TN_STANDARD_DEFAULT,
    minimum_safe_aeration: float | None = None,
) -> tuple[dict[str, float], str]:
    """Return structured predictions plus the safety-graded recommendation."""

    if tn_standard <= 0:
        raise ValueError("出水 TN 限值必须大于 0。")
    prediction = model_v4.predict_one_condition(condition, runtime.selected, runtime.trained)
    recommendation = model_v4.recommend_aeration(
        row=condition,
        tn_standard=tn_standard,
        data=runtime.data,
        selected=runtime.selected,
        trained=runtime.trained,
        error_q90=runtime.error_q90,
        support=runtime.support,
        gate=runtime.gate,
        minimum_safe_aeration=minimum_safe_aeration,
    )
    return prediction, recommendation


def aeration_response_curve(
    runtime: DemoRuntime,
    condition: pd.DataFrame,
    points: int = 41,
    minimum_safe_aeration: float | None = None,
) -> pd.DataFrame:
    """Evaluate the model only inside the historical aeration range."""

    if points < 2:
        raise ValueError("响应曲线至少需要 2 个点。")
    low, _, high = model_v4.resolve_aeration_safety_floor(
        runtime.data,
        minimum_safe_aeration,
    )
    records: list[dict[str, object]] = []
    for aeration in np.linspace(low, high, points):
        candidate = condition.copy()
        candidate.loc[candidate.index[0], model_v4.AERATION_COL] = float(aeration)
        prediction = model_v4.predict_one_condition(candidate, runtime.selected, runtime.trained)
        removal_lower = max(
            0.0,
            prediction[model_v4.REMOVAL_COL] - runtime.error_q90[model_v4.REMOVAL_COL],
        )
        conservative_tn = float(candidate.iloc[0][model_v4.TN_IN_COL] * (1.0 - removal_lower))
        tn_supported = model_v4.support_distance(
            candidate,
            model_v4.REMOVAL_COL,
            runtime.selected[model_v4.REMOVAL_COL],
            runtime.support,
        )[2]
        snd_supported = model_v4.support_distance(
            candidate,
            model_v4.SND_COL,
            runtime.selected[model_v4.SND_COL],
            runtime.support,
        )[2]
        records.append(
            {
                model_v4.AERATION_COL: float(aeration),
                "预测出水TN(mg/L)": prediction[model_v4.EFFLUENT_PROXY_COL],
                "90%保守上界(mg/L)": conservative_tn,
                "预测TN去除率": prediction[model_v4.REMOVAL_COL],
                "预测SND率": prediction[model_v4.SND_COL],
                "联合支持域": bool(tn_supported and snd_supported),
            }
        )
    return pd.DataFrame.from_records(records)
