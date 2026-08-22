"""Guided demo scenarios and portable decision-report builders."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from . import __version__, model_v4


@dataclass(frozen=True)
class ScenarioPreset:
    """A one-click operating scenario derived from the active runtime data."""

    key: str
    label: str
    description: str
    values: dict[str, float]
    tn_standard: float
    minimum_safe_aeration: float


def build_scenario_presets(data: pd.DataFrame) -> tuple[ScenarioPreset, ...]:
    """Build four bounded scenarios without embedding private plant values."""

    if data.empty:
        raise ValueError("无法从空数据建立典型工况。")
    median = data[model_v4.RAW_MODEL_INPUTS].median(numeric_only=True)
    if median.isna().any():
        raise ValueError("典型工况所需字段包含空值。")

    base = {column: float(median[column]) for column in model_v4.RAW_MODEL_INPUTS}
    historical_low = float(data[model_v4.AERATION_COL].min())
    historical_high = float(data[model_v4.AERATION_COL].max())
    span = max(historical_high - historical_low, 0.1)
    mixing_floor = min(historical_high, historical_low + 0.25 * span)
    near_floor = min(historical_high, mixing_floor + 0.025 * span)
    below_floor = max(0.1, mixing_floor - 0.125 * span)
    if below_floor >= mixing_floor:
        below_floor = max(0.1, mixing_floor * 0.9)

    stable = base.copy()
    risk = base.copy()
    risk[model_v4.TN_IN_COL] = float(data[model_v4.TN_IN_COL].max())
    risk[model_v4.COD_IN_COL] = float(data[model_v4.COD_IN_COL].quantile(0.25))
    risk[model_v4.AERATION_COL] = historical_low
    near = base.copy()
    near[model_v4.AERATION_COL] = near_floor
    below = base.copy()
    below[model_v4.AERATION_COL] = below_floor

    return (
        ScenarioPreset(
            key="stable",
            label="稳定运行",
            description="采用当前数据中位工况，展示常规预测与分级建议。",
            values=stable,
            tn_standard=15.0,
            minimum_safe_aeration=historical_low,
        ),
        ScenarioPreset(
            key="high_tn_risk",
            label="高 TN 风险",
            description="在历史范围内组合高进水 TN、较低 C/N 和低曝气量。",
            values=risk,
            tn_standard=15.0,
            minimum_safe_aeration=historical_low,
        ),
        ScenarioPreset(
            key="near_mixing_floor",
            label="接近混合下限",
            description="当前曝气量略高于配置下限，用于观察降曝气保护。",
            values=near,
            tn_standard=15.0,
            minimum_safe_aeration=mixing_floor,
        ),
        ScenarioPreset(
            key="below_mixing_floor",
            label="低于安全下限",
            description="当前曝气量低于配置下限，触发工程硬约束纠偏。",
            values=below,
            tn_standard=15.0,
            minimum_safe_aeration=mixing_floor,
        ),
    )


def build_decision_report_payload(
    *,
    condition: pd.DataFrame,
    prediction: Mapping[str, float],
    recommendation: str,
    recommended_aeration: float | None,
    scenario_label: str,
    tn_standard: float,
    minimum_safe_aeration: float,
    is_synthetic: bool,
    generated_at_utc: str | None = None,
) -> dict[str, object]:
    """Return a JSON-safe report containing only the current input and result."""

    if len(condition) != 1:
        raise ValueError("决策报告只能包含一条工况。")
    generated_at = generated_at_utc or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    inputs = {
        column: float(condition.iloc[0][column])
        for column in model_v4.RAW_MODEL_INPUTS
    }
    payload: dict[str, object] = {
        "report_schema": "our-snd-decision-report/1.0",
        "project_version": __version__,
        "generated_at_utc": generated_at,
        "data_mode": "public_synthetic" if is_synthetic else "local_session",
        "data_notice": (
            "公开演示：仅使用运行时合成数据。"
            if is_synthetic
            else "本地模式：报告由当前本机会话数据生成。"
        ),
        "scenario": scenario_label,
        "inputs": inputs,
        "constraints": {
            "effluent_tn_limit_mg_l": float(tn_standard),
            "biological_mixing_floor_l_min": float(minimum_safe_aeration),
            "mbr_scour_air_included": False,
        },
        "predictions": {
            "tn_removal_rate": float(prediction[model_v4.REMOVAL_COL]),
            "snd_rate": float(prediction[model_v4.SND_COL]),
            "derived_effluent_tn_mg_l": float(
                prediction[model_v4.EFFLUENT_PROXY_COL]
            ),
        },
        "recommended_aeration_l_min": (
            None if recommended_aeration is None else float(recommended_aeration)
        ),
        "recommendation": recommendation,
        "training_rows_included": False,
        "disclaimer": (
            "科研与教学用途的决策支持原型；不能替代独立实测、专业人员审核、"
            "安全联锁或现场自动控制。"
        ),
    }
    return payload


def decision_report_markdown(payload: Mapping[str, object]) -> str:
    """Render a human-readable Markdown decision report."""

    inputs = payload["inputs"]
    constraints = payload["constraints"]
    predictions = payload["predictions"]
    if not isinstance(inputs, Mapping):
        raise ValueError("报告 inputs 格式无效。")
    if not isinstance(constraints, Mapping) or not isinstance(predictions, Mapping):
        raise ValueError("报告结果格式无效。")

    lines = [
        "# OUR-SND 曝气决策报告",
        "",
        f"- 项目版本：v{payload['project_version']}",
        f"- 生成时间（UTC）：{payload['generated_at_utc']}",
        f"- 典型工况：{payload['scenario']}",
        f"- 数据说明：{payload['data_notice']}",
        "",
        "## 输入工况",
        "",
        "| 字段 | 数值 |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {float(value):.4f} |" for name, value in inputs.items())
    lines.extend(
        [
            "",
            "## 约束与预测",
            "",
            "| 指标 | 数值 |",
            "| --- | ---: |",
            (
                "| 出水 TN 限值 | "
                f"{float(constraints['effluent_tn_limit_mg_l']):.2f} mg/L |"
            ),
            (
                "| 生化池混合安全下限 | "
                f"{float(constraints['biological_mixing_floor_l_min']):.2f} L/min |"
            ),
            f"| TN 去除率 | {float(predictions['tn_removal_rate']):.2%} |",
            f"| SND 率 | {float(predictions['snd_rate']):.2%} |",
            (
                "| 推导出水 TN | "
                f"{float(predictions['derived_effluent_tn_mg_l']):.2f} mg/L |"
            ),
            (
                "| 本次建议曝气量 | — |"
                if payload["recommended_aeration_l_min"] is None
                else "| 本次建议曝气量 | "
                f"{float(payload['recommended_aeration_l_min']):.2f} L/min |"
            ),
            "",
            "## 分级建议",
            "",
            "```text",
            str(payload["recommendation"]),
            "```",
            "",
            "## 安全边界",
            "",
            "- 本报告不包含训练数据，只包含本次工况输入与计算结果。",
            "- 生化池混合下限不包含 MBR 膜擦洗风量，两者必须独立核算。",
            f"- {payload['disclaimer']}",
            "",
        ]
    )
    return "\n".join(lines)


def decision_report_json(payload: Mapping[str, object]) -> str:
    """Render the structured decision report as readable UTF-8 JSON."""

    return json.dumps(payload, ensure_ascii=False, indent=2)
