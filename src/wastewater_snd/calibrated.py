from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from wastewater_snd import model_v4
from wastewater_snd.schema import (
    AERATION_COL,
    DATE_COL,
    REMOVAL_COL,
    SND_COL,
    TN_IN_COL,
)

CALIBRATION_POINTS = 3
SLOPE_RIDGE_ALPHA = 1.0


@dataclass(frozen=True)
class ResidualCalibrator:
    center: float
    scale: float
    intercept: float
    slope: float
    slope_ridge_alpha: float


def fixed_calibrated_base_specs() -> dict[str, model_v4.ModelSpec]:
    """Pre-declared V4 base models; outer-date targets do not select the spec."""
    return {
        REMOVAL_COL: model_v4.ModelSpec(
            name="双SVR集成_日期等权",
            target=REMOVAL_COL,
            kind="tn_ensemble",
            feature_set="loads8",
        ),
        SND_COL: model_v4.ModelSpec(
            name="TN-SND联合集成",
            target=SND_COL,
            kind="snd_joint",
            feature_set="loads8",
        ),
    }


def select_calibration_indices(
    group: pd.DataFrame,
    calibration_points: int = CALIBRATION_POINTS,
) -> list[int]:
    """Select one row at low/middle/high distinct aeration levels."""
    if calibration_points < 2:
        raise ValueError("校准点至少为 2 个。")
    levels = np.sort(
        pd.to_numeric(group[AERATION_COL], errors="coerce").dropna().unique()
    )
    if len(levels) < calibration_points:
        raise ValueError(
            f"至少需要 {calibration_points} 个不同曝气水平，"
            f"当前只有 {len(levels)} 个。"
        )
    positions = np.rint(
        np.linspace(0, len(levels) - 1, calibration_points)
    ).astype(int)
    chosen_levels = levels[positions]
    if len(np.unique(chosen_levels)) != calibration_points:
        raise RuntimeError("无法选择互不相同的曝气校准水平。")
    indices: list[int] = []
    numeric_aeration = pd.to_numeric(group[AERATION_COL], errors="coerce")
    for level in chosen_levels:
        indices.append(int(group.index[numeric_aeration.eq(level)][0]))
    return indices


def fit_residual_calibrator(
    aeration: np.ndarray,
    residual: np.ndarray,
    slope_ridge_alpha: float = SLOPE_RIDGE_ALPHA,
) -> ResidualCalibrator:
    aeration = np.asarray(aeration, dtype=float).reshape(-1)
    residual = np.asarray(residual, dtype=float).reshape(-1)
    if len(aeration) != len(residual) or len(aeration) < 2:
        raise ValueError("曝气量和残差必须等长，且至少包含 2 个点。")
    low = float(np.min(aeration))
    high = float(np.max(aeration))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("校准曝气量必须包含至少两个不同的有限数值。")
    if slope_ridge_alpha < 0:
        raise ValueError("斜率正则化系数不能为负。")
    center = 0.5 * (low + high)
    scale = 0.5 * (high - low)
    x = (aeration - center) / scale
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.diag([0.0, float(slope_ridge_alpha)])
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ residual,
    )
    return ResidualCalibrator(
        center=center,
        scale=scale,
        intercept=float(coefficients[0]),
        slope=float(coefficients[1]),
        slope_ridge_alpha=float(slope_ridge_alpha),
    )


def residual_correction(
    calibrator: ResidualCalibrator,
    aeration: np.ndarray,
) -> np.ndarray:
    values = np.asarray(aeration, dtype=float).reshape(-1)
    standardized = (values - calibrator.center) / calibrator.scale
    return calibrator.intercept + calibrator.slope * standardized


def evaluate_three_point_calibration(
    data: pd.DataFrame,
    selected: dict[str, model_v4.ModelSpec],
    calibration_points: int = CALIBRATION_POINTS,
    slope_ridge_alpha: float = SLOPE_RIDGE_ALPHA,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Strictly leave a date out, then calibrate from its measured dose points.

    The global model never sees the held-out date during fitting. Within that date,
    low/middle/high measured points are used only to estimate a regularized residual
    intercept and slope. Metrics are calculated on the remaining rows.
    """
    working = data.reset_index(drop=True).copy()
    result = working[[DATE_COL, AERATION_COL, TN_IN_COL, REMOVAL_COL, SND_COL]].copy()
    result["记录用途"] = "日期不满足校准条件"

    base_predictions: dict[str, np.ndarray] = {}
    for target in [REMOVAL_COL, SND_COL]:
        spec = selected[target]
        logo = model_v4.oof_predict(spec, working, LeaveOneGroupOut())
        base_predictions[target] = logo.predictions
        result[f"{target}_基础留一日期预测"] = logo.predictions
        result[f"{target}_校准后预测"] = np.nan
        result[f"{target}_校准修正量"] = np.nan

    eligible_dates = 0
    calibration_rows = 0
    validation_rows = 0
    for _, group in working.groupby(DATE_COL, sort=False):
        if group[AERATION_COL].nunique() < calibration_points:
            continue
        if len(group) <= calibration_points:
            continue
        calibration_index = select_calibration_indices(group, calibration_points)
        validation_index = group.index.difference(calibration_index)
        result.loc[calibration_index, "记录用途"] = "当日校准"
        result.loc[validation_index, "记录用途"] = "盲测验证"
        eligible_dates += 1
        calibration_rows += len(calibration_index)
        validation_rows += len(validation_index)

        calibration_aeration = working.loc[
            calibration_index, AERATION_COL
        ].to_numpy(dtype=float)
        validation_aeration = working.loc[
            validation_index, AERATION_COL
        ].to_numpy(dtype=float)
        for target in [REMOVAL_COL, SND_COL]:
            residual = (
                working.loc[calibration_index, target].to_numpy(dtype=float)
                - base_predictions[target][calibration_index]
            )
            calibrator = fit_residual_calibrator(
                calibration_aeration,
                residual,
                slope_ridge_alpha=slope_ridge_alpha,
            )
            correction = residual_correction(calibrator, validation_aeration)
            calibrated_prediction = np.clip(
                base_predictions[target][validation_index] + correction,
                0.0,
                1.0,
            )
            result.loc[
                validation_index, f"{target}_校准修正量"
            ] = correction
            result.loc[
                validation_index, f"{target}_校准后预测"
            ] = calibrated_prediction

    if eligible_dates == 0 or validation_rows == 0:
        raise ValueError("没有日期同时满足三个不同曝气水平和剩余盲测记录。")

    validation_mask = result["记录用途"].eq("盲测验证")
    summary_rows: list[dict[str, object]] = []
    calibrated_q90: dict[str, float] = {}
    for target in [REMOVAL_COL, SND_COL]:
        observed = result.loc[validation_mask, target].to_numpy(dtype=float)
        base = result.loc[
            validation_mask, f"{target}_基础留一日期预测"
        ].to_numpy(dtype=float)
        calibrated = result.loc[
            validation_mask, f"{target}_校准后预测"
        ].to_numpy(dtype=float)
        base_metrics = model_v4.regression_metrics(observed, base)
        calibrated_metrics = model_v4.regression_metrics(observed, calibrated)
        q90 = model_v4.conformal_abs_error_quantile(observed, calibrated)
        calibrated_q90[target] = q90
        summary_rows.append(
            {
                "目标": target,
                "验证方式": "整日留出后用当日低/中/高三点校准",
                "校准点数": calibration_points,
                "校准日期数": eligible_dates,
                "校准记录数": calibration_rows,
                "盲测记录数": validation_rows,
                "基础模型_R2": base_metrics["r2"],
                "基础模型_MAE": base_metrics["mae"],
                "校准后_R2": calibrated_metrics["r2"],
                "校准后_MAE": calibrated_metrics["mae"],
                "校准后_RMSE": calibrated_metrics["rmse"],
                "校准后绝对误差90%分位": q90,
                "斜率正则化系数": slope_ridge_alpha,
            }
        )
    return pd.DataFrame.from_records(summary_rows), result, calibrated_q90


def _date_order_value(value: object) -> int:
    text = str(value).strip()
    parts = text.split(".")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        month, day = (int(part) for part in parts)
        return int(pd.Timestamp(year=2000, month=month, day=day).toordinal())
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"无法按时间排序日期/批次：{value}")
    return int(pd.Timestamp(parsed).toordinal())


def evaluate_rolling_calibration(
    data: pd.DataFrame,
    selected: dict[str, model_v4.ModelSpec],
    calibration_points: int = CALIBRATION_POINTS,
    slope_ridge_alpha: float = SLOPE_RIDGE_ALPHA,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Train only on earlier dates, then calibrate and test the current date."""
    working = data.reset_index(drop=True).copy()
    working["__date_order"] = working[DATE_COL].map(_date_order_value)
    result = working[[DATE_COL, AERATION_COL, TN_IN_COL, REMOVAL_COL, SND_COL]].copy()
    result["记录用途"] = "日期不满足滚动校准条件"
    for target in [REMOVAL_COL, SND_COL]:
        result[f"{target}_基础滚动预测"] = np.nan
        result[f"{target}_校准后预测"] = np.nan
        result[f"{target}_校准修正量"] = np.nan

    eligible_dates = 0
    calibration_rows = 0
    validation_rows = 0
    training_date_counts: list[int] = []
    ordered_dates = (
        working[[DATE_COL, "__date_order"]]
        .drop_duplicates()
        .sort_values("__date_order")
    )
    for _, date_row in ordered_dates.iterrows():
        date = date_row[DATE_COL]
        date_order = int(date_row["__date_order"])
        group = working[working[DATE_COL].eq(date)]
        if group[AERATION_COL].nunique() < calibration_points:
            continue
        if len(group) <= calibration_points:
            continue
        training = working[working["__date_order"] < date_order].drop(
            columns=["__date_order"]
        )
        training_date_count = int(training[DATE_COL].nunique())
        if training_date_count < 5:
            continue

        calibration_index = select_calibration_indices(group, calibration_points)
        validation_index = group.index.difference(calibration_index)
        result.loc[calibration_index, "记录用途"] = "当日校准"
        result.loc[validation_index, "记录用途"] = "向前盲测验证"
        eligible_dates += 1
        calibration_rows += len(calibration_index)
        validation_rows += len(validation_index)
        training_date_counts.append(training_date_count)

        calibration_aeration = working.loc[
            calibration_index, AERATION_COL
        ].to_numpy(dtype=float)
        validation_aeration = working.loc[
            validation_index, AERATION_COL
        ].to_numpy(dtype=float)
        group_without_order = group.drop(columns=["__date_order"])
        for target in [REMOVAL_COL, SND_COL]:
            fitted = model_v4.fit_candidate(selected[target], training)
            base_calibration = model_v4.predict_fitted(
                fitted, group_without_order.loc[calibration_index]
            )
            base_validation = model_v4.predict_fitted(
                fitted, group_without_order.loc[validation_index]
            )
            residual = (
                working.loc[calibration_index, target].to_numpy(dtype=float)
                - base_calibration
            )
            calibrator = fit_residual_calibrator(
                calibration_aeration,
                residual,
                slope_ridge_alpha=slope_ridge_alpha,
            )
            correction = residual_correction(calibrator, validation_aeration)
            calibrated_prediction = np.clip(
                base_validation + correction,
                0.0,
                1.0,
            )
            result.loc[
                validation_index, f"{target}_基础滚动预测"
            ] = base_validation
            result.loc[
                validation_index, f"{target}_校准修正量"
            ] = correction
            result.loc[
                validation_index, f"{target}_校准后预测"
            ] = calibrated_prediction

    if eligible_dates == 0 or validation_rows == 0:
        raise ValueError("没有日期满足滚动训练和三点校准验证条件。")

    validation_mask = result["记录用途"].eq("向前盲测验证")
    summary_rows: list[dict[str, object]] = []
    calibrated_q90: dict[str, float] = {}
    for target in [REMOVAL_COL, SND_COL]:
        observed = result.loc[validation_mask, target].to_numpy(dtype=float)
        base = result.loc[
            validation_mask, f"{target}_基础滚动预测"
        ].to_numpy(dtype=float)
        calibrated = result.loc[
            validation_mask, f"{target}_校准后预测"
        ].to_numpy(dtype=float)
        base_metrics = model_v4.regression_metrics(observed, base)
        calibrated_metrics = model_v4.regression_metrics(observed, calibrated)
        q90 = model_v4.conformal_abs_error_quantile(observed, calibrated)
        calibrated_q90[target] = q90
        summary_rows.append(
            {
                "目标": target,
                "验证方式": "只用更早日期训练后做当日低/中/高三点校准",
                "校准点数": calibration_points,
                "校准日期数": eligible_dates,
                "校准记录数": calibration_rows,
                "盲测记录数": validation_rows,
                "最少历史日期数": min(training_date_counts),
                "基础模型_R2": base_metrics["r2"],
                "基础模型_MAE": base_metrics["mae"],
                "校准后_R2": calibrated_metrics["r2"],
                "校准后_MAE": calibrated_metrics["mae"],
                "校准后_RMSE": calibrated_metrics["rmse"],
                "校准后绝对误差90%分位": q90,
                "斜率正则化系数": slope_ridge_alpha,
            }
        )
    return pd.DataFrame.from_records(summary_rows), result, calibrated_q90


def fit_calibrators_for_new_date(
    bundle: dict[str, object],
    calibration: pd.DataFrame,
    calibration_points: int = CALIBRATION_POINTS,
    slope_ridge_alpha: float = SLOPE_RIDGE_ALPHA,
) -> dict[str, ResidualCalibrator]:
    if len(calibration) < calibration_points:
        raise ValueError(f"当日校准至少需要 {calibration_points} 条实测记录。")
    if calibration[AERATION_COL].nunique() < calibration_points:
        raise ValueError(f"当日校准至少需要 {calibration_points} 个不同曝气水平。")
    chosen = select_calibration_indices(calibration, calibration_points)
    selected_calibration = calibration.loc[chosen].copy()
    calibrators: dict[str, ResidualCalibrator] = {}
    for target in [REMOVAL_COL, SND_COL]:
        if target not in selected_calibration:
            raise ValueError(f"当日校准文件缺少实测目标：{target}")
        actual = pd.to_numeric(selected_calibration[target], errors="coerce")
        if actual.isna().any() or ((actual < 0) | (actual > 1)).any():
            raise ValueError(f"当日校准的 {target} 必须是 0–1 之间的实测值。")
        base = model_v4.predict_fitted(
            bundle["models"][target], selected_calibration
        )
        calibrators[target] = fit_residual_calibrator(
            selected_calibration[AERATION_COL].to_numpy(dtype=float),
            actual.to_numpy(dtype=float) - base,
            slope_ridge_alpha=slope_ridge_alpha,
        )
    return calibrators


def predict_calibrated_condition(
    bundle: dict[str, object],
    calibration: pd.DataFrame,
    condition: pd.DataFrame,
) -> dict[str, object]:
    if len(condition) != 1:
        raise ValueError("待预测工况必须恰好包含一行。")
    configuration = bundle.get("same_day_calibration", {})
    calibration_points = int(
        configuration.get("calibration_points", CALIBRATION_POINTS)
    )
    slope_ridge_alpha = float(
        configuration.get("slope_ridge_alpha", SLOPE_RIDGE_ALPHA)
    )
    calibrators = fit_calibrators_for_new_date(
        bundle,
        calibration,
        calibration_points=calibration_points,
        slope_ridge_alpha=slope_ridge_alpha,
    )
    low = float(calibration[AERATION_COL].min())
    high = float(calibration[AERATION_COL].max())
    aeration = float(condition.iloc[0][AERATION_COL])
    if aeration < low or aeration > high:
        raise ValueError(
            f"待预测曝气量 {aeration:.3f} L/min 超出当日校准范围 "
            f"[{low:.3f}, {high:.3f}]，不能引用 R²>0.8 的插值验证结果。"
        )

    prediction: dict[str, object] = {
        "校准曝气范围(L/min)": [low, high],
        "当前曝气量(L/min)": aeration,
    }
    q90_store = bundle.get("calibrated_abs_error_q90", {})
    for target in [REMOVAL_COL, SND_COL]:
        base = float(model_v4.predict_fitted(bundle["models"][target], condition)[0])
        correction = float(
            residual_correction(calibrators[target], np.array([aeration]))[0]
        )
        calibrated = float(np.clip(base + correction, 0.0, 1.0))
        q90 = float(q90_store.get(target, np.nan))
        prediction[target] = {
            "基础预测": base,
            "当日校准修正": correction,
            "校准后预测": calibrated,
            "90%绝对误差分位": q90,
            "保守范围": (
                [max(0.0, calibrated - q90), min(1.0, calibrated + q90)]
                if np.isfinite(q90)
                else None
            ),
        }

    tn_in = float(condition.iloc[0][TN_IN_COL])
    removal = float(prediction[REMOVAL_COL]["校准后预测"])
    removal_q90 = float(prediction[REMOVAL_COL]["90%绝对误差分位"])
    prediction["出水TN_推导值(mg/L)"] = tn_in * (1.0 - removal)
    prediction["出水TN_保守上界(mg/L)"] = (
        tn_in * (1.0 - max(0.0, removal - removal_q90))
        if np.isfinite(removal_q90)
        else None
    )
    prediction["适用范围"] = (
        "仅适用于已有同日低/中/高三个实测校准点，且待预测曝气量位于"
        "校准范围内的插值预测；不适用于新日期冷启动。"
    )
    return prediction
