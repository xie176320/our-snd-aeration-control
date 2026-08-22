from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from wastewater_snd import model_v4
from wastewater_snd.ablation import evaluate_realtime_our_ablation
from wastewater_snd.calibrated import (
    CALIBRATION_POINTS,
    SLOPE_RIDGE_ALPHA,
    evaluate_rolling_calibration,
    evaluate_three_point_calibration,
    fixed_calibrated_base_specs,
    predict_calibrated_condition,
)
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
    OPTIONAL_MODEL_COLUMNS,
    REMOVAL_COL,
    REQUIRED_MODEL_COLUMNS,
    SND_COL,
    TEMP_COL,
    TN_IN_COL,
    TN_OUT_COL,
)
from wastewater_snd.sources import (
    audit_sources,
    model_row_audit,
    read_model_csv,
    validate_model_frame,
)
from wastewater_snd.synthetic import demo_frame


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _print_issues(issues) -> None:
    for issue in issues:
        row = "" if issue.row is None else f" 第{issue.row}行"
        print(
            f"[{issue.severity}] {issue.code}: {issue.source}/{issue.sheet}{row} — "
            f"{issue.message}"
        )


def command_schema(_: argparse.Namespace) -> int:
    print("V4 训练 CSV 必需字段：")
    for column in REQUIRED_MODEL_COLUMNS:
        print(f"- {column}")
    print("\n建议同时保留的字段：")
    for column in OPTIONAL_MODEL_COLUMNS:
        print(f"- {column}")
    return 0


def command_audit(args: argparse.Namespace) -> int:
    report = audit_sources(
        legacy_water_path=_path(args.legacy_water),
        legacy_our_path=_path(args.legacy_our),
        current_water_path=_path(args.current_water),
        mbr_our_path=_path(args.mbr_our),
        output_dir=_path(args.output_dir),
    )
    validation = report["model_input_validation"]
    print(f"审计状态：{report['status']}")
    print(
        f"标准草稿：{report['record_counts']['model_draft_rows']} 条；"
        f"完整有效：{validation['valid_rows']} 条；"
        f"有效日期：{validation['date_groups']} 个。"
    )
    print("问题计数：" + json.dumps(report["issue_counts_by_code"], ensure_ascii=False))
    print("输出文件：")
    for label, path in report["outputs"].items():
        print(f"- {label}: {path}")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    path = _path(args.data)
    if not path.exists():
        raise FileNotFoundError(f"未找到 CSV：{path}")
    frame = read_model_csv(path)
    summary, issues = validate_model_frame(frame)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    _print_issues(issues)
    return 0 if summary["train_ready"] else 2


def command_demo_data(args: argparse.Namespace) -> int:
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = demo_frame(seed=args.seed)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    print(
        f"已生成仅用于验证程序的合成数据：{output}"
        f"（{len(frame)} 条，12 个日期）"
    )
    print("该文件不是实验结果，不得用于论文结论或现场控制。")
    return 0


def _train(
    data_path: Path,
    output_dir: Path,
    interactive: bool,
    minimum_safe_aeration: float | None = None,
) -> int:
    frame = read_model_csv(data_path)
    validation, issues = validate_model_frame(frame)
    if not validation["train_ready"]:
        print("训练前校验未通过。", file=sys.stderr)
        _print_issues(issues)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    _, row_audit = model_row_audit(frame)
    excluded_positions = row_audit.index[row_audit["训练状态"].eq("排除")]
    excluded = frame.iloc[excluded_positions].copy()
    excluded.insert(0, "排除原因", row_audit.loc[excluded_positions, "排除原因"].to_numpy())
    excluded.insert(0, "CSV行号", row_audit.loc[excluded_positions, "CSV行号"].to_numpy())
    validation_path = output_dir / "input_validation.json"
    excluded_path = output_dir / "input_excluded_rows.csv"
    validation_payload = {
        "summary": validation,
        "issues": [issue.as_dict() for issue in issues],
    }
    validation_path.write_text(
        json.dumps(validation_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    excluded.to_csv(excluded_path, index=False, encoding="utf-8-sig")
    if issues:
        print(
            f"输入共 {validation['rows']} 条，训练纳入 {validation['valid_rows']} 条，"
            f"完整案例排除 {validation['excluded_rows']} 条。"
        )
        _print_issues(issues)

    print("正在执行固定 5 折、重复分组 5 折和留一日期验证……")
    data, info = model_v4.load_and_clean_data(data_path)
    diagnostics = model_v4.duplicate_input_diagnostic(data)
    effect_table = model_v4.aeration_effect_diagnostic(data)
    gate = model_v4.aeration_optimization_gate(effect_table)
    evaluation, selected, oof_store = model_v4.evaluate_candidates(data)
    trained, error_q90, support = model_v4.train_selected_models(data, selected, oof_store)
    paths = model_v4.save_outputs(
        data_path=data_path,
        output_dir=output_dir,
        data=data,
        evaluation=evaluation,
        selected=selected,
        oof_store=oof_store,
        trained=trained,
        error_q90=error_q90,
        support=support,
        effect_table=effect_table,
        gate=gate,
    )
    paths["input_validation"] = validation_path
    paths["excluded_rows"] = excluded_path
    model_v4.print_model_report(
        info=info,
        diagnostics=diagnostics,
        evaluation=evaluation,
        selected=selected,
        oof_store=oof_store,
        data=data,
        gate=gate,
        paths=paths,
    )
    if interactive:
        model_v4.interactive_loop(
            data=data,
            selected=selected,
            trained=trained,
            error_q90=error_q90,
            support=support,
            gate=gate,
            trial_step_abs=model_v4.TRIAL_STEP_ABS_DEFAULT,
            trial_step_fraction=model_v4.TRIAL_STEP_FRACTION_DEFAULT,
            safety_margin=model_v4.TN_SAFETY_MARGIN_DEFAULT,
            minimum_safe_aeration=minimum_safe_aeration,
        )
    return 0


def command_train(args: argparse.Namespace) -> int:
    data_path = _path(args.data)
    output_dir = _path(args.output_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"未找到训练 CSV：{data_path}")
    return _train(
        data_path,
        output_dir,
        args.interactive,
        minimum_safe_aeration=args.minimum_safe_aeration,
    )


def command_ablate_our(args: argparse.Namespace) -> int:
    data_path = _path(args.data)
    output_path = _path(args.output)
    if not data_path.exists():
        raise FileNotFoundError(f"未找到训练 CSV：{data_path}")
    result = evaluate_realtime_our_ablation(
        data_path=data_path,
        output_path=output_path,
        repeats=args.repeats,
    )
    print(f"OUR 特征消融已保存：{output_path}")
    for target in [REMOVAL_COL, SND_COL]:
        subset = result[result["目标"] == target]
        best = subset.sort_values("稳定得分", ascending=False).iloc[0]
        print(
            f"{target} 最稳组合：{best['模型']} / {best['特征组']}；"
            f"固定5折 R²={best['固定5折_R2']:.3f}，"
            f"重复分组 R²={best['重复分组_R2均值']:.3f}±"
            f"{best['重复分组_R2标准差']:.3f}。"
        )
    return 0


def command_train_calibrated(args: argparse.Namespace) -> int:
    data_path = _path(args.data)
    output_dir = _path(args.output_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"未找到训练 CSV：{data_path}")
    status = _train(data_path, output_dir, interactive=False)
    if status:
        return status

    data, _ = model_v4.load_and_clean_data(data_path)
    base_bundle_path = output_dir / "model_bundle.joblib"
    bundle = joblib.load(base_bundle_path)
    fixed_specs = fixed_calibrated_base_specs()
    logo_summary, logo_predictions, logo_q90 = evaluate_three_point_calibration(
        data,
        fixed_specs,
        calibration_points=args.calibration_points,
        slope_ridge_alpha=args.slope_ridge_alpha,
    )
    rolling_summary, rolling_predictions, rolling_q90 = evaluate_rolling_calibration(
        data,
        fixed_specs,
        calibration_points=args.calibration_points,
        slope_ridge_alpha=args.slope_ridge_alpha,
    )
    summary = pd.concat([logo_summary, rolling_summary], ignore_index=True)
    calibrated_q90 = {
        target: max(logo_q90[target], rolling_q90[target])
        for target in [REMOVAL_COL, SND_COL]
    }
    summary_path = output_dir / "calibrated_evaluation.csv"
    predictions_path = output_dir / "calibrated_oof_predictions.csv"
    rolling_predictions_path = output_dir / "calibrated_rolling_predictions.csv"
    calibrated_bundle_path = output_dir / "calibrated_model_bundle.joblib"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logo_predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    rolling_predictions.to_csv(
        rolling_predictions_path, index=False, encoding="utf-8-sig"
    )

    gate_passed = bool((summary["校准后_R2"] > args.required_r2).all())
    bundle["selected_specs"] = fixed_specs
    bundle["selected_model_names"] = {
        target: spec.name for target, spec in fixed_specs.items()
    }
    bundle["feature_sets"] = {
        target: spec.feature_set for target, spec in fixed_specs.items()
    }
    bundle["models"] = {
        target: model_v4.fit_candidate(spec, data)
        for target, spec in fixed_specs.items()
    }
    bundle["version"] = "4C-1.0"
    bundle["same_day_calibration"] = {
        "method": "整日留出后用当日低/中/高三点拟合正则化残差截距与斜率",
        "calibration_points": int(args.calibration_points),
        "slope_ridge_alpha": float(args.slope_ridge_alpha),
        "required_r2_exclusive": float(args.required_r2),
        "r2_gate_passed": gate_passed,
        "scope": "同日校准范围内插值，不适用于新日期冷启动",
    }
    bundle["calibrated_evaluation"] = summary
    bundle["calibrated_abs_error_q90"] = calibrated_q90
    joblib.dump(bundle, calibrated_bundle_path)

    print("\n当日三点校准验证：")
    for _, row in summary.iterrows():
        print(
            f"- {row['目标']} / {row['验证方式']}："
            f"基础模型 R²={row['基础模型_R2']:.3f}；"
            f"校准后盲测 R²={row['校准后_R2']:.3f}，"
            f"MAE={row['校准后_MAE']:.3f}；"
            f"{int(row['校准日期数'])} 个日期、"
            f"{int(row['盲测记录数'])} 条盲测记录。"
        )
    print(f"- 评估表：{summary_path}")
    print(f"- 留一日期逐行预测：{predictions_path}")
    print(f"- 滚动时间逐行预测：{rolling_predictions_path}")
    print(f"- 校准模型包：{calibrated_bundle_path}")
    if not gate_passed:
        print(
            f"R² 门控未通过：至少一个目标没有严格大于 {args.required_r2:.3f}；"
            "模型包仅供诊断，不得标记为达标。",
            file=sys.stderr,
        )
        return 3
    print(
        "R² 门控通过：两个目标在限定的当日三点校准插值场景下"
        "均严格大于 "
        f"{args.required_r2:.3f}。"
    )
    return 0


def _read_condition(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("工况 JSON 顶层必须是一个对象。")
        return pd.DataFrame([payload])
    frame = read_model_csv(path)
    if len(frame) != 1:
        raise ValueError("工况 CSV 必须恰好包含一行。")
    return frame


def _prepare_calibrated_input(
    frame: pd.DataFrame,
    training_data: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    prepared = frame.copy()
    prepared.columns = prepared.columns.astype(str).str.strip()
    if DATE_COL not in prepared:
        prepared[DATE_COL] = label

    required_without_live = [
        column for column in model_v4.RAW_MODEL_INPUTS if column != H_LIVE_COL
    ]
    missing = [column for column in required_without_live if column not in prepared]
    if missing:
        raise ValueError(f"{label}缺少字段：" + "、".join(missing))
    for column in required_without_live:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    if prepared[required_without_live].isna().any().any():
        raise ValueError(f"{label}含无法解析的空值或非数字。")

    if H_LIVE_COL not in prepared:
        prepared[H_LIVE_COL] = np.nan
    prepared[H_LIVE_COL] = pd.to_numeric(prepared[H_LIVE_COL], errors="coerce")
    for index in prepared.index[prepared[H_LIVE_COL].isna()]:
        prepared.loc[index, H_LIVE_COL] = model_v4.estimate_live_h(
            training_data,
            prepared.loc[[index]],
        )

    nonnegative = [H_MAX_COL, A_MAX_COL, N_MAX_COL, H_LIVE_COL, COD_IN_COL]
    if (prepared[nonnegative] < 0).any().any():
        raise ValueError(f"{label}的 OUR 与进水 COD 不能为负数。")
    if (prepared[TN_IN_COL] <= 0).any():
        raise ValueError(f"{label}的进水 TN 必须大于 0。")
    if (prepared[AERATION_COL] <= 0).any():
        raise ValueError(f"{label}的曝气量必须大于 0。")
    return prepared


def command_predict_calibrated(args: argparse.Namespace) -> int:
    model_path = _path(args.model)
    calibration_path = _path(args.calibration)
    condition_path = _path(args.condition)
    bundle = joblib.load(model_path)
    required_keys = {
        "models",
        "training_data",
        "same_day_calibration",
        "calibrated_abs_error_q90",
    }
    missing_bundle = sorted(required_keys - set(bundle))
    if missing_bundle:
        raise ValueError(
            "不是可用的当日校准模型包，缺少：" + "、".join(missing_bundle)
        )
    if not bool(bundle["same_day_calibration"].get("r2_gate_passed")):
        raise ValueError("该模型包没有通过设定的 R² 门控。")

    calibration = read_model_csv(calibration_path)
    condition = _read_condition(condition_path)
    training_data = bundle["training_data"].copy()
    calibration = _prepare_calibrated_input(
        calibration, training_data, "当日校准文件"
    )
    condition = _prepare_calibrated_input(condition, training_data, "待预测工况")
    calibration_dates = calibration[DATE_COL].astype(str).unique()
    if len(calibration_dates) != 1:
        raise ValueError("当日校准文件必须只包含一个日期/批次。")
    condition_date = str(condition.iloc[0][DATE_COL])
    if condition_date != str(calibration_dates[0]):
        raise ValueError("待预测工况与校准记录必须属于同一日期/批次。")

    prediction = predict_calibrated_condition(bundle, calibration, condition)
    print(json.dumps(prediction, ensure_ascii=False, indent=2))
    print(
        "\n说明：R²>0.8 仅对应同日已有低/中/高三个实测点，"
        "且在其曝气范围内的"
        "插值场景；不能解释为完全未见日期的冷启动 R²。"
    )
    return 0


def command_predict(args: argparse.Namespace) -> int:
    model_path = _path(args.model)
    condition_path = _path(args.condition)
    bundle = joblib.load(model_path)
    required_bundle_keys = {
        "selected_specs",
        "models",
        "training_data",
        "strict_oof_abs_error_q90",
        "support",
        "aeration_optimization_gate",
    }
    missing_bundle = sorted(required_bundle_keys - set(bundle))
    if missing_bundle:
        raise ValueError(
            "模型包版本过旧，缺少字段："
            + "、".join(missing_bundle)
            + "；请用当前版本重新训练。"
        )
    condition = _read_condition(condition_path)
    condition.columns = condition.columns.astype(str).str.strip()
    condition[DATE_COL] = condition.get(DATE_COL, pd.Series(["new-condition"]))
    training_data = bundle["training_data"].copy()

    if H_LIVE_COL not in condition or pd.isna(
        pd.to_numeric(condition[H_LIVE_COL], errors="coerce")
    ).any():
        match_columns = [H_MAX_COL, A_MAX_COL, N_MAX_COL, TEMP_COL, TN_IN_COL, COD_IN_COL]
        missing_for_estimate = [column for column in match_columns if column not in condition]
        if missing_for_estimate:
            raise ValueError(
                "缺少异养菌实时 OUR，且无法估算；还缺："
                + "、".join(missing_for_estimate)
            )
        for column in match_columns:
            condition[column] = pd.to_numeric(condition[column], errors="coerce")
        condition[H_LIVE_COL] = model_v4.estimate_live_h(training_data, condition)
        print(
            "异养菌实时 OUR 未提供，按历史近邻估算为 "
            f"{condition.iloc[0][H_LIVE_COL]:.4f}。"
        )

    missing_inputs = [column for column in model_v4.RAW_MODEL_INPUTS if column not in condition]
    if missing_inputs:
        raise ValueError("工况缺少字段：" + "、".join(missing_inputs))
    for column in model_v4.RAW_MODEL_INPUTS:
        condition[column] = pd.to_numeric(condition[column], errors="coerce")
    if condition[model_v4.RAW_MODEL_INPUTS].isna().any().any():
        raise ValueError("工况字段含无法解析的空值或非数字。")
    nonnegative_columns = [H_MAX_COL, A_MAX_COL, N_MAX_COL, H_LIVE_COL, COD_IN_COL]
    if any(float(condition.iloc[0][column]) < 0 for column in nonnegative_columns):
        raise ValueError("OUR 与进水 COD 不能为负数。")
    if float(condition.iloc[0][TN_IN_COL]) <= 0:
        raise ValueError("进水 TN 必须大于 0。")
    if float(condition.iloc[0][AERATION_COL]) <= 0:
        raise ValueError("当前曝气量必须大于 0。")

    selected = bundle["selected_specs"]
    trained = bundle["models"]
    prediction = model_v4.predict_one_condition(condition, selected, trained)
    print("预测结果：")
    print(json.dumps(prediction, ensure_ascii=False, indent=2))
    print("\n分级曝气建议：")
    print(
        model_v4.recommend_aeration(
            row=condition,
            tn_standard=args.tn_standard,
            data=training_data,
            selected=selected,
            trained=trained,
            error_q90=bundle["strict_oof_abs_error_q90"],
            support=bundle["support"],
            gate=bundle["aeration_optimization_gate"],
            trial_step_abs=args.trial_step_abs,
            trial_step_fraction=args.trial_step_fraction,
            safety_margin=args.tn_safety_margin,
            minimum_safe_aeration=args.minimum_safe_aeration,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snd-control",
        description="污水处理 SND/OUR 数据审计、V4 模型训练与曝气调控",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema_parser = subparsers.add_parser("schema", help="列出 V4 标准训练字段")
    schema_parser.set_defaults(handler=command_schema)

    audit_parser = subparsers.add_parser(
        "audit-sources", help="解析四类授权工作簿并生成标准草稿和质量问题清单"
    )
    audit_parser.add_argument("--legacy-water", required=True, help="历史水质工作簿")
    audit_parser.add_argument("--legacy-our", required=True, help="历史 OUR 工作簿")
    audit_parser.add_argument("--current-water", required=True, help="当前水质工作簿")
    audit_parser.add_argument("--mbr-our", required=True, help="当前反应器 OUR 工作簿")
    audit_parser.add_argument("--output-dir", default="outputs/source-audit")
    audit_parser.set_defaults(handler=command_audit)

    validate_parser = subparsers.add_parser("validate", help="校验标准训练 CSV")
    validate_parser.add_argument("--data", required=True)
    validate_parser.set_defaults(handler=command_validate)

    demo_parser = subparsers.add_parser(
        "demo-data", help="生成仅用于验收程序的合成数据"
    )
    demo_parser.add_argument("--output", default="outputs/demo/demo_model_input.csv")
    demo_parser.add_argument("--seed", type=int, default=42)
    demo_parser.set_defaults(handler=command_demo_data)

    train_parser = subparsers.add_parser(
        "train", help="执行 V4 日期分组验证并训练模型"
    )
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--output-dir", default="outputs/model-v4")
    train_parser.add_argument(
        "--interactive",
        action="store_true",
        help="训练完成后进入逐项输入新工况的交互界面",
    )
    train_parser.add_argument(
        "--minimum-safe-aeration",
        type=float,
        default=None,
        help="交互建议采用的生化池混合曝气硬下限",
    )
    train_parser.set_defaults(handler=command_train)

    ablation_parser = subparsers.add_parser(
        "ablate-our", help="在相同日期分组模型下比较最大 OUR 与实时 OUR 特征"
    )
    ablation_parser.add_argument("--data", required=True)
    ablation_parser.add_argument(
        "--output", default="outputs/model-v4/realtime_our_ablation.csv"
    )
    ablation_parser.add_argument("--repeats", type=int, default=10)
    ablation_parser.set_defaults(handler=command_ablate_our)

    calibrated_train_parser = subparsers.add_parser(
        "train-calibrated",
        help="训练基础模型并验证当日低/中/高三点校准模型",
    )
    calibrated_train_parser.add_argument("--data", required=True)
    calibrated_train_parser.add_argument(
        "--output-dir", default="outputs/model-v4-calibrated"
    )
    calibrated_train_parser.add_argument(
        "--calibration-points", type=int, default=CALIBRATION_POINTS
    )
    calibrated_train_parser.add_argument(
        "--slope-ridge-alpha", type=float, default=SLOPE_RIDGE_ALPHA
    )
    calibrated_train_parser.add_argument(
        "--required-r2",
        type=float,
        default=0.8,
        help="两个目标必须严格超过的校准盲测 R²；不通过时退出码为 3",
    )
    calibrated_train_parser.set_defaults(handler=command_train_calibrated)

    predict_parser = subparsers.add_parser("predict", help="加载模型包预测一条新工况")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--condition", required=True, help="一条工况的 JSON 或 CSV")
    predict_parser.add_argument("--tn-standard", type=float, default=15.0)
    predict_parser.add_argument("--trial-step-abs", type=float, default=0.5)
    predict_parser.add_argument("--trial-step-fraction", type=float, default=0.05)
    predict_parser.add_argument("--tn-safety-margin", type=float, default=1.0)
    predict_parser.add_argument(
        "--minimum-safe-aeration",
        type=float,
        default=None,
        help="生化池混合曝气硬下限；省略时使用训练数据历史下限",
    )
    predict_parser.set_defaults(handler=command_predict)

    calibrated_predict_parser = subparsers.add_parser(
        "predict-calibrated",
        help="使用同日三个实测校准点预测校准范围内的一条工况",
    )
    calibrated_predict_parser.add_argument("--model", required=True)
    calibrated_predict_parser.add_argument(
        "--calibration", required=True, help="含三个实测目标点的 CSV"
    )
    calibrated_predict_parser.add_argument(
        "--condition", required=True, help="一条同日工况的 JSON 或 CSV"
    )
    calibrated_predict_parser.set_defaults(handler=command_predict_calibrated)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        exit_code = int(args.handler(args))
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        exit_code = 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        exit_code = 1
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
