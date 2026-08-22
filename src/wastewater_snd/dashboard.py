"""Streamlit portfolio dashboard for the public synthetic-data demo."""

from __future__ import annotations

import hashlib
import io
import os
import re
from collections.abc import Mapping

import pandas as pd

from . import model_v4
from .demo_runtime import (
    DemoRuntime,
    aeration_response_curve,
    build_demo_runtime,
    make_condition,
    predict_condition,
)
from .experience import (
    build_decision_report_payload,
    build_scenario_presets,
    decision_report_json,
    decision_report_markdown,
)
from .sources import validate_model_frame

REPOSITORY_URL = "https://github.com/xie176320/our-snd-aeration-control"
PUBLIC_MODE = "public"
LOCAL_MODE = "local"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def resolve_app_mode(environment: Mapping[str, str] | None = None) -> str:
    """Resolve the runtime mode, failing closed to the public demo."""

    env = os.environ if environment is None else environment
    requested = env.get("SND_APP_MODE", PUBLIC_MODE).strip().lower()
    return LOCAL_MODE if requested == LOCAL_MODE else PUBLIC_MODE


def local_import_enabled(environment: Mapping[str, str] | None = None) -> bool:
    """Allow CSV import only when both local-only switches are explicit."""

    env = os.environ if environment is None else environment
    import_switch = env.get("SND_LOCAL_IMPORT", "").strip().lower()
    return resolve_app_mode(env) == LOCAL_MODE and import_switch in _TRUE_VALUES


def _read_uploaded_csv(payload: bytes) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "gbk", "utf-8"):
        try:
            frame = pd.read_csv(io.BytesIO(payload), encoding=encoding)
            frame.columns = frame.columns.astype(str).str.strip()
            return frame, encoding
        except (UnicodeDecodeError, LookupError, ValueError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("CSV 编码读取失败：" + "；".join(errors))


def _recommended_aeration(recommendation: str) -> float | None:
    match = re.search(r"本次建议曝气量：([0-9.]+) L/min", recommendation)
    return float(match.group(1)) if match else None


def _condition_inputs(
    st, runtime: DemoRuntime, model_source: str
) -> tuple[dict[str, float], float, float, str]:
    median = runtime.data[model_v4.RAW_MODEL_INPUTS].median(numeric_only=True)
    presets = build_scenario_presets(runtime.data)
    preset_by_key = {preset.key: preset for preset in presets}
    st.sidebar.markdown("### 典型工况一键体验")
    st.sidebar.caption("点击后自动填入一组合成或本地统计工况，仍可继续手动修改。")
    button_columns = st.sidebar.columns(2)
    for index, preset in enumerate(presets):
        if button_columns[index % 2].button(
            preset.label,
            key=f"scenario_{preset.key}",
            use_container_width=True,
        ):
            for column, value in preset.values.items():
                st.session_state[f"condition_{column}"] = float(value)
            st.session_state["tn_standard"] = float(preset.tn_standard)
            st.session_state["minimum_safe_aeration"] = float(
                preset.minimum_safe_aeration
            )
            st.session_state["_active_scenario"] = preset.key

    active_key = str(st.session_state.get("_active_scenario", "stable"))
    active = preset_by_key.get(active_key, presets[0])
    st.sidebar.info(f"当前起点：{active.label}。{active.description}")
    st.sidebar.markdown("### 新工况输入")
    st.sidebar.caption(f"默认值来自{model_source}的中位数，可直接修改。")

    labels = {
        model_v4.H_MAX_COL: "异养菌最大 OUR",
        model_v4.A_MAX_COL: "AOB 最大 OUR",
        model_v4.N_MAX_COL: "NOB 最大 OUR",
        model_v4.H_LIVE_COL: "异养菌实时 OUR",
        model_v4.TEMP_COL: "温度（℃）",
        model_v4.AERATION_COL: "当前曝气量（L/min）",
        model_v4.TN_IN_COL: "进水 TN（mg/L）",
        model_v4.COD_IN_COL: "进水 COD（mg/L）",
    }
    values: dict[str, float] = {}
    for column in model_v4.RAW_MODEL_INPUTS:
        lower = 0.1 if column in (model_v4.AERATION_COL, model_v4.TN_IN_COL) else 0.0
        widget_key = f"condition_{column}"
        widget_options = {
            "min_value": lower,
            "step": 0.1,
            "format": "%.4f",
            "key": widget_key,
        }
        if widget_key not in st.session_state:
            widget_options["value"] = float(median[column])
        values[column] = float(
            st.sidebar.number_input(
                labels[column],
                **widget_options,
            )
        )
    tn_options = {
        "min_value": 0.1,
        "step": 0.5,
        "key": "tn_standard",
    }
    if "tn_standard" not in st.session_state:
        tn_options["value"] = 15.0
    tn_standard = float(
        st.sidebar.number_input(
            "出水 TN 限值（mg/L）",
            **tn_options,
        )
    )
    historical_low = float(runtime.data[model_v4.AERATION_COL].min())
    historical_high = float(runtime.data[model_v4.AERATION_COL].max())
    stored_floor = st.session_state.get("minimum_safe_aeration")
    if stored_floor is not None and not 0.1 <= float(stored_floor) <= historical_high:
        st.session_state["minimum_safe_aeration"] = historical_low
    floor_options = {
        "min_value": 0.1,
        "max_value": historical_high,
        "step": 0.1,
        "help": (
            "现场应依据池型、搅拌与污泥沉降风险确认。该值仅约束生化池混合曝气，"
            "MBR 膜擦洗风量需单独核算。"
        ),
        "key": "minimum_safe_aeration",
    }
    if "minimum_safe_aeration" not in st.session_state:
        floor_options["value"] = historical_low
    minimum_safe_aeration = float(
        st.sidebar.number_input(
            "生化池混合安全下限（L/min）",
            **floor_options,
        )
    )
    return values, tn_standard, minimum_safe_aeration, active.label


def _render_control_tab(
    st,
    runtime: DemoRuntime,
    values: dict[str, float],
    tn_standard: float,
    minimum_safe_aeration: float,
    scenario_label: str,
    *,
    is_synthetic: bool,
):
    st.subheader("新工况预测与分级曝气建议")
    st.info(f"当前体验工况：{scenario_label}。侧栏参数可继续修改。")
    if is_synthetic:
        st.caption(
            "模型只在合成演示数据的历史范围内给出 A/B/C 级建议；"
            "它用于展示工程方法，不可直接连接现场设备。"
        )
    else:
        st.caption(
            "模型由本地 CSV 在当前进程中即时建立，只在导入数据的历史范围内给出建议；"
            "结果仍需专业人员复核，不可直接连接现场设备。"
        )
    try:
        condition = make_condition(values)
        prediction, recommendation = predict_condition(
            runtime,
            condition,
            tn_standard=tn_standard,
            minimum_safe_aeration=minimum_safe_aeration,
        )
        curve = aeration_response_curve(
            runtime,
            condition,
            minimum_safe_aeration=minimum_safe_aeration,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    recommended = _recommended_aeration(recommendation)
    metrics = st.columns(4)
    metrics[0].metric("TN 去除率", f"{prediction[model_v4.REMOVAL_COL]:.1%}")
    metrics[1].metric("SND 率", f"{prediction[model_v4.SND_COL]:.1%}")
    metrics[2].metric(
        "推导出水 TN",
        f"{prediction[model_v4.EFFLUENT_PROXY_COL]:.2f} mg/L",
    )
    metrics[3].metric(
        "本次建议曝气量",
        f"{recommended:.2f} L/min" if recommended is not None else "—",
    )

    left, right = st.columns([1.1, 1.4])
    with left:
        st.markdown("#### 安全分级结果")
        st.code(recommendation, language=None)
    with right:
        st.markdown("#### 曝气—出水 TN 响应")
        tn_curve = curve.set_index(model_v4.AERATION_COL)[
            ["预测出水TN(mg/L)", "90%保守上界(mg/L)"]
        ].copy()
        tn_curve["TN限值(mg/L)"] = tn_standard
        st.line_chart(tn_curve)
        st.caption(
            "曲线从配置的生化池混合安全下限开始，且不超出训练数据曝气范围；"
            "保守上界包含跨日期 90% 误差裕量。"
        )

    st.markdown("#### TN 去除率与 SND 响应")
    ratio_curve = curve.set_index(model_v4.AERATION_COL)[["预测TN去除率", "预测SND率"]]
    st.line_chart(ratio_curve)

    report_payload = build_decision_report_payload(
        condition=condition,
        prediction=prediction,
        recommendation=recommendation,
        recommended_aeration=recommended,
        scenario_label=scenario_label,
        tn_standard=tn_standard,
        minimum_safe_aeration=minimum_safe_aeration,
        is_synthetic=is_synthetic,
    )
    markdown_report = decision_report_markdown(report_payload)
    json_report = decision_report_json(report_payload)
    st.markdown("#### 导出本次决策报告")
    st.caption(
        "报告只包含本次输入、预测和安全建议，不包含训练数据；"
        "公开演示导出的仍是合成数据。"
    )
    report_columns = st.columns(2)
    report_columns[0].download_button(
        "下载 Markdown 报告",
        data=markdown_report,
        file_name="our_snd_decision_report.md",
        mime="text/markdown; charset=utf-8",
        use_container_width=True,
    )
    report_columns[1].download_button(
        "下载 JSON 报告",
        data=json_report,
        file_name="our_snd_decision_report.json",
        mime="application/json; charset=utf-8",
        use_container_width=True,
    )


def _render_quality_tab(
    st,
    frame: pd.DataFrame,
    *,
    encoding: str,
    source_label: str,
    allow_local_import: bool,
):
    st.subheader("训练数据质量闸门")
    if allow_local_import:
        st.caption("文件导入只在本地启动模式开放；CSV 由本机应用进程在内存中解析，不会写入仓库。")
    else:
        st.caption("公开在线演示已关闭文件上传；下方仅检查应用运行时生成的合成样例。")

    summary, issues = validate_model_frame(frame)
    row1 = st.columns(4)
    row1[0].metric("记录数", summary.get("rows", 0))
    row1[1].metric("有效记录", summary.get("valid_rows", 0))
    row1[2].metric("日期组", summary.get("date_groups", 0))
    row1[3].metric("训练就绪", "是" if summary.get("train_ready") else "否")
    st.caption(f"数据来源：{source_label} · 识别编码：{encoding}")

    if issues:
        st.warning(f"检测到 {len(issues)} 项质量问题。")
        st.dataframe(
            pd.DataFrame([issue.as_dict() for issue in issues]),
            width="stretch",
            hide_index=True,
        )
    else:
        st.success("字段、样本量、日期组数和数值范围均通过校验。")
    with st.expander("查看数据预览"):
        st.dataframe(frame.head(20), width="stretch", hide_index=True)


def _render_evidence_tab(st, runtime: DemoRuntime, *, is_synthetic: bool):
    st.subheader("可信度与适用边界")
    if is_synthetic:
        st.warning(
            "下表来自公开合成数据，只用于 CI 和界面验收；不能作为论文实验结果或现场性能承诺。"
        )
    else:
        st.info(
            "下表由本地导入数据即时计算，数据与结果仅存在于当前本机进程；"
            "它仍不是未经现场验证的性能承诺。"
        )
    metrics = runtime.metrics.copy()
    st.dataframe(
        metrics.style.format({"日期分组R²": "{:.3f}", "MAE": "{:.3f}", "RMSE": "{:.3f}"}),
        width="stretch",
        hide_index=True,
    )

    a, b, c = st.columns(3)
    a.metric(
        "合成记录" if is_synthetic else "本地有效记录",
        runtime.info["clean_rows"],
    )
    b.metric("独立日期组", runtime.info["date_groups"])
    c.metric("剂量效应门控", "通过" if runtime.gate["allowed"] else "未通过")

    st.markdown(
        """
        - **防止泄漏：** 同一日期的全部记录始终进入同一个训练折或验证折。
        - **不伪造第三目标：** 出水 TN 是由进水 TN 与预测去除率推导的代理值。
        - **不盲目外推：** 逐目标计算标准化最近邻距离，并检查原始变量范围。
        - **不直接闭环：** 建议按 A/B/C 证据等级执行，B 级要求稳定 1 个 HRT 后复测。
        """
    )


def _render_engineering_tab(st):
    st.subheader("从研究问题到工程落地")
    st.markdown(
        """
        1. **数据层**：Excel/CSV 解析、字段字典、来源行追踪、完整案例排除审计。
        2. **建模层**：按日期整批交叉验证、OUR 特征消融、TN/SND 双目标建模。
        3. **决策层**：支持域检查、90% 误差裕量、曝气剂量效应门控。
        4. **交付层**：CLI、网页演示、Docker、Codespaces、GitHub Actions 和自动测试。
        """
    )
    st.code(
        """# macOS / Linux：一条命令启动本地导入
bash scripts/start_local.sh

# Windows：双击 scripts\\start_local.cmd

# 命令行端到端验收
bash scripts/run_demo.sh

# 容器运行
docker compose up --build""",
        language="bash",
    )
    st.link_button("查看 GitHub 源码", REPOSITORY_URL)


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="OUR-SND 数据建模与曝气决策支持",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .hero {padding: 1.4rem 1.6rem; border-radius: 18px;
               background: linear-gradient(120deg, #082f49, #0369a1 55%, #0f766e);
               color: white; margin-bottom: 1rem;}
        .hero h1 {margin: 0; font-size: 2.15rem;}
        .hero p {margin: .55rem 0 0; color: #dbeafe;}
        [data-testid="stMetric"] {border: 1px solid #dbeafe; padding: .8rem;
                                  border-radius: 12px; background: #f8fafc;}
        </style>
        <div class="hero">
          <h1>OUR-SND 数据建模与曝气决策支持平台</h1>
          <p>把污水处理实验数据转化为可审计预测、风险边界和分级曝气建议。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    @st.cache_resource(show_spinner="正在建立公开合成数据演示模型……")
    def _load_synthetic_runtime() -> DemoRuntime:
        return build_demo_runtime()

    def _load_local_runtime(payload: bytes) -> DemoRuntime:
        frame, _ = _read_uploaded_csv(payload)
        summary, _ = validate_model_frame(frame)
        if not summary.get("train_ready"):
            raise ValueError("本地 CSV 未通过训练数据质量闸门。")
        return build_demo_runtime(frame)

    try:
        runtime = _load_synthetic_runtime()
    except Exception as exc:
        st.error(f"演示模型启动失败：{exc}")
        st.stop()

    def _clear_local_runtime() -> None:
        st.session_state.pop("_local_model_digest", None)
        st.session_state.pop("_local_model_runtime", None)

    allow_local_import = local_import_enabled()
    model_source = "公开合成数据"
    quality_frame = runtime.data.drop(columns=[model_v4.EFFLUENT_PROXY_COL], errors="ignore")
    quality_encoding = "运行时合成生成器"
    quality_source = "公开合成样例"
    is_synthetic = True

    if allow_local_import:
        st.info(
            "当前为本地导入模式：可从左侧选择标准 CSV。文件只交给本机应用进程，"
            "不会上传到公开演示或写入仓库。"
        )
        st.sidebar.markdown("### 本地 CSV 导入")
        st.sidebar.caption("至少 40 条有效记录、5 个日期/批次；文件上限 20 MB。")
        upload = st.sidebar.file_uploader(
            "选择标准 CSV",
            type=["csv"],
            help="字段需符合仓库 README 中的 11 项数据契约。",
            key="local_model_csv",
        )
        if upload is None:
            _clear_local_runtime()
            st.sidebar.info("尚未选择文件，当前预测使用公开合成样例。")
        else:
            payload = upload.getvalue()
            try:
                imported_frame, imported_encoding = _read_uploaded_csv(payload)
            except ValueError as exc:
                _clear_local_runtime()
                st.sidebar.error(f"CSV 读取失败：{exc}")
            else:
                quality_frame = imported_frame
                quality_encoding = imported_encoding
                quality_source = "本地 CSV（仅当前进程）"
                summary, _ = validate_model_frame(imported_frame)
                if summary.get("train_ready"):
                    try:
                        digest = hashlib.sha256(payload).hexdigest()
                        if st.session_state.get("_local_model_digest") != digest:
                            with st.spinner("正在建立本地数据模型……"):
                                st.session_state["_local_model_runtime"] = _load_local_runtime(
                                    payload
                                )
                            st.session_state["_local_model_digest"] = digest
                        runtime = st.session_state["_local_model_runtime"]
                    except Exception as exc:
                        _clear_local_runtime()
                        st.sidebar.error(f"本地模型建立失败：{exc}")
                    else:
                        model_source = "本地导入数据"
                        is_synthetic = False
                        st.sidebar.success(
                            f"已在本机加载 {runtime.info['clean_rows']} 条有效记录，"
                            f"共 {runtime.info['date_groups']} 个日期/批次。"
                        )
                else:
                    _clear_local_runtime()
                    st.sidebar.error(
                        "CSV 未通过质量闸门，当前预测仍使用合成样例；请在“数据质检”中查看问题。"
                    )
    else:
        st.info("当前为公开在线演示：文件上传已关闭，全部结果来自运行时生成的合成数据。")
        st.sidebar.success("公开合成演示 · 线上文件上传已关闭")

    values, tn_standard, minimum_safe_aeration, scenario_label = _condition_inputs(
        st,
        runtime,
        model_source,
    )
    control, quality, evidence, engineering = st.tabs(
        ["智能调控", "数据质检", "模型可信度", "工程架构"]
    )
    with control:
        _render_control_tab(
            st,
            runtime,
            values,
            tn_standard,
            minimum_safe_aeration,
            scenario_label,
            is_synthetic=is_synthetic,
        )
    with quality:
        _render_quality_tab(
            st,
            quality_frame,
            encoding=quality_encoding,
            source_label=quality_source,
            allow_local_import=allow_local_import,
        )
    with evidence:
        _render_evidence_tab(st, runtime, is_synthetic=is_synthetic)
    with engineering:
        _render_engineering_tab(st)

    st.divider()
    footer_source = "公开合成演示" if is_synthetic else "本地数据模型"
    st.caption(f"科研演示软件 · MIT License · {footer_source}不可用于未经验证的现场自动控制")


if __name__ == "__main__":
    main()
