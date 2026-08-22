"""Streamlit portfolio dashboard for the public synthetic-data demo."""

from __future__ import annotations

import io
import re
import pandas as pd

from . import model_v4
from .demo_runtime import (
    DemoRuntime,
    aeration_response_curve,
    build_demo_runtime,
    make_condition,
    predict_condition,
)
from .sources import validate_model_frame


REPOSITORY_URL = "https://github.com/xie176320/our-snd-aeration-control"


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


def _condition_inputs(st, runtime: DemoRuntime) -> tuple[dict[str, float], float]:
    median = runtime.data[model_v4.RAW_MODEL_INPUTS].median(numeric_only=True)
    st.sidebar.markdown("### 新工况输入")
    st.sidebar.caption("默认值来自公开合成数据的中位数，可直接修改。")

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
        values[column] = float(
            st.sidebar.number_input(
                labels[column],
                min_value=lower,
                value=float(median[column]),
                step=0.1,
                format="%.4f",
                key=f"condition_{column}",
            )
        )
    tn_standard = float(
        st.sidebar.number_input(
            "出水 TN 限值（mg/L）",
            min_value=0.1,
            value=15.0,
            step=0.5,
        )
    )
    return values, tn_standard


def _render_control_tab(st, runtime: DemoRuntime, values: dict[str, float], tn_standard: float):
    st.subheader("新工况预测与分级曝气建议")
    st.caption(
        "模型只在合成演示数据的历史范围内给出 A/B/C 级建议；"
        "它用于展示工程方法，不可直接连接现场设备。"
    )
    try:
        condition = make_condition(values)
        prediction, recommendation = predict_condition(
            runtime, condition, tn_standard=tn_standard
        )
        curve = aeration_response_curve(runtime, condition)
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
        st.caption("曲线仅覆盖训练数据中的曝气范围；保守上界包含跨日期 90% 误差裕量。")

    st.markdown("#### TN 去除率与 SND 响应")
    ratio_curve = curve.set_index(model_v4.AERATION_COL)[
        ["预测TN去除率", "预测SND率"]
    ]
    st.line_chart(ratio_curve)


def _render_quality_tab(st, runtime: DemoRuntime):
    st.subheader("训练数据质量闸门")
    st.caption(
        "上传文件只在当前应用进程中解析，不写入仓库。公开部署时仍不建议上传敏感原始数据。"
    )
    upload = st.file_uploader("上传标准 CSV（可选）", type=["csv"])
    if upload is None:
        frame = runtime.data.drop(columns=[model_v4.EFFLUENT_PROXY_COL], errors="ignore")
        encoding = "内置 UTF-8 合成样例"
    else:
        try:
            frame, encoding = _read_uploaded_csv(upload.getvalue())
        except ValueError as exc:
            st.error(str(exc))
            return

    summary, issues = validate_model_frame(frame)
    row1 = st.columns(4)
    row1[0].metric("记录数", summary.get("rows", 0))
    row1[1].metric("有效记录", summary.get("valid_rows", 0))
    row1[2].metric("日期组", summary.get("date_groups", 0))
    row1[3].metric("训练就绪", "是" if summary.get("train_ready") else "否")
    st.caption(f"识别编码：{encoding}")

    if issues:
        st.warning(f"检测到 {len(issues)} 项质量问题。")
        st.dataframe(
            pd.DataFrame([issue.as_dict() for issue in issues]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("字段、样本量、日期组数和数值范围均通过校验。")
    with st.expander("查看数据预览"):
        st.dataframe(frame.head(20), use_container_width=True, hide_index=True)


def _render_evidence_tab(st, runtime: DemoRuntime):
    st.subheader("可信度与适用边界")
    st.warning(
        "下表来自公开合成数据，只用于 CI 和界面验收；不能作为论文实验结果或现场性能承诺。"
    )
    metrics = runtime.metrics.copy()
    st.dataframe(
        metrics.style.format(
            {"日期分组R²": "{:.3f}", "MAE": "{:.3f}", "RMSE": "{:.3f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    a, b, c = st.columns(3)
    a.metric("合成记录", runtime.info["clean_rows"])
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
        """# 本地网页
python -m pip install -e \".[web,plot]\"
streamlit run streamlit_app.py

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
        page_title="OUR-SND 智能曝气调控",
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
          <h1>OUR-SND 智能曝气调控平台</h1>
          <p>把污水处理实验数据转化为可审计预测、风险边界和分级曝气建议。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    @st.cache_resource(show_spinner="正在建立合成数据演示模型……")
    def _load_runtime() -> DemoRuntime:
        return build_demo_runtime()

    try:
        runtime = _load_runtime()
    except Exception as exc:
        st.error(f"演示模型启动失败：{exc}")
        st.stop()

    values, tn_standard = _condition_inputs(st, runtime)
    control, quality, evidence, engineering = st.tabs(
        ["智能调控", "数据质检", "模型可信度", "工程架构"]
    )
    with control:
        _render_control_tab(st, runtime, values, tn_standard)
    with quality:
        _render_quality_tab(st, runtime)
    with evidence:
        _render_evidence_tab(st, runtime)
    with engineering:
        _render_engineering_tab(st)

    st.divider()
    st.caption(
        "科研演示软件 · MIT License · 合成数据不可用于论文结论或现场自动控制"
    )


if __name__ == "__main__":
    main()
