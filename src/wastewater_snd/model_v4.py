# -*- coding: utf-8 -*-
"""
曝气预测调控模型 v4.0（日期等权 + 小样本集成）

适用数据：符合公开数据契约的标准 CSV。

这版脚本面向同日重复、跨批次差异明显的小样本结构，包含五项设计：
1. 候选模型只按日期整批留出评价，并同时报告固定 5 折、重复分组 5 折和
   留一日期三种结果；部署模型按重复分组验证的稳定得分选型。
2. 同一天有 1、2、5 或 10 条记录。增强 TN 模型先按日期取中位数，使每个
   日期具有相同训练权重，再集成 RBF-SVR 与线性 SVR，减少重复记录偏权。
3. 增强 SND 模型联合使用直接 SND-SVR、由预测 TN 映射得到的 SND，以及
   TN/SND 共享潜变量 PLS；预测时不需要输入真实 TN 去除率或真实 SND。
4. 直接预测 TN 去除率和 SND 率；若 CSV 没有独立实测出水 TN，出水 TN
   只能按 进水TN × (1 - 预测TN去除率) 推导，不能把它当成第三个独立目标。
5. 将“剂量效应可识别性”作为分级控制条件：
   - 数据和工况均可信时，给出正式优化推荐；
   - 数据不足但当前预测留有安全余量时，给出小步试运行推荐；
   - 当前工况风险较高时，给出维持当前值的数值推荐。
   因此程序每次都会输出一个明确曝气量，但不会把缺少试验依据的数学最小值
   伪装成已经验证的现场最优值。

运行：
    python aeration_deployment_v4.py

无交互验证：
    python aeration_deployment_v4.py --no-interactive

指定 CSV：
    python aeration_deployment_v4.py --data data/processed/model_input.csv

依赖：pandas、numpy、scikit-learn、joblib；matplotlib 为可选依赖。
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

# 固定为单进程即可满足当前小样本规模，并避免部分容器/Windows 环境的
# 物理核心探测警告。该设置不改变模型算法或随机种子。
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------------------------------------------------------
# 0. 配置与字段
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = Path("data/processed/model_input.csv")
DEFAULT_OUTPUT_DIR = BASE_DIR / "v4_model_outputs"
RANDOM_STATE = 42
MAX_FOLDS = 5
REPEATED_GROUP_FOLDS = 10
TN_STANDARD_DEFAULT = 15.0
TN_SAFETY_MARGIN_DEFAULT = 1.0
TRIAL_STEP_ABS_DEFAULT = 0.5
TRIAL_STEP_FRACTION_DEFAULT = 0.05
SEARCH_STEP_DEFAULT = 0.10

DATE_COL = "日期"
H_MAX_COL = "异养菌最大OUR"
A_MAX_COL = "AOB最大OUR"
N_MAX_COL = "NOB最大OUR"
TEMP_COL = "温度（摄氏度）"
AERATION_COL = "曝气量(L/min)"
SND_COL = "SND率"
REMOVAL_COL = "TN去除率"
TN_IN_COL = "进水TN(mg/L)"
COD_IN_COL = "进水COD(mg/L)"
H_LIVE_COL = "异养菌实时OUR"
A_LIVE_COL = "AOB实时OUR"
N_LIVE_COL = "NOB实时OUR"
EFFLUENT_PROXY_COL = "出水TN_推导值(mg/L)"

RAW_MODEL_INPUTS = [
    H_MAX_COL,
    A_MAX_COL,
    N_MAX_COL,
    TEMP_COL,
    AERATION_COL,
    TN_IN_COL,
    COD_IN_COL,
    H_LIVE_COL,
]
TARGETS = [REMOVAL_COL, SND_COL]

BASE_FEATURE_NAMES = [
    "H_max",
    "AOB_max",
    "NOB_max",
    "temperature",
    "TN_in",
    "COD_in",
    "C_N",
    "aeration",
]

ENGINEERED_FEATURE_NAMES = BASE_FEATURE_NAMES + [
    "H_live",
    "AOB_per_H",
    "NOB_per_AOB",
    "autotrophic_sum",
    "H_per_autotrophic",
    "aeration_per_TN",
    "aeration_per_COD",
    "TN_per_aeration",
    "COD_per_aeration",
    "temperature_x_aeration",
    "Q10_H",
    "Q10_AOB",
    "Q10_NOB",
    "log1p_COD",
]

BASE9_FEATURE_NAMES = [
    "H_max",
    "AOB_max",
    "NOB_max",
    "temperature",
    "aeration",
    "TN_in",
    "COD_in",
    "H_live",
    "C_N",
]

# 精简后的机理特征。相关性审计显示，绝对 H/AOB/NOB 最大 OUR 之间共线性
# 较强；保留 AOB/H、NOB/AOB 比值后，跨日期表现比堆叠全部工程特征更稳。
LOAD_FEATURE_NAMES = [
    "temperature",
    "aeration",
    "TN_in",
    "COD_in",
    "H_live",
    "C_N",
    "AOB_per_H",
    "NOB_per_AOB",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    target: str
    kind: str
    feature_set: str
    train_mode: str = "raw"
    estimator_code: str = ""


@dataclass
class OOFResult:
    predictions: np.ndarray
    fold_id: np.ndarray
    metrics: dict[str, float]


class ShuffledGroupKFoldCompat:
    """Compatibility splitter for scikit-learn versions before GroupKFold.shuffle.

    Every date remains wholly in one fold. Newer scikit-learn versions use their
    native implementation; this class only keeps the documented repeat validation
    runnable on older supported environments.
    """

    def __init__(self, n_splits: int, random_state: int):
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)

    def split(self, x, y=None, groups=None):
        if groups is None:
            raise ValueError("分组交叉验证必须提供日期 groups。")
        group_array = np.asarray(groups)
        unique_groups = np.unique(group_array)
        if len(unique_groups) < self.n_splits:
            raise ValueError("日期组数量少于折数。")
        shuffled = unique_groups.copy()
        np.random.default_rng(self.random_state).shuffle(shuffled)
        for valid_groups in np.array_split(shuffled, self.n_splits):
            valid_mask = np.isin(group_array, valid_groups)
            yield np.flatnonzero(~valid_mask), np.flatnonzero(valid_mask)

    def get_n_splits(self, x=None, y=None, groups=None):
        return self.n_splits


def make_group_kfold(
    n_splits: int,
    *,
    shuffle: bool = False,
    random_state: int | None = None,
):
    if not shuffle:
        return GroupKFold(n_splits=n_splits)
    try:
        return GroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
    except TypeError:
        return ShuffledGroupKFoldCompat(
            n_splits=n_splits,
            random_state=RANDOM_STATE if random_state is None else random_state,
        )


def make_pls(n_components: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                PLSRegression(
                    n_components=n_components,
                    scale=False,
                    max_iter=1000,
                ),
            ),
        ]
    )


def make_ridge(alpha: float) -> Pipeline:
    return Pipeline(
        [("scale", StandardScaler()), ("model", Ridge(alpha=alpha))]
    )


def make_estimator(code: str) -> object:
    """用固定字符串配置创建估计器，便于复现与模型保存。"""
    if code == "pls2":
        return make_pls(2)
    if code == "ridge10":
        return make_ridge(10.0)
    if code == "ridge30":
        return make_ridge(30.0)
    if code == "tn_svr_rbf":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVR(kernel="rbf", C=1.0, gamma=0.01, epsilon=0.03)),
            ]
        )
    if code == "tn_svr_linear":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVR(kernel="linear", C=0.1, epsilon=0.03)),
            ]
        )
    if code == "snd_svr_rbf":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVR(kernel="rbf", C=1.0, gamma=0.003, epsilon=0.03)),
            ]
        )
    if code == "snd_svr_rbf_mean":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVR(kernel="rbf", C=1.0, gamma=0.03, epsilon=0.03)),
            ]
        )
    if code == "snd_svr_linear":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVR(kernel="linear", C=1.0, epsilon=0.03)),
            ]
        )
    raise ValueError(f"未知估计器配置：{code}")


def candidate_models(target: str) -> list[ModelSpec]:
    """预先固定的小候选集，避免在 41 个日期上进行大规模调参。"""
    if target == REMOVAL_COL:
        return [
            ModelSpec(
                "PLS2_原工程特征_记录级",
                target,
                "direct",
                "engineered",
                "raw",
                "pls2",
            ),
            ModelSpec(
                "Ridge30_精简机理特征_记录级",
                target,
                "direct",
                "loads8",
                "raw",
                "ridge30",
            ),
            ModelSpec(
                "RBF-SVR_精简机理特征_日期中位数",
                target,
                "direct",
                "loads8",
                "median",
                "tn_svr_rbf",
            ),
            ModelSpec(
                "线性SVR_精简机理特征_日期中位数",
                target,
                "direct",
                "loads8",
                "median",
                "tn_svr_linear",
            ),
            ModelSpec(
                "双SVR集成_日期等权",
                target,
                "tn_ensemble",
                "loads8",
                "median",
            ),
        ]

    return [
        ModelSpec(
            "Ridge10_原基础特征_记录级",
            target,
            "direct",
            "base",
            "raw",
            "ridge10",
        ),
        ModelSpec(
            "RBF-SVR_精简机理特征_记录级",
            target,
            "direct",
            "loads8",
            "raw",
            "snd_svr_rbf",
        ),
        ModelSpec(
            "RBF-SVR_精简机理特征_日期均值",
            target,
            "direct",
            "loads8",
            "mean",
            "snd_svr_rbf_mean",
        ),
        ModelSpec(
            "线性SVR_九项基础特征_日期中位数",
            target,
            "direct",
            "base9",
            "median",
            "snd_svr_linear",
        ),
        ModelSpec(
            "TN-SND联合集成",
            target,
            "snd_joint",
            "loads8",
            "联合",
        ),
    ]


# -----------------------------------------------------------------------------
# 1. 数据读取、清洗与特征构造
# -----------------------------------------------------------------------------
def read_csv_flexibly(path: Path) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for encoding in ("gbk", "utf-8-sig", "utf-8"):
        try:
            data = pd.read_csv(path, encoding=encoding, dtype={DATE_COL: "string"})
            data.columns = data.columns.astype(str).str.strip()
            return data, encoding
        except (UnicodeDecodeError, LookupError, ValueError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("CSV 编码读取失败：\n" + "\n".join(errors))


def clean_model_frame(
    raw: pd.DataFrame, *, encoding: str = "in-memory"
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate and normalize an already loaded model frame."""

    raw = raw.copy()
    raw.columns = raw.columns.astype(str).str.strip()
    missing = [col for col in [DATE_COL] + RAW_MODEL_INPUTS + TARGETS if col not in raw]
    if missing:
        raise ValueError(f"CSV 缺少必要字段：{missing}")

    raw[DATE_COL] = raw[DATE_COL].astype("string").str.strip().replace("", pd.NA)
    for col in RAW_MODEL_INPUTS + TARGETS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    before = len(raw)
    clean = raw.dropna(subset=[DATE_COL] + RAW_MODEL_INPUTS + TARGETS).copy()
    clean = clean[
        (clean[H_MAX_COL] >= 0)
        & (clean[A_MAX_COL] >= 0)
        & (clean[N_MAX_COL] >= 0)
        & (clean[H_LIVE_COL] >= 0)
        & (clean[TN_IN_COL] > 0)
        & (clean[COD_IN_COL] >= 0)
        & (clean[AERATION_COL] > 0)
        & clean[REMOVAL_COL].between(0, 1)
        & clean[SND_COL].between(0, 1)
    ].copy()
    clean.reset_index(drop=True, inplace=True)
    clean[EFFLUENT_PROXY_COL] = clean[TN_IN_COL] * (1.0 - clean[REMOVAL_COL])

    if len(clean) < 40:
        raise ValueError(f"有效样本仅 {len(clean)} 条，不足以进行本脚本的 5 折比较。")
    if clean[DATE_COL].nunique() < 5:
        raise ValueError("有效日期/批次少于 5 个，无法进行可靠的按日期交叉验证。")

    empty_live_cols = [
        col
        for col in [A_LIVE_COL, N_LIVE_COL]
        if col in raw.columns and raw[col].notna().sum() == 0
    ]
    info = {
        "encoding": encoding,
        "raw_rows": before,
        "clean_rows": len(clean),
        "removed_rows": before - len(clean),
        "date_groups": clean[DATE_COL].nunique(),
        "empty_live_columns": empty_live_cols,
    }
    return clean, info


def load_and_clean_data(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{path}\n"
            "请使用 --data 指定符合数据契约的 CSV 路径。"
        )

    raw, encoding = read_csv_flexibly(path)
    return clean_model_frame(raw, encoding=encoding)


def build_features(data: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    eps = 1e-6
    h = data[H_MAX_COL].astype(float)
    a = data[A_MAX_COL].astype(float)
    n = data[N_MAX_COL].astype(float)
    temp = data[TEMP_COL].astype(float)
    air = data[AERATION_COL].astype(float)
    tn = data[TN_IN_COL].astype(float)
    cod = data[COD_IN_COL].astype(float)
    live = data[H_LIVE_COL].astype(float)

    features = pd.DataFrame(
        {
            "H_max": h,
            "AOB_max": a,
            "NOB_max": n,
            "temperature": temp,
            "TN_in": tn,
            "COD_in": cod,
            "C_N": cod / tn,
            "aeration": air,
            "H_live": live,
            "AOB_per_H": a / (h + eps),
            "NOB_per_AOB": n / (a + eps),
        },
        index=data.index,
    )
    if feature_set == "base":
        return features[BASE_FEATURE_NAMES]
    if feature_set == "base9":
        return features[BASE9_FEATURE_NAMES]
    if feature_set == "loads8":
        return features[LOAD_FEATURE_NAMES]
    if feature_set != "engineered":
        raise ValueError(f"未知特征集：{feature_set}")

    q10 = 2.0 ** ((temp - 20.0) / 10.0)
    features = features.assign(
        autotrophic_sum=a + n,
        H_per_autotrophic=h / (a + n + eps),
        aeration_per_TN=air / tn,
        aeration_per_COD=air / (cod + eps),
        TN_per_aeration=tn / air,
        COD_per_aeration=cod / air,
        temperature_x_aeration=temp * air,
        Q10_H=h / q10,
        Q10_AOB=a / q10,
        Q10_NOB=n / q10,
        log1p_COD=np.log1p(cod),
    )
    return features[ENGINEERED_FEATURE_NAMES]


# -----------------------------------------------------------------------------
# 2. 日期等权训练、联合集成与三重日期级验证
# -----------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def aggregate_training_rows(
    data: pd.DataFrame,
    x: pd.DataFrame,
    y: np.ndarray,
    mode: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """把同日记录聚合为一个训练点；验证集始终保留原始记录。"""
    if mode == "raw":
        return x.reset_index(drop=True), np.asarray(y, dtype=float)
    if mode not in {"mean", "median"}:
        raise ValueError(f"未知训练聚合方式：{mode}")
    frame = x.reset_index(drop=True).copy()
    frame["__target"] = np.asarray(y, dtype=float)
    frame["__date"] = data[DATE_COL].astype(str).reset_index(drop=True)
    grouped = frame.groupby("__date", sort=False)
    aggregate = grouped.mean(numeric_only=True) if mode == "mean" else grouped.median(numeric_only=True)
    return aggregate[x.columns], aggregate["__target"].to_numpy(dtype=float)


def fit_tn_ensemble(data: pd.DataFrame) -> dict[str, object]:
    x = build_features(data, "loads8")
    y = data[REMOVAL_COL].to_numpy(dtype=float)
    x_date, y_date = aggregate_training_rows(data, x, y, "median")
    rbf = make_estimator("tn_svr_rbf")
    linear = make_estimator("tn_svr_linear")
    rbf.fit(x_date, y_date)
    linear.fit(x_date, y_date)
    return {
        "kind": "tn_ensemble",
        "feature_set": "loads8",
        "rbf": rbf,
        "linear": linear,
        "weights": (0.5, 0.5),
    }


def predict_tn_ensemble(bundle: dict[str, object], data: pd.DataFrame) -> np.ndarray:
    x = build_features(data, "loads8")
    rbf_prediction = np.asarray(bundle["rbf"].predict(x)).reshape(-1)
    linear_prediction = np.asarray(bundle["linear"].predict(x)).reshape(-1)
    weight_rbf, weight_linear = bundle["weights"]
    return weight_rbf * rbf_prediction + weight_linear * linear_prediction


def fit_snd_joint(data: pd.DataFrame) -> dict[str, object]:
    """拟合不依赖新工况真实目标值的 TN-SND 联合集成。"""
    x = build_features(data, "loads8")
    y_tn = data[REMOVAL_COL].to_numpy(dtype=float)
    y_snd = data[SND_COL].to_numpy(dtype=float)

    direct = make_estimator("snd_svr_rbf")
    direct.fit(x, y_snd)

    tn_model = fit_tn_ensemble(data)
    relation = Ridge(alpha=0.001)
    relation.fit(y_tn.reshape(-1, 1), y_snd)

    shared_target = 0.5 * (y_tn + y_snd)
    x_date, shared_date = aggregate_training_rows(
        data, x, shared_target, "median"
    )
    shared = make_pls(2)
    shared.fit(x_date, shared_date)
    snd_shared_offset = float(np.mean(y_snd - shared_target))

    return {
        "kind": "snd_joint",
        "feature_set": "loads8",
        "direct": direct,
        "tn_model": tn_model,
        "relation": relation,
        "shared": shared,
        "snd_shared_offset": snd_shared_offset,
        # 权重在模型搜索前固定；三项和为 1，避免在当前 41 个日期上再拟合元模型。
        "weights": (0.40, 0.35, 0.25),
    }


def predict_snd_joint(bundle: dict[str, object], data: pd.DataFrame) -> np.ndarray:
    x = build_features(data, "loads8")
    direct = np.asarray(bundle["direct"].predict(x)).reshape(-1)
    tn_prediction = predict_tn_ensemble(bundle["tn_model"], data)
    from_tn = np.asarray(
        bundle["relation"].predict(tn_prediction.reshape(-1, 1))
    ).reshape(-1)
    shared = np.asarray(bundle["shared"].predict(x)).reshape(-1)
    shared = shared + float(bundle["snd_shared_offset"])
    weight_direct, weight_relation, weight_shared = bundle["weights"]
    return (
        weight_direct * direct
        + weight_relation * from_tn
        + weight_shared * shared
    )


def fit_candidate(spec: ModelSpec, data: pd.DataFrame) -> dict[str, object]:
    if spec.kind == "tn_ensemble":
        return fit_tn_ensemble(data)
    if spec.kind == "snd_joint":
        return fit_snd_joint(data)
    if spec.kind != "direct":
        raise ValueError(f"未知候选模型类型：{spec.kind}")

    x = build_features(data, spec.feature_set)
    y = data[spec.target].to_numpy(dtype=float)
    x_train, y_train = aggregate_training_rows(data, x, y, spec.train_mode)
    model = make_estimator(spec.estimator_code)
    model.fit(x_train, y_train)
    return {
        "kind": "direct",
        "feature_set": spec.feature_set,
        "model": model,
    }


def predict_fitted(bundle: dict[str, object], data: pd.DataFrame) -> np.ndarray:
    kind = str(bundle["kind"])
    if kind == "tn_ensemble":
        prediction = predict_tn_ensemble(bundle, data)
    elif kind == "snd_joint":
        prediction = predict_snd_joint(bundle, data)
    elif kind == "direct":
        x = build_features(data, str(bundle["feature_set"]))
        prediction = np.asarray(bundle["model"].predict(x)).reshape(-1)
    else:
        raise ValueError(f"未知已训练模型类型：{kind}")
    return np.clip(np.asarray(prediction).reshape(-1), 0.0, 1.0)


def oof_predict(
    spec: ModelSpec,
    data: pd.DataFrame,
    splitter: object,
) -> OOFResult:
    y = data[spec.target].to_numpy(dtype=float)
    dates = data[DATE_COL].astype(str)
    predictions = np.full(len(data), np.nan, dtype=float)
    fold_id = np.full(len(data), -1, dtype=int)

    split_source = np.zeros((len(data), 1), dtype=float)
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(split_source, y, groups=dates), start=1
    ):
        fitted = fit_candidate(spec, data.iloc[train_idx].copy())
        predictions[valid_idx] = predict_fitted(
            fitted, data.iloc[valid_idx].copy()
        )
        fold_id[valid_idx] = fold

    if np.isnan(predictions).any() or (fold_id < 0).any():
        raise RuntimeError("交叉验证预测不完整。")
    return OOFResult(predictions, fold_id, regression_metrics(y, predictions))


def evaluate_candidates(data: pd.DataFrame):
    evaluation_rows: list[dict[str, object]] = []
    oof_store: dict[tuple[str, str, str], OOFResult] = {}
    spec_store: dict[tuple[str, str], ModelSpec] = {}

    for target in TARGETS:
        for spec in candidate_models(target):
            spec_store[(target, spec.name)] = spec
            fixed = oof_predict(
                spec,
                data,
                make_group_kfold(
                    n_splits=min(MAX_FOLDS, data[DATE_COL].nunique())
                ),
            )
            logo = oof_predict(spec, data, LeaveOneGroupOut())
            oof_store[(target, spec.name, "固定日期5折")] = fixed
            oof_store[(target, spec.name, "留一日期")] = logo

            repeated_r2: list[float] = []
            repeated_mae: list[float] = []
            for repeat in range(REPEATED_GROUP_FOLDS):
                repeated = oof_predict(
                    spec,
                    data,
                    make_group_kfold(
                        n_splits=min(MAX_FOLDS, data[DATE_COL].nunique()),
                        shuffle=True,
                        random_state=RANDOM_STATE + repeat,
                    ),
                )
                repeated_r2.append(repeated.metrics["r2"])
                repeated_mae.append(repeated.metrics["mae"])

            repeated_mean = float(np.mean(repeated_r2))
            repeated_std = float(np.std(repeated_r2, ddof=1))
            stability_score = repeated_mean - 0.25 * repeated_std
            evaluation_rows.append(
                {
                    "目标": target,
                    "模型": spec.name,
                    "类型": spec.kind,
                    "特征集": spec.feature_set,
                    "训练层级": spec.train_mode,
                    "固定5折_R2": fixed.metrics["r2"],
                    "固定5折_MAE": fixed.metrics["mae"],
                    "固定5折_RMSE": fixed.metrics["rmse"],
                    "重复分组_R2均值": repeated_mean,
                    "重复分组_R2中位数": float(np.median(repeated_r2)),
                    "重复分组_R2标准差": repeated_std,
                    "重复分组_R2最小值": float(np.min(repeated_r2)),
                    "重复分组_MAE均值": float(np.mean(repeated_mae)),
                    "留一日期_R2": logo.metrics["r2"],
                    "留一日期_MAE": logo.metrics["mae"],
                    "稳定选型得分": stability_score,
                }
            )

    evaluation = pd.DataFrame(evaluation_rows)
    selected: dict[str, ModelSpec] = {}
    for target in TARGETS:
        strict = evaluation[evaluation["目标"] == target].sort_values(
            ["稳定选型得分", "固定5折_R2", "固定5折_RMSE"],
            ascending=[False, False, True],
        )
        best_name = str(strict.iloc[0]["模型"])
        selected[target] = spec_store[(target, best_name)]
    return evaluation, selected, oof_store


def conformal_abs_error_quantile(
    y_true: np.ndarray, y_pred: np.ndarray, coverage: float = 0.90
) -> float:
    residuals = np.sort(np.abs(y_true - y_pred))
    rank = min(len(residuals), math.ceil((len(residuals) + 1) * coverage))
    return float(residuals[rank - 1])


def train_selected_models(
    data: pd.DataFrame,
    selected: dict[str, ModelSpec],
    oof_store: dict[tuple[str, str, str], OOFResult],
):
    trained: dict[str, object] = {}
    error_q90: dict[str, float] = {}
    support: dict[str, dict[str, object]] = {}

    for target, spec in selected.items():
        x = build_features(data, spec.feature_set)
        y = data[target].to_numpy(dtype=float)
        trained[target] = fit_candidate(spec, data)

        strict_oof = oof_store[(target, spec.name, "固定日期5折")]
        error_q90[target] = conformal_abs_error_quantile(y, strict_oof.predictions)

        scaler = StandardScaler().fit(x)
        x_scaled = scaler.transform(x)
        if len(x_scaled) >= 2:
            two_nn = NearestNeighbors(n_neighbors=2).fit(x_scaled)
            distances = two_nn.kneighbors(x_scaled)[0][:, 1]
            limit = float(np.quantile(distances, 0.95))
        else:
            limit = float("inf")
        one_nn = NearestNeighbors(n_neighbors=1).fit(x_scaled)
        support[target] = {
            "scaler": scaler,
            "nearest_neighbor": one_nn,
            "distance_limit": limit,
        }
    return trained, error_q90, support


# -----------------------------------------------------------------------------
# 3. 数据结构与曝气剂量效应诊断
# -----------------------------------------------------------------------------
def duplicate_input_diagnostic(data: pd.DataFrame) -> dict[str, int]:
    base = build_features(data, "base")
    combined = base.copy()
    combined[REMOVAL_COL] = data[REMOVAL_COL].to_numpy()
    combined[SND_COL] = data[SND_COL].to_numpy()
    grouped = combined.groupby(BASE_FEATURE_NAMES, dropna=False).agg(
        样本数=(REMOVAL_COL, "size"),
        TN极差=(REMOVAL_COL, lambda values: float(values.max() - values.min())),
        SND极差=(SND_COL, lambda values: float(values.max() - values.min())),
    )
    duplicates = grouped[grouped["样本数"] > 1]
    return {
        "independent_inputs": int(len(grouped)),
        "duplicate_groups": int(len(duplicates)),
        "ambiguous_tn_groups": int((duplicates["TN极差"] > 0.05).sum()),
        "ambiguous_snd_groups": int((duplicates["SND极差"] > 0.05).sum()),
    }


def aeration_effect_diagnostic(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, group in data.groupby(DATE_COL):
        if len(group) < 3 or group[AERATION_COL].nunique() < 2:
            continue
        x = group[[AERATION_COL]].to_numpy(dtype=float)
        row: dict[str, object] = {
            DATE_COL: date,
            "样本数": len(group),
            "曝气极差": float(group[AERATION_COL].max() - group[AERATION_COL].min()),
        }
        for target in TARGETS:
            model = LinearRegression().fit(x, group[target].to_numpy(dtype=float))
            row[f"{target}_斜率"] = float(model.coef_[0])
        rows.append(row)
    return pd.DataFrame(rows)


def aeration_optimization_gate(effect_table: pd.DataFrame) -> dict[str, object]:
    if effect_table.empty:
        return {
            "allowed": False,
            "reason": "没有日期包含足够的曝气梯度。",
            "variable_dates": 0,
            "median_range": 0.0,
            "tn_sign_consistency": 0.0,
            "snd_sign_consistency": 0.0,
        }

    def sign_consistency(column: str) -> float:
        values = effect_table[column].to_numpy(dtype=float)
        nonzero = values[np.abs(values) > 1e-12]
        if len(nonzero) == 0:
            return 0.0
        positive = float(np.mean(nonzero > 0))
        return max(positive, 1.0 - positive)

    variable_dates = len(effect_table)
    median_range = float(effect_table["曝气极差"].median())
    tn_consistency = sign_consistency(f"{REMOVAL_COL}_斜率")
    snd_consistency = sign_consistency(f"{SND_COL}_斜率")
    allowed = (
        variable_dates >= 10
        and median_range >= 1.0
        and tn_consistency >= 0.70
        and snd_consistency >= 0.70
    )
    reasons = []
    if variable_dates < 10:
        reasons.append("有曝气梯度的日期不足 10 个")
    if median_range < 1.0:
        reasons.append("日期内曝气极差中位数小于 1.0 L/min")
    if tn_consistency < 0.70:
        reasons.append("TN 去除率对曝气的方向一致性低于 70%")
    if snd_consistency < 0.70:
        reasons.append("SND 对曝气的方向一致性低于 70%")
    return {
        "allowed": allowed,
        "reason": "；".join(reasons) if reasons else "剂量效应门控通过",
        "variable_dates": variable_dates,
        "median_range": median_range,
        "tn_sign_consistency": tn_consistency,
        "snd_sign_consistency": snd_consistency,
    }


# -----------------------------------------------------------------------------
# 4. 输出文件与图形
# -----------------------------------------------------------------------------
def save_validation_plot(
    data: pd.DataFrame,
    selected: dict[str, ModelSpec],
    oof_store: dict[tuple[str, str, str], OOFResult],
    output_path: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=160)
    labels = {REMOVAL_COL: "TN removal rate", SND_COL: "SND rate"}
    colors = {REMOVAL_COL: "#2563EB", SND_COL: "#059669"}
    for ax, target in zip(axes, TARGETS):
        spec = selected[target]
        result = oof_store[(target, spec.name, "固定日期5折")]
        observed = data[target].to_numpy(dtype=float)
        predicted = result.predictions
        low = float(min(observed.min(), predicted.min()))
        high = float(max(observed.max(), predicted.max()))
        ax.scatter(observed, predicted, s=24, alpha=0.72, color=colors[target])
        ax.plot([low, high], [low, high], "--", color="#475569", linewidth=1)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Cross-date OOF prediction")
        ax.set_title(
            f"{labels[target]}\nR2={result.metrics['r2']:.3f}, "
            f"MAE={result.metrics['mae']:.3f}"
        )
        ax.grid(alpha=0.2)
    fig.suptitle("Strict date-grouped cross-validation")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def save_outputs(
    data_path: Path,
    output_dir: Path,
    data: pd.DataFrame,
    evaluation: pd.DataFrame,
    selected: dict[str, ModelSpec],
    oof_store: dict[tuple[str, str, str], OOFResult],
    trained: dict[str, object],
    error_q90: dict[str, float],
    support: dict[str, dict[str, object]],
    effect_table: pd.DataFrame,
    gate: dict[str, object],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "evaluation": output_dir / "model_evaluation.csv",
        "oof": output_dir / "oof_predictions.csv",
        "date_error": output_dir / "date_error_summary.csv",
        "effect": output_dir / "aeration_effect_diagnostic.csv",
        "model": output_dir / "model_bundle.joblib",
        "plot": output_dir / "validation_plots.png",
    }
    evaluation.sort_values(
        ["目标", "稳定选型得分"], ascending=[True, False]
    ).to_csv(paths["evaluation"], index=False, encoding="utf-8-sig")

    oof_frame = data[
        [DATE_COL, TN_IN_COL, REMOVAL_COL, SND_COL, EFFLUENT_PROXY_COL]
    ].copy()
    for target in TARGETS:
        spec = selected[target]
        strict = oof_store[(target, spec.name, "固定日期5折")]
        logo = oof_store[(target, spec.name, "留一日期")]
        oof_frame[f"{target}_跨日期OOF预测"] = strict.predictions
        oof_frame[f"{target}_留一日期OOF预测"] = logo.predictions
        oof_frame[f"{target}_跨日期折号"] = strict.fold_id
    oof_frame["出水TN_跨日期OOF推导值(mg/L)"] = oof_frame[TN_IN_COL] * (
        1.0 - oof_frame[f"{REMOVAL_COL}_跨日期OOF预测"]
    )
    oof_frame.to_csv(paths["oof"], index=False, encoding="utf-8-sig")

    error_frame = oof_frame.assign(
        TN去除率_绝对误差=(
            oof_frame[REMOVAL_COL]
            - oof_frame[f"{REMOVAL_COL}_跨日期OOF预测"]
        ).abs(),
        SND率_绝对误差=(
            oof_frame[SND_COL]
            - oof_frame[f"{SND_COL}_跨日期OOF预测"]
        ).abs(),
        出水TN代理_绝对误差_mgL=(
            oof_frame[EFFLUENT_PROXY_COL]
            - oof_frame["出水TN_跨日期OOF推导值(mg/L)"]
        ).abs(),
    )
    date_error = (
        error_frame.groupby(DATE_COL, as_index=False)
        .agg(
            样本数=(REMOVAL_COL, "size"),
            TN去除率_平均绝对误差=("TN去除率_绝对误差", "mean"),
            TN去除率_最大绝对误差=("TN去除率_绝对误差", "max"),
            SND率_平均绝对误差=("SND率_绝对误差", "mean"),
            SND率_最大绝对误差=("SND率_绝对误差", "max"),
            出水TN代理_MAE_mgL=("出水TN代理_绝对误差_mgL", "mean"),
        )
        .sort_values("TN去除率_平均绝对误差", ascending=False)
    )
    date_error.to_csv(paths["date_error"], index=False, encoding="utf-8-sig")
    effect_table.to_csv(paths["effect"], index=False, encoding="utf-8-sig")

    raw_ranges = {
        col: (float(data[col].min()), float(data[col].max()))
        for col in RAW_MODEL_INPUTS
    }
    bundle = {
        "version": "4.0",
        "source_data": str(data_path),
        "selected_model_names": {
            target: selected[target].name for target in TARGETS
        },
        "feature_sets": {
            target: selected[target].feature_set for target in TARGETS
        },
        "selected_specs": selected,
        "models": trained,
        "strict_oof_abs_error_q90": error_q90,
        "support": support,
        "raw_training_ranges": raw_ranges,
        "historical_aeration_values": np.sort(data[AERATION_COL].unique()),
        "aeration_optimization_gate": gate,
        "training_data": data[[DATE_COL] + RAW_MODEL_INPUTS + TARGETS].copy(),
        "model_evaluation": evaluation,
        "validation_note": (
            "模型按重复日期分组验证的稳定得分选型；固定5折 OOF 用于误差区间。"
        ),
    }
    joblib.dump(bundle, paths["model"])
    plotted = save_validation_plot(data, selected, oof_store, paths["plot"])
    if not plotted:
        paths.pop("plot")
    return paths


# -----------------------------------------------------------------------------
# 5. 新工况预测与受控曝气搜索
# -----------------------------------------------------------------------------
def estimate_live_h(data: pd.DataFrame, row: pd.DataFrame, k: int = 5) -> float:
    match_cols = [H_MAX_COL, A_MAX_COL, N_MAX_COL, TEMP_COL, TN_IN_COL, COD_IN_COL]
    scaler = StandardScaler().fit(data[match_cols])
    train_scaled = scaler.transform(data[match_cols])
    input_scaled = scaler.transform(row[match_cols])
    neighbors = NearestNeighbors(n_neighbors=min(k, len(data))).fit(train_scaled)
    indices = neighbors.kneighbors(input_scaled)[1][0]
    return float(data.iloc[indices][H_LIVE_COL].median())


def support_distance(
    row: pd.DataFrame,
    target: str,
    spec: ModelSpec,
    support: dict[str, dict[str, object]],
) -> tuple[float, float, bool]:
    x = build_features(row, spec.feature_set)
    support_item = support[target]
    scaled = support_item["scaler"].transform(x)
    distance = float(
        support_item["nearest_neighbor"].kneighbors(scaled)[0][0, 0]
    )
    limit = float(support_item["distance_limit"])
    return distance, limit, distance <= limit


def raw_range_warnings(row: pd.DataFrame, data: pd.DataFrame) -> list[str]:
    messages = []
    for col in RAW_MODEL_INPUTS:
        value = float(row.iloc[0][col])
        low = float(data[col].min())
        high = float(data[col].max())
        if value < low or value > high:
            messages.append(f"{col}={value:.3g} 超出历史范围 [{low:.3g}, {high:.3g}]")
    return messages


def predict_one_condition(
    row: pd.DataFrame,
    selected: dict[str, ModelSpec],
    trained: dict[str, object],
) -> dict[str, float]:
    predictions: dict[str, float] = {}
    for target in TARGETS:
        value = float(predict_fitted(trained[target], row)[0])
        predictions[target] = float(np.clip(value, 0.0, 1.0))
    predictions[EFFLUENT_PROXY_COL] = float(
        row.iloc[0][TN_IN_COL] * (1.0 - predictions[REMOVAL_COL])
    )
    return predictions


def resolve_aeration_safety_floor(
    data: pd.DataFrame,
    minimum_safe_aeration: float | None = None,
) -> tuple[float, float, float]:
    """Return the effective mixing floor and observed aeration bounds.

    This is a hard engineering constraint for biological-tank mixing. It is
    deliberately separate from MBR membrane-scour air demand. When omitted,
    the observed lower bound remains the conservative default.
    """

    historical_low = float(data[AERATION_COL].min())
    historical_high = float(data[AERATION_COL].max())
    if not np.isfinite(historical_low) or not np.isfinite(historical_high):
        raise ValueError("训练数据的曝气范围必须是有限数字。")
    requested_floor = (
        historical_low
        if minimum_safe_aeration is None
        else float(minimum_safe_aeration)
    )
    if not np.isfinite(requested_floor) or requested_floor <= 0:
        raise ValueError("生化池混合安全下限必须是大于 0 的有限数字。")
    effective_floor = max(historical_low, requested_floor)
    if effective_floor > historical_high:
        raise ValueError(
            "生化池混合安全下限高于训练数据曝气上限，当前模型没有可用的安全搜索域；"
            "请先补充覆盖该下限以上的实测数据。"
        )
    return effective_floor, historical_low, historical_high


def recommend_aeration(
    row: pd.DataFrame,
    tn_standard: float,
    data: pd.DataFrame,
    selected: dict[str, ModelSpec],
    trained: dict[str, object],
    error_q90: dict[str, float],
    support: dict[str, dict[str, object]],
    gate: dict[str, object],
    trial_step_abs: float = TRIAL_STEP_ABS_DEFAULT,
    trial_step_fraction: float = TRIAL_STEP_FRACTION_DEFAULT,
    safety_margin: float = TN_SAFETY_MARGIN_DEFAULT,
    search_step: float = SEARCH_STEP_DEFAULT,
    minimum_safe_aeration: float | None = None,
) -> str:
    """把预测结果转换为始终有数值输出的分级曝气建议。

    A 级：剂量效应门控通过、候选工况在历史支持域内，并且 90% 误差裕量
    下仍满足 TN 限值。此时可以搜索稳态目标，但单次动作仍受信赖域限制。

    B 级：门控或联合支持域未通过，但各原始非曝气输入仍处于历史范围内，
    且当前预测有足够安全余量。只允许一次小步试运行，必须经一个 HRT 后
    的实测结果确认，才能进行下一步。

    C 级：不具备安全调节余量时，明确推荐维持当前曝气量。维持也是数值
    控制建议，避免程序在证据不足时输出危险的大幅升降。
    """
    if tn_standard <= 0:
        raise ValueError("出水 TN 限值必须大于 0。")
    search_step = max(float(search_step), 0.01)
    trial_step_abs = max(float(trial_step_abs), search_step)
    trial_step_fraction = max(float(trial_step_fraction), 0.0)
    safety_margin = max(float(safety_margin), 0.0)

    current_aeration = float(row.iloc[0][AERATION_COL])
    if not np.isfinite(current_aeration) or current_aeration <= 0:
        raise ValueError("当前曝气量必须是大于 0 的有限数字。")
    engineering_low, historical_low, historical_high = resolve_aeration_safety_floor(
        data,
        minimum_safe_aeration,
    )
    requested_floor = (
        historical_low
        if minimum_safe_aeration is None
        else float(minimum_safe_aeration)
    )

    def rounded(value: float) -> float:
        return max(search_step, float(round(value / search_step) * search_step))

    def rounded_action(value: float) -> float:
        """向当前值方向取整，确保取整后不突破单次调节上限。"""
        if value < current_aeration:
            result = math.ceil((value - 1e-10) / search_step) * search_step
        elif value > current_aeration:
            result = math.floor((value + 1e-10) / search_step) * search_step
        else:
            result = current_aeration
        return max(search_step, float(result))

    def evaluate(aeration: float) -> dict[str, object]:
        candidate = row.copy()
        candidate[AERATION_COL] = candidate[AERATION_COL].astype(float)
        candidate.loc[candidate.index[0], AERATION_COL] = float(aeration)
        prediction = predict_one_condition(candidate, selected, trained)
        removal_lower = max(0.0, prediction[REMOVAL_COL] - error_q90[REMOVAL_COL])
        conservative_effluent = float(
            candidate.iloc[0][TN_IN_COL] * (1.0 - removal_lower)
        )
        tn_distance, tn_limit, tn_supported = support_distance(
            candidate, REMOVAL_COL, selected[REMOVAL_COL], support
        )
        snd_distance, snd_limit, snd_supported = support_distance(
            candidate, SND_COL, selected[SND_COL], support
        )
        return {
            "aeration": float(aeration),
            "tn": prediction[EFFLUENT_PROXY_COL],
            "tn_upper": conservative_effluent,
            "removal": prediction[REMOVAL_COL],
            "snd": prediction[SND_COL],
            "tn_distance": tn_distance,
            "tn_limit": tn_limit,
            "tn_supported": tn_supported,
            "snd_distance": snd_distance,
            "snd_limit": snd_limit,
            "snd_supported": snd_supported,
        }

    # 先计算严格的全历史范围优化目标。只有门控和联合支持域均通过时，
    # 才允许把它称为正式优化目标。
    grid = np.arange(
        engineering_low,
        historical_high + search_step * 0.5,
        search_step,
    )
    grid = np.unique(
        np.clip(
            np.append(grid, [engineering_low, historical_high, current_aeration]),
            engineering_low,
            historical_high,
        )
    )
    strict_target = max(0.0, tn_standard - safety_margin)
    strict_candidates = []
    if gate["allowed"]:
        for value in grid:
            item = evaluate(float(value))
            if (
                item["tn_upper"] <= strict_target
                and item["tn_supported"]
                and item["snd_supported"]
            ):
                strict_candidates.append(item)

    level: str
    decision_reason: str
    steady_target: float | None = None
    current = evaluate(current_aeration)

    # 单次控制动作只允许位于当前值附近的信赖域内。
    maximum_step = min(
        trial_step_abs,
        current_aeration * trial_step_fraction
        if trial_step_fraction > 0
        else trial_step_abs,
    )
    maximum_step = max(maximum_step, search_step)

    if current_aeration < engineering_low:
        # The plant-defined mixing floor is a hard constraint, not a model
        # optimum, so it takes precedence over the normal one-step trust region.
        recommended_value = engineering_low
        level = "B级—混合安全下限纠偏推荐"
        decision_reason = (
            f"当前曝气量低于有效混合安全下限 {engineering_low:.2f} L/min；"
            "本次值由工程硬约束给出，不代表模型节能最优点。"
        )
    elif strict_candidates:
        best = min(strict_candidates, key=lambda item: float(item["aeration"]))
        steady_target = float(best["aeration"])
        delta_to_target = steady_target - current_aeration
        action_delta = float(np.clip(delta_to_target, -maximum_step, maximum_step))
        recommended_value = rounded_action(current_aeration + action_delta)
        level = "A级—正式优化的分步执行推荐"
        decision_reason = (
            "剂量效应门控和历史联合支持域均通过；为避免一次调节过大，"
            "本次动作仍限制在局部信赖域内。"
        )
    else:
        # 门控未通过时不再硬停止，而是在当前值附近构造最多一个小步动作。
        if current_aeration > historical_high:
            lower_value = rounded_action(
                max(engineering_low, current_aeration - maximum_step)
            )
            upper_value = current_aeration
        else:
            lower_value = rounded_action(
                max(engineering_low, current_aeration - maximum_step)
            )
            upper_value = rounded_action(
                min(historical_high, current_aeration + maximum_step)
            )
        local_values = sorted({lower_value, current_aeration, upper_value})
        local = [evaluate(value) for value in local_values]
        lower = max(
            (item for item in local if float(item["aeration"]) < current_aeration),
            key=lambda item: float(item["aeration"]),
            default=None,
        )

        # 联合距离可能超限，但若每个原始非曝气变量仍在历史实测范围内，
        # 可以允许一次小步验证；超出单变量范围时则只维持当前值。
        non_aeration_breaches = []
        for column in RAW_MODEL_INPUTS:
            if column == AERATION_COL:
                continue
            value = float(row.iloc[0][column])
            low = float(data[column].min())
            high = float(data[column].max())
            if value < low or value > high:
                non_aeration_breaches.append(column)

        current_margin = tn_standard - float(current["tn_upper"])
        safe_lower_step = (
            lower is not None
            and not non_aeration_breaches
            and current_margin >= safety_margin
            and float(lower["tn_upper"]) <= tn_standard
            and float(lower["tn"])
            <= float(current["tn"]) + 0.50
            and float(lower["snd"])
            >= float(current["snd"]) - 0.05
        )

        if safe_lower_step:
            recommended_value = float(lower["aeration"])
            level = "B级—保守降曝气试运行推荐"
            decision_reason = (
                f"当前 TN 保守上界距限值仍有 {current_margin:.2f} mg/L 余量，"
                "因此只建议下降一个小步，并以实测结果闭环确认。"
            )
        elif (
            float(current["tn_upper"]) > tn_standard
            and not non_aeration_breaches
        ):
            best_local = min(local, key=lambda item: float(item["tn_upper"]))
            improvement = float(current["tn_upper"]) - float(best_local["tn_upper"])
            if (
                float(best_local["aeration"]) != current_aeration
                and improvement >= 0.30
            ):
                recommended_value = float(best_local["aeration"])
                level = "B级—局部纠偏试运行推荐"
                decision_reason = (
                    f"当前 TN 保守上界超过限值，局部模型认为该小步可改善约 "
                    f"{improvement:.2f} mg/L；必须用实测结果确认方向。"
                )
            else:
                recommended_value = current_aeration
                level = "C级—维持当前曝气推荐"
                decision_reason = (
                    "当前 TN 风险偏高，且局部升降均没有显示出足够明确的改善，"
                    "因此本轮不改变曝气量。"
                )
        else:
            recommended_value = current_aeration
            level = "C级—维持当前曝气推荐"
            if non_aeration_breaches:
                decision_reason = (
                    "以下输入超出历史单变量范围："
                    + "、".join(non_aeration_breaches)
                    + "；本轮不改变曝气量。"
                )
            elif current_margin < safety_margin:
                decision_reason = (
                    f"当前 TN 保守安全余量仅 {current_margin:.2f} mg/L，小于设定的 "
                    f"{safety_margin:.2f} mg/L；本轮不降曝气。"
                )
            else:
                decision_reason = "当前值已接近历史曝气下限，本轮维持。"

    # Defensive final clamp: no model branch may cross the physical mixing floor.
    recommended_value = max(float(recommended_value), engineering_low)
    recommended = evaluate(recommended_value)
    change = recommended_value - current_aeration
    change_percent = 100.0 * change / current_aeration
    action = (
        "维持"
        if abs(change) < 1e-9
        else ("增加" if change > 0 else "降低")
    )
    lines = [
        f"本次建议曝气量：{recommended_value:.2f} L/min",
        (
            f"有效混合安全下限：{engineering_low:.2f} L/min"
            f"（配置 {requested_floor:.2f}；历史下限 {historical_low:.2f}）"
        ),
        f"推荐等级：{level}",
        (
            f"调节动作：{current_aeration:.2f} → {recommended_value:.2f} L/min；"
            f"{action} {abs(change):.2f} L/min（{change_percent:+.1f}%）"
        ),
    ]
    if steady_target is not None:
        lines.append(f"模型稳态目标：{steady_target:.2f} L/min（需分步到达）")
    lines.extend(
        [
            f"推荐值下推导出水 TN：{float(recommended['tn']):.2f} mg/L",
            (
                "加入跨日期 90% 误差裕量后的 TN 上界："
                f"{float(recommended['tn_upper']):.2f} mg/L"
            ),
            f"推荐值下 TN 去除率：{float(recommended['removal']):.2%}",
            f"推荐值下 SND 率：{float(recommended['snd']):.2%}",
            "推荐依据：" + decision_reason,
            "边界说明：该下限只约束生化池混合曝气，不包含 MBR 膜擦洗风量。",
        ]
    )

    if not gate["allowed"]:
        lines.append("证据限制：曝气剂量效应门控未通过——" + str(gate["reason"]) + "。")
    if not (recommended["tn_supported"] and recommended["snd_supported"]):
        lines.append(
            "工况限制：推荐值的特征组合仍超出历史联合支持域，故不能称为"
            "已验证的最低曝气量。"
        )

    if level.startswith("B级"):
        lines.append(
            "执行要求：调节后至少稳定 1 个 HRT，再实测出水 TN、NH4+-N、"
            "NO3--N、NO2--N 和 DO；若 TN 超限、氨氮超限或系统异常，立即恢复到 "
            f"{current_aeration:.2f} L/min。实测合格后才能进行下一次小步调节。"
        )
    elif level.startswith("C级"):
        lines.append(
            "下一步：先补充当前组合附近的分级曝气数据；在此之前程序仍会"
            "给出明确的维持值，但不宣称存在可靠的节能最优点。"
        )
    return "\n".join(lines)


def read_float(prompt: str, allow_blank: bool = False) -> float | None:
    while True:
        raw = input(prompt).strip()
        if raw.lower() == "q":
            raise KeyboardInterrupt
        if allow_blank and raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            print("请输入数字；输入 q 可退出。")


def interactive_loop(
    data: pd.DataFrame,
    selected: dict[str, ModelSpec],
    trained: dict[str, object],
    error_q90: dict[str, float],
    support: dict[str, dict[str, object]],
    gate: dict[str, object],
    trial_step_abs: float,
    trial_step_fraction: float,
    safety_margin: float,
    minimum_safe_aeration: float | None = None,
) -> None:
    while True:
        print("\n" + "=" * 68)
        print("输入新工况进行预测（输入 q 退出）")
        try:
            h_max = read_float("异养菌最大 OUR：")
            a_max = read_float("AOB 最大 OUR：")
            n_max = read_float("NOB 最大 OUR：")
            temp = read_float("温度（℃）：")
            tn_in = read_float("进水 TN（mg/L）：")
            cod_in = read_float("进水 COD（mg/L）：")
            h_live = read_float("异养菌实时 OUR（没有则直接回车估算）：", allow_blank=True)
            aeration = read_float("当前曝气量（L/min）：")
            tn_standard = read_float(
                f"出水 TN 限值（默认 {TN_STANDARD_DEFAULT} mg/L）：",
                allow_blank=True,
            )
            if tn_standard is None:
                tn_standard = TN_STANDARD_DEFAULT
            if tn_standard <= 0:
                raise ValueError("出水 TN 限值必须大于 0。")
            if tn_in is None or tn_in <= 0:
                raise ValueError("进水 TN 必须大于 0。")
            if cod_in is None or cod_in < 0:
                raise ValueError("进水 COD 不能小于 0。")
            if aeration is None or aeration <= 0:
                raise ValueError("曝气量必须大于 0。")

            row = pd.DataFrame(
                {
                    H_MAX_COL: [h_max],
                    A_MAX_COL: [a_max],
                    N_MAX_COL: [n_max],
                    TEMP_COL: [temp],
                    AERATION_COL: [aeration],
                    TN_IN_COL: [tn_in],
                    COD_IN_COL: [cod_in],
                    H_LIVE_COL: [h_live if h_live is not None else np.nan],
                }
            )
            if h_live is None:
                estimated = estimate_live_h(data, row)
                row.loc[0, H_LIVE_COL] = estimated
                print(f"异养菌实时 OUR 已按 5 个相似工况估算为：{estimated:.3f}")

            prediction = predict_one_condition(row, selected, trained)
            q90_removal = error_q90[REMOVAL_COL]
            q90_snd = error_q90[SND_COL]
            removal_low = max(0.0, prediction[REMOVAL_COL] - q90_removal)
            removal_high = min(1.0, prediction[REMOVAL_COL] + q90_removal)
            snd_low = max(0.0, prediction[SND_COL] - q90_snd)
            snd_high = min(1.0, prediction[SND_COL] + q90_snd)
            effluent_upper = tn_in * (1.0 - removal_low)

            print("\n[当前曝气量预测]")
            print(f"TN 去除率：{prediction[REMOVAL_COL]:.2%}")
            print(f"  跨日期 OOF 绝对误差 90% 范围：{removal_low:.2%}–{removal_high:.2%}")
            print(f"SND 率：{prediction[SND_COL]:.2%}")
            print(f"  跨日期 OOF 绝对误差 90% 范围：{snd_low:.2%}–{snd_high:.2%}")
            print(f"推导出水 TN：{prediction[EFFLUENT_PROXY_COL]:.2f} mg/L")
            print(f"  加入 TN 去除率误差后的保守上界：{effluent_upper:.2f} mg/L")

            warnings_list = raw_range_warnings(row, data)
            for target in TARGETS:
                distance, limit, supported = support_distance(
                    row, target, selected[target], support
                )
                if not supported:
                    warnings_list.append(
                        f"{target}模型的标准化最近邻距离 {distance:.2f} 超过历史上限 {limit:.2f}"
                    )
            if warnings_list:
                print("\n[外推警告]")
                for message in warnings_list:
                    print("- " + message)

            print("\n[曝气搜索]")
            print(
                recommend_aeration(
                    row=row,
                    tn_standard=float(tn_standard),
                    data=data,
                    selected=selected,
                    trained=trained,
                    error_q90=error_q90,
                    support=support,
                    gate=gate,
                    trial_step_abs=trial_step_abs,
                    trial_step_fraction=trial_step_fraction,
                    safety_margin=safety_margin,
                    minimum_safe_aeration=minimum_safe_aeration,
                )
            )
        except KeyboardInterrupt:
            print("\n系统已退出。")
            return
        except ValueError as exc:
            print(f"输入错误：{exc}")
            continue

        again = input("\n继续预测其他工况？（y/n，默认 y）：").strip().lower()
        if again == "n":
            return


# -----------------------------------------------------------------------------
# 6. 主程序与报告
# -----------------------------------------------------------------------------
def print_model_report(
    info: dict[str, object],
    diagnostics: dict[str, int],
    evaluation: pd.DataFrame,
    selected: dict[str, ModelSpec],
    oof_store: dict[tuple[str, str, str], OOFResult],
    data: pd.DataFrame,
    gate: dict[str, object],
    paths: dict[str, Path],
) -> None:
    print("\n" + "=" * 76)
    print("数据与模型验证报告")
    print("=" * 76)
    print(
        f"CSV 编码：{info['encoding']}；原始 {info['raw_rows']} 条；"
        f"有效 {info['clean_rows']} 条；删除 {info['removed_rows']} 条；"
        f"日期/批次 {info['date_groups']} 个。"
    )
    if info["empty_live_columns"]:
        print(
            "未纳入模型的全空字段：" + "、".join(info["empty_live_columns"])
        )
    print(
        f"独立基础输入工况 {diagnostics['independent_inputs']} 种；"
        f"重复工况 {diagnostics['duplicate_groups']} 组；"
        f"TN 波动>5个百分点 {diagnostics['ambiguous_tn_groups']} 组；"
        f"SND 波动>5个百分点 {diagnostics['ambiguous_snd_groups']} 组。"
    )

    print("\n[部署模型：只按日期级验证选型]")
    for target in TARGETS:
        spec = selected[target]
        strict = oof_store[(target, spec.name, "固定日期5折")]
        logo = oof_store[(target, spec.name, "留一日期")]
        row = evaluation[
            (evaluation["目标"] == target) & (evaluation["模型"] == spec.name)
        ].iloc[0]
        print(
            f"{target}：{spec.name}\n"
            f"  固定日期5折：R²={strict.metrics['r2']:.3f}, "
            f"MAE={strict.metrics['mae']:.3f}, RMSE={strict.metrics['rmse']:.3f}\n"
            f"  重复日期5折：R²={row['重复分组_R2均值']:.3f} ± "
            f"{row['重复分组_R2标准差']:.3f}\n"
            f"  留一日期：R²={logo.metrics['r2']:.3f}, "
            f"MAE={logo.metrics['mae']:.3f}"
        )

    removal_spec = selected[REMOVAL_COL]
    removal_oof = oof_store[
        (REMOVAL_COL, removal_spec.name, "固定日期5折")
    ].predictions
    observed_effluent = data[EFFLUENT_PROXY_COL].to_numpy(dtype=float)
    predicted_effluent = data[TN_IN_COL].to_numpy(dtype=float) * (1.0 - removal_oof)
    effluent_metrics = regression_metrics(observed_effluent, predicted_effluent)
    print(
        "推导出水 TN（不是独立实测目标）的跨日期结果："
        f"R²={effluent_metrics['r2']:.3f}, "
        f"MAE={effluent_metrics['mae']:.2f} mg/L, "
        f"RMSE={effluent_metrics['rmse']:.2f} mg/L"
    )

    print("\n[相对 v3 部署基线的固定日期5折提升]")
    baseline_names = {
        REMOVAL_COL: "PLS2_原工程特征_记录级",
        SND_COL: "Ridge10_原基础特征_记录级",
    }
    for target in TARGETS:
        selected_row = evaluation[
            (evaluation["目标"] == target)
            & (evaluation["模型"] == selected[target].name)
        ].iloc[0]
        baseline_row = evaluation[
            (evaluation["目标"] == target)
            & (evaluation["模型"] == baseline_names[target])
        ].iloc[0]
        gain = float(selected_row["固定5折_R2"] - baseline_row["固定5折_R2"])
        print(
            f"{target}：{baseline_row['固定5折_R2']:.3f} → "
            f"{selected_row['固定5折_R2']:.3f}（+{gain:.3f}）"
        )
    print("未使用随机记录拆分成绩参与选型。")
    print(
        "说明：以上仍是当前数据的内部日期级验证，不等同于独立未来月份验证；"
        "新日期实测数据应持续作为外部测试集。"
    )

    print("\n[曝气剂量效应门控]")
    print(
        f"有曝气梯度的日期：{gate['variable_dates']}；"
        f"日期内曝气极差中位数：{gate['median_range']:.2f} L/min；"
        f"TN方向一致性：{gate['tn_sign_consistency']:.1%}；"
        f"SND方向一致性：{gate['snd_sign_consistency']:.1%}。"
    )
    print("正式全范围优化：" + ("允许" if gate["allowed"] else "暂不允许"))
    if not gate["allowed"]:
        print("原因：" + str(gate["reason"]))
    print(
        "分级曝气推荐：始终输出（正式优化 / 小步试运行 / 维持当前值）。"
    )

    print("\n输出文件：")
    for name, path in paths.items():
        print(f"- {name}: {path}")
    print("=" * 76)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="曝气预测调控模型 v4.0")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help="符合数据契约的模型 CSV 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="评估表、OOF预测和模型文件的输出目录",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="只训练、验证和保存结果，不进入新工况输入界面",
    )
    parser.add_argument(
        "--trial-step-abs",
        type=float,
        default=TRIAL_STEP_ABS_DEFAULT,
        help="证据不足时单次允许调整的绝对上限，默认 0.5 L/min",
    )
    parser.add_argument(
        "--trial-step-fraction",
        type=float,
        default=TRIAL_STEP_FRACTION_DEFAULT,
        help="证据不足时单次允许调整的相对上限，默认当前曝气量的 5%%",
    )
    parser.add_argument(
        "--tn-safety-margin",
        type=float,
        default=TN_SAFETY_MARGIN_DEFAULT,
        help="允许降曝气所需的 TN 保守安全余量，默认 1.0 mg/L",
    )
    parser.add_argument(
        "--minimum-safe-aeration",
        type=float,
        default=None,
        help="生化池混合曝气硬下限；省略时使用训练数据历史下限",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    print("正在读取数据并进行三重日期级验证，请稍候……")
    data, info = load_and_clean_data(data_path)
    diagnostics = duplicate_input_diagnostic(data)
    effect_table = aeration_effect_diagnostic(data)
    gate = aeration_optimization_gate(effect_table)
    evaluation, selected, oof_store = evaluate_candidates(data)
    trained, error_q90, support = train_selected_models(data, selected, oof_store)
    paths = save_outputs(
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
    print_model_report(
        info=info,
        diagnostics=diagnostics,
        evaluation=evaluation,
        selected=selected,
        oof_store=oof_store,
        data=data,
        gate=gate,
        paths=paths,
    )

    if not args.no_interactive:
        interactive_loop(
            data=data,
            selected=selected,
            trained=trained,
            error_q90=error_q90,
            support=support,
            gate=gate,
            trial_step_abs=args.trial_step_abs,
            trial_step_fraction=args.trial_step_fraction,
            safety_margin=args.tn_safety_margin,
            minimum_safe_aeration=args.minimum_safe_aeration,
        )


if __name__ == "__main__":
    main()
