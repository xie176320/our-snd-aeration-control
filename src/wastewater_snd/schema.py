from __future__ import annotations

from dataclasses import dataclass


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
TN_OUT_COL = "出水TN实测(mg/L)"

REQUIRED_MODEL_COLUMNS = [
    DATE_COL,
    H_MAX_COL,
    A_MAX_COL,
    N_MAX_COL,
    TEMP_COL,
    AERATION_COL,
    SND_COL,
    REMOVAL_COL,
    TN_IN_COL,
    COD_IN_COL,
    H_LIVE_COL,
]

OPTIONAL_MODEL_COLUMNS = [A_LIVE_COL, N_LIVE_COL, TN_OUT_COL]

MODEL_DRAFT_COLUMNS = REQUIRED_MODEL_COLUMNS + OPTIONAL_MODEL_COLUMNS + [
    "source_dataset",
    "source_sample_index",
]


@dataclass(frozen=True)
class QualityIssue:
    severity: str
    code: str
    source: str
    sheet: str
    row: int | None
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "code": self.code,
            "source": self.source,
            "sheet": self.sheet,
            "row": self.row,
            "message": self.message,
        }

