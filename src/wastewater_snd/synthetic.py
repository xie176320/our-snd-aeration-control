"""Deterministic synthetic records used only for software verification.

The generator is intentionally independent of private workbooks and measured
plant data.  The far-future dates make the fictional provenance visible in
screenshots and exported demo files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import (
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
    TN_OUT_COL,
)


SYNTHETIC_START_DATE = "2099-01-05"


def demo_frame(seed: int = 42) -> pd.DataFrame:
    """Return 60 fictional records with repeatable dose-response structure."""

    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    dates = pd.date_range(SYNTHETIC_START_DATE, periods=12, freq="D")
    for date_index, sample_date in enumerate(dates):
        temperature = 19.0 + 0.35 * date_index + rng.normal(0.0, 0.15)
        tn_in = 24.0 + 0.55 * date_index + rng.normal(0.0, 0.5)
        cod_in = 175.0 + 4.5 * date_index + rng.normal(0.0, 5.0)
        h_max = 31.0 + 0.8 * date_index + rng.normal(0.0, 0.8)
        a_max = 8.5 + 0.30 * date_index + rng.normal(0.0, 0.25)
        n_max = 4.2 + 0.16 * date_index + rng.normal(0.0, 0.15)
        h_live = 7.2 + 0.22 * date_index + rng.normal(0.0, 0.2)
        for sample_index, aeration in enumerate((2.0, 3.0, 4.0, 5.0, 6.0), start=1):
            removal = np.clip(
                0.48
                + 0.038 * aeration
                + 0.004 * date_index
                + rng.normal(0.0, 0.008),
                0.0,
                1.0,
            )
            snd = np.clip(
                0.44
                + 0.030 * aeration
                + 0.005 * date_index
                + rng.normal(0.0, 0.009),
                0.0,
                1.0,
            )
            records.append(
                {
                    DATE_COL: sample_date.strftime("%Y-%m-%d"),
                    H_MAX_COL: round(h_max, 4),
                    A_MAX_COL: round(a_max, 4),
                    N_MAX_COL: round(n_max, 4),
                    TEMP_COL: round(temperature, 4),
                    AERATION_COL: aeration,
                    SND_COL: round(float(snd), 6),
                    REMOVAL_COL: round(float(removal), 6),
                    TN_IN_COL: round(tn_in, 4),
                    COD_IN_COL: round(cod_in, 4),
                    H_LIVE_COL: round(h_live, 4),
                    A_LIVE_COL: round(0.8 + 0.04 * sample_index, 4),
                    N_LIVE_COL: round(0.4 + 0.02 * sample_index, 4),
                    TN_OUT_COL: round(tn_in * (1.0 - removal), 4),
                }
            )
    return pd.DataFrame.from_records(records)
