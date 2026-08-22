from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from wastewater_snd.calibrated import (
    fit_residual_calibrator,
    residual_correction,
    select_calibration_indices,
)
from wastewater_snd.model_v4 import ShuffledGroupKFoldCompat
from wastewater_snd.sources import canonical_date, validate_model_frame
from wastewater_snd.synthetic import demo_frame


class SourceTests(unittest.TestCase):
    def test_excel_date_formats_are_normalized(self) -> None:
        self.assertEqual(canonical_date(46244), "2026-08-10")
        self.assertEqual(canonical_date(8.1), "2026-08-10")
        self.assertEqual(canonical_date(5.6), "2026-05-06")

    def test_candidate_dates_resolve_ambiguous_decimal(self) -> None:
        candidates = {"2026-06-02", "2026-06-11"}
        self.assertEqual(
            canonical_date(6.2, candidates=candidates), "2026-06-02"
        )

    def test_demo_data_satisfies_v4_schema(self) -> None:
        frame = demo_frame()
        summary, issues = validate_model_frame(frame)
        self.assertTrue(summary["train_ready"])
        self.assertEqual(summary["rows"], 60)
        self.assertEqual(summary["date_groups"], 12)
        self.assertFalse(issues)

    def test_incomplete_rows_are_audited_but_do_not_block_enough_valid_data(self) -> None:
        frame = demo_frame()
        frame.loc[0, "异养菌最大OUR"] = np.nan
        summary, issues = validate_model_frame(frame)
        self.assertTrue(summary["train_ready"])
        self.assertEqual(summary["valid_rows"], 59)
        self.assertEqual(summary["excluded_rows"], 1)
        self.assertTrue(any(issue.severity == "warning" for issue in issues))

    def test_compat_group_split_never_leaks_a_date(self) -> None:
        groups = np.repeat(["d1", "d2", "d3", "d4", "d5", "d6"], 3)
        splitter = ShuffledGroupKFoldCompat(n_splits=3, random_state=42)
        seen_valid: set[int] = set()
        for train_index, valid_index in splitter.split(
            np.zeros((len(groups), 1)), groups=groups
        ):
            self.assertTrue(
                set(groups[train_index]).isdisjoint(set(groups[valid_index]))
            )
            seen_valid.update(valid_index.tolist())
        self.assertEqual(seen_valid, set(range(len(groups))))

    def test_calibration_selects_three_distinct_aeration_levels(self) -> None:
        frame = pd.DataFrame(
            {"曝气量(L/min)": [8.0, 8.2, 8.2, 8.5, 9.0]},
            index=[10, 11, 12, 13, 14],
        )
        selected = select_calibration_indices(frame, calibration_points=3)
        self.assertEqual(selected, [10, 13, 14])

    def test_residual_calibrator_recovers_constant_day_bias(self) -> None:
        calibrator = fit_residual_calibrator(
            np.array([8.0, 8.5, 9.0]),
            np.array([0.08, 0.08, 0.08]),
        )
        correction = residual_correction(calibrator, np.array([8.25, 8.75]))
        np.testing.assert_allclose(correction, [0.08, 0.08], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
