from __future__ import annotations

import unittest

import pandas as pd

from wastewater_snd import model_v4
from wastewater_snd.calibrated import (
    evaluate_rolling_calibration,
    evaluate_three_point_calibration,
    fit_calibrators_for_new_date,
    fixed_calibrated_base_specs,
    predict_calibrated_condition,
)
from wastewater_snd.synthetic import demo_frame


class CalibratedWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data, _ = model_v4.clean_model_frame(demo_frame(), encoding="unit-test")
        cls.selected = fixed_calibrated_base_specs()

    def test_three_point_and_rolling_validations_produce_blind_predictions(self) -> None:
        logo_summary, logo_rows, logo_q90 = evaluate_three_point_calibration(
            self.data,
            self.selected,
        )
        rolling_summary, rolling_rows, rolling_q90 = evaluate_rolling_calibration(
            self.data,
            self.selected,
        )

        self.assertEqual(set(logo_summary["目标"]), set(model_v4.TARGETS))
        self.assertEqual(set(rolling_summary["目标"]), set(model_v4.TARGETS))
        self.assertGreater((logo_rows["记录用途"] == "盲测验证").sum(), 0)
        self.assertGreater((rolling_rows["记录用途"] == "向前盲测验证").sum(), 0)
        self.assertTrue(all(value >= 0 for value in logo_q90.values()))
        self.assertTrue(all(value >= 0 for value in rolling_q90.values()))

    def test_new_date_calibration_is_limited_to_measured_aeration_range(self) -> None:
        selected = {
            target: next(
                spec
                for spec in model_v4.candidate_models(target)
                if spec.name
                == {
                    model_v4.REMOVAL_COL: "PLS2_原工程特征_记录级",
                    model_v4.SND_COL: "Ridge10_原基础特征_记录级",
                }[target]
            )
            for target in model_v4.TARGETS
        }
        models = {
            target: model_v4.fit_candidate(spec, self.data)
            for target, spec in selected.items()
        }
        calibration = self.data[self.data[model_v4.DATE_COL].eq(
            self.data.iloc[0][model_v4.DATE_COL]
        )].copy()
        bundle = {
            "models": models,
            "same_day_calibration": {
                "calibration_points": 3,
                "slope_ridge_alpha": 1.0,
            },
            "calibrated_abs_error_q90": {
                model_v4.REMOVAL_COL: 0.04,
                model_v4.SND_COL: 0.05,
            },
        }

        calibrators = fit_calibrators_for_new_date(bundle, calibration)
        self.assertEqual(set(calibrators), set(model_v4.TARGETS))
        condition = calibration.iloc[[len(calibration) // 2]].copy()
        prediction = predict_calibrated_condition(bundle, calibration, condition)
        self.assertIn("出水TN_推导值(mg/L)", prediction)
        self.assertIsNotNone(
            prediction[model_v4.REMOVAL_COL]["保守范围"]
        )

        outside = condition.copy()
        outside.loc[outside.index[0], model_v4.AERATION_COL] = (
            float(calibration[model_v4.AERATION_COL].max()) + 1.0
        )
        with self.assertRaisesRegex(ValueError, "超出当日校准范围"):
            predict_calibrated_condition(bundle, calibration, outside)

    def test_calibration_rejects_too_few_levels(self) -> None:
        frame = pd.DataFrame({model_v4.AERATION_COL: [4.0, 4.0, 4.0]})
        with self.assertRaisesRegex(ValueError, "不同曝气水平"):
            fit_calibrators_for_new_date({"models": {}}, frame)


if __name__ == "__main__":
    unittest.main()
