from __future__ import annotations

import re
import unittest
from wastewater_snd import model_v4
from wastewater_snd.demo_runtime import (
    aeration_response_curve,
    build_demo_runtime,
    make_condition,
    predict_condition,
)


class DemoRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = build_demo_runtime()
        row = cls.runtime.data.iloc[len(cls.runtime.data) // 2]
        cls.condition = make_condition(
            {
                model_v4.DATE_COL: "unit-test",
                **{column: row[column] for column in model_v4.RAW_MODEL_INPUTS},
            }
        )

    def test_prediction_is_bounded_and_recommendation_is_numeric(self) -> None:
        prediction, recommendation = predict_condition(self.runtime, self.condition)
        self.assertGreaterEqual(prediction[model_v4.REMOVAL_COL], 0.0)
        self.assertLessEqual(prediction[model_v4.REMOVAL_COL], 1.0)
        self.assertGreaterEqual(prediction[model_v4.SND_COL], 0.0)
        self.assertLessEqual(prediction[model_v4.SND_COL], 1.0)
        self.assertRegex(recommendation, r"本次建议曝气量：\d+\.\d{2} L/min")

    def test_response_curve_stays_inside_training_range(self) -> None:
        curve = aeration_response_curve(self.runtime, self.condition, points=9)
        self.assertEqual(len(curve), 9)
        self.assertAlmostEqual(
            curve[model_v4.AERATION_COL].min(),
            self.runtime.data[model_v4.AERATION_COL].min(),
        )
        self.assertAlmostEqual(
            curve[model_v4.AERATION_COL].max(),
            self.runtime.data[model_v4.AERATION_COL].max(),
        )

    def test_invalid_condition_is_rejected(self) -> None:
        values = {
            column: 1.0 for column in model_v4.RAW_MODEL_INPUTS
        }
        values[model_v4.H_MAX_COL] = -1.0
        with self.assertRaisesRegex(ValueError, re.escape("OUR 与进水 COD 不能为负数")):
            make_condition(values)


if __name__ == "__main__":
    unittest.main()
