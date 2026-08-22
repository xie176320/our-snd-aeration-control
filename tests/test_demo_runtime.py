from __future__ import annotations

import re
import unittest

from wastewater_snd import model_v4
from wastewater_snd.dashboard import (
    LOCAL_MODE,
    PUBLIC_MODE,
    local_import_enabled,
    resolve_app_mode,
)
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

    def test_recommendation_cannot_cross_configured_mixing_floor(self) -> None:
        historical_low = float(self.runtime.data[model_v4.AERATION_COL].min())
        historical_high = float(self.runtime.data[model_v4.AERATION_COL].max())
        floor = historical_low + 0.35 * (historical_high - historical_low)
        condition = self.condition.copy()
        condition.loc[condition.index[0], model_v4.AERATION_COL] = floor - 0.1

        _, recommendation = predict_condition(
            self.runtime,
            condition,
            minimum_safe_aeration=floor,
        )

        match = re.search(r"本次建议曝气量：([0-9.]+) L/min", recommendation)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(float(match.group(1)), floor - 0.01)
        self.assertIn("B级—混合安全下限纠偏推荐", recommendation)
        self.assertIn("不包含 MBR 膜擦洗风量", recommendation)

    def test_response_curve_starts_at_configured_mixing_floor(self) -> None:
        historical_low = float(self.runtime.data[model_v4.AERATION_COL].min())
        historical_high = float(self.runtime.data[model_v4.AERATION_COL].max())
        floor = historical_low + 0.25 * (historical_high - historical_low)
        curve = aeration_response_curve(
            self.runtime,
            self.condition,
            points=7,
            minimum_safe_aeration=floor,
        )
        self.assertAlmostEqual(curve[model_v4.AERATION_COL].min(), floor)
        self.assertAlmostEqual(curve[model_v4.AERATION_COL].max(), historical_high)

    def test_invalid_mixing_floor_is_rejected(self) -> None:
        historical_high = float(self.runtime.data[model_v4.AERATION_COL].max())
        with self.assertRaisesRegex(ValueError, "必须是大于 0 的有限数字"):
            predict_condition(
                self.runtime,
                self.condition,
                minimum_safe_aeration=0.0,
            )
        with self.assertRaisesRegex(ValueError, "高于训练数据曝气上限"):
            predict_condition(
                self.runtime,
                self.condition,
                minimum_safe_aeration=historical_high + 1.0,
            )

    def test_invalid_condition_is_rejected(self) -> None:
        values = {column: 1.0 for column in model_v4.RAW_MODEL_INPUTS}
        values[model_v4.H_MAX_COL] = -1.0
        with self.assertRaisesRegex(ValueError, re.escape("OUR 与进水 COD 不能为负数")):
            make_condition(values)

    def test_runtime_accepts_an_in_memory_local_frame(self) -> None:
        frame = self.runtime.data.drop(columns=[model_v4.EFFLUENT_PROXY_COL], errors="ignore")
        local_runtime = build_demo_runtime(frame)
        self.assertEqual(local_runtime.info["encoding"], "in-memory-local-import")
        self.assertEqual(local_runtime.info["clean_rows"], self.runtime.info["clean_rows"])


class DashboardModeTests(unittest.TestCase):
    def test_public_is_the_secure_default(self) -> None:
        self.assertEqual(resolve_app_mode({}), PUBLIC_MODE)
        self.assertEqual(resolve_app_mode({"SND_APP_MODE": "unexpected"}), PUBLIC_MODE)
        self.assertFalse(local_import_enabled({}))

    def test_one_switch_cannot_enable_import(self) -> None:
        self.assertFalse(local_import_enabled({"SND_APP_MODE": LOCAL_MODE}))
        self.assertFalse(local_import_enabled({"SND_LOCAL_IMPORT": "1"}))
        self.assertFalse(
            local_import_enabled({"SND_APP_MODE": PUBLIC_MODE, "SND_LOCAL_IMPORT": "1"})
        )

    def test_explicit_local_mode_enables_import(self) -> None:
        self.assertTrue(
            local_import_enabled({"SND_APP_MODE": " LOCAL ", "SND_LOCAL_IMPORT": "TRUE"})
        )


if __name__ == "__main__":
    unittest.main()
