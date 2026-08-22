from __future__ import annotations

import json
import unittest

from wastewater_snd import model_v4
from wastewater_snd.demo_runtime import build_demo_runtime, make_condition, predict_condition
from wastewater_snd.experience import (
    build_decision_report_payload,
    build_scenario_presets,
    decision_report_json,
    decision_report_markdown,
)


class GuidedExperienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = build_demo_runtime()
        cls.presets = build_scenario_presets(cls.runtime.data)

    def test_four_presets_cover_normal_risk_and_mixing_boundaries(self) -> None:
        self.assertEqual(len(self.presets), 4)
        self.assertEqual(
            {preset.key for preset in self.presets},
            {
                "stable",
                "high_tn_risk",
                "near_mixing_floor",
                "below_mixing_floor",
            },
        )
        for preset in self.presets:
            self.assertEqual(set(preset.values), set(model_v4.RAW_MODEL_INPUTS))
            self.assertEqual(preset.tn_standard, 15.0)

        below = next(p for p in self.presets if p.key == "below_mixing_floor")
        self.assertLess(
            below.values[model_v4.AERATION_COL],
            below.minimum_safe_aeration,
        )

    def test_report_contains_current_result_but_not_training_rows(self) -> None:
        preset = next(p for p in self.presets if p.key == "below_mixing_floor")
        condition = make_condition(preset.values)
        prediction, recommendation = predict_condition(
            self.runtime,
            condition,
            tn_standard=preset.tn_standard,
            minimum_safe_aeration=preset.minimum_safe_aeration,
        )
        payload = build_decision_report_payload(
            condition=condition,
            prediction=prediction,
            recommendation=recommendation,
            recommended_aeration=preset.minimum_safe_aeration,
            scenario_label=preset.label,
            tn_standard=preset.tn_standard,
            minimum_safe_aeration=preset.minimum_safe_aeration,
            is_synthetic=True,
            generated_at_utc="2026-08-22T12:00:00+00:00",
        )
        markdown = decision_report_markdown(payload)
        structured = json.loads(decision_report_json(payload))

        self.assertIn("# OUR-SND 曝气决策报告", markdown)
        self.assertIn("低于安全下限", markdown)
        self.assertIn("不包含训练数据", markdown)
        self.assertEqual(structured["data_mode"], "public_synthetic")
        self.assertFalse(structured["training_rows_included"])
        self.assertFalse(structured["constraints"]["mbr_scour_air_included"])

    def test_report_rejects_multiple_conditions(self) -> None:
        preset = self.presets[0]
        condition = make_condition(preset.values)
        with self.assertRaisesRegex(ValueError, "只能包含一条工况"):
            build_decision_report_payload(
                condition=condition.loc[[0, 0]],
                prediction={
                    model_v4.REMOVAL_COL: 0.6,
                    model_v4.SND_COL: 0.5,
                    model_v4.EFFLUENT_PROXY_COL: 10.0,
                },
                recommendation="test",
                recommended_aeration=3.0,
                scenario_label=preset.label,
                tn_standard=15.0,
                minimum_safe_aeration=2.0,
                is_synthetic=True,
            )


if __name__ == "__main__":
    unittest.main()
