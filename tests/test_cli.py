from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import joblib

from wastewater_snd import cli, model_v4
from wastewater_snd.demo_runtime import build_demo_runtime


class CliTests(unittest.TestCase):
    def test_parser_exposes_mixing_floor_for_train_and_predict(self) -> None:
        parser = cli.build_parser()
        predict_args = parser.parse_args(
            [
                "predict",
                "--model",
                "model.joblib",
                "--condition",
                "condition.json",
                "--minimum-safe-aeration",
                "3.8",
            ]
        )
        train_args = parser.parse_args(
            [
                "train",
                "--data",
                "data.csv",
                "--minimum-safe-aeration",
                "3.8",
            ]
        )
        self.assertEqual(predict_args.minimum_safe_aeration, 3.8)
        self.assertEqual(train_args.minimum_safe_aeration, 3.8)

    def test_schema_demo_generation_and_validation_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "demo.csv"
            output = io.StringIO()
            with redirect_stdout(output):
                cli.main(["schema"])
                cli.main(["demo-data", "--output", str(data_path), "--seed", "7"])
                cli.main(["validate", "--data", str(data_path)])
            self.assertTrue(data_path.exists())
            self.assertIn(model_v4.AERATION_COL, output.getvalue())
            self.assertIn('"train_ready": true', output.getvalue())

    def test_predict_command_applies_configured_mixing_floor(self) -> None:
        runtime = build_demo_runtime()
        row = runtime.data.iloc[len(runtime.data) // 2]
        historical_low = float(runtime.data[model_v4.AERATION_COL].min())
        historical_high = float(runtime.data[model_v4.AERATION_COL].max())
        floor = historical_low + 0.3 * (historical_high - historical_low)
        condition = {
            model_v4.DATE_COL: "cli-test",
            **{column: float(row[column]) for column in model_v4.RAW_MODEL_INPUTS},
        }
        condition[model_v4.AERATION_COL] = floor - 0.1
        bundle = {
            "selected_specs": runtime.selected,
            "models": runtime.trained,
            "training_data": runtime.data,
            "strict_oof_abs_error_q90": runtime.error_q90,
            "support": runtime.support,
            "aeration_optimization_gate": runtime.gate,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.joblib"
            condition_path = root / "condition.json"
            joblib.dump(bundle, model_path)
            condition_path.write_text(
                json.dumps(condition, ensure_ascii=False),
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                cli.main(
                    [
                        "predict",
                        "--model",
                        str(model_path),
                        "--condition",
                        str(condition_path),
                        "--minimum-safe-aeration",
                        str(floor),
                    ]
                )
        self.assertIn("B级—混合安全下限纠偏推荐", output.getvalue())

    def test_main_returns_nonzero_for_missing_training_file(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as context:
            cli.main(["train", "--data", "does-not-exist.csv"])
        self.assertEqual(context.exception.code, 1)
        self.assertIn("未找到训练 CSV", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
