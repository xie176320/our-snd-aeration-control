from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wastewater_snd.ablation import evaluate_realtime_our_ablation
from wastewater_snd.schema import A_LIVE_COL, DATE_COL, N_LIVE_COL
from wastewater_snd.synthetic import demo_frame


class AblationTests(unittest.TestCase):
    def test_realtime_our_ablation_runs_with_grouped_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "demo.csv"
            output_path = root / "ablation.csv"
            demo_frame().to_csv(data_path, index=False, encoding="utf-8-sig")
            result = evaluate_realtime_our_ablation(
                data_path,
                output_path,
                repeats=2,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(len(result), 24)
            self.assertEqual(result["目标"].nunique(), 2)
            self.assertEqual(result["模型"].nunique(), 2)
            self.assertEqual(result["特征组"].nunique(), 6)

    def test_ablation_rejects_missing_live_our_and_too_few_dates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "ablation.csv"
            missing_path = root / "missing.csv"
            demo_frame().drop(columns=[A_LIVE_COL, N_LIVE_COL]).to_csv(
                missing_path,
                index=False,
            )
            with self.assertRaisesRegex(ValueError, "实时 OUR 消融缺少字段"):
                evaluate_realtime_our_ablation(missing_path, output_path)

            few_dates_path = root / "few-dates.csv"
            frame = demo_frame()
            dates = frame[DATE_COL].unique()[:4]
            frame[frame[DATE_COL].isin(dates)].to_csv(few_dates_path, index=False)
            with self.assertRaisesRegex(ValueError, "有效日期少于 5 个"):
                evaluate_realtime_our_ablation(few_dates_path, output_path)


if __name__ == "__main__":
    unittest.main()
