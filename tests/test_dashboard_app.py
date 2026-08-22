from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from streamlit.testing.v1 import AppTest
except ModuleNotFoundError:  # The web dependency is optional for core-only installs.
    AppTest = None

from wastewater_snd.synthetic import demo_frame

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


@unittest.skipIf(AppTest is None, "Streamlit web extra is not installed")
class DashboardAppTests(unittest.TestCase):
    def _run(self, *, mode: str, import_switch: str):
        environment = {
            "SND_APP_MODE": mode,
            "SND_LOCAL_IMPORT": import_switch,
        }
        with patch.dict(os.environ, environment, clear=False):
            return AppTest.from_file(str(APP_PATH)).run(timeout=60)

    def test_public_app_has_no_file_uploader(self) -> None:
        app = self._run(mode="public", import_switch="0")
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("file_uploader")), 0)
        self.assertTrue(any("文件上传已关闭" in item.value for item in app.info))

    def test_incomplete_local_config_falls_back_to_public(self) -> None:
        app = self._run(mode="local", import_switch="0")
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("file_uploader")), 0)

    def test_local_app_has_exactly_one_file_uploader(self) -> None:
        environment = {
            "SND_APP_MODE": "local",
            "SND_LOCAL_IMPORT": "1",
        }
        with patch.dict(os.environ, environment, clear=False):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
            self.assertEqual(len(app.exception), 0)
            uploaders = app.get("file_uploader")
            self.assertEqual(len(uploaders), 1)
            self.assertEqual(uploaders[0].label, "选择标准 CSV")

            payload = demo_frame().to_csv(index=False).encode("utf-8-sig")
            app = (
                uploaders[0]
                .upload("synthetic-contract-example.csv", payload, "text/csv")
                .run(timeout=60)
            )
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("已在本机加载" in item.value for item in app.success))


if __name__ == "__main__":
    unittest.main()
