import os
import unittest

from clef_demo.modal_app import _resolve_streamlit_script_path


class DemoModalAppTests(unittest.TestCase):
    def test_resolve_streamlit_script_path_points_to_existing_file(self):
        script_path = _resolve_streamlit_script_path()
        self.assertTrue(script_path.endswith("streamlit_app.py"))
        self.assertTrue(os.path.exists(script_path))


if __name__ == "__main__":
    unittest.main()
