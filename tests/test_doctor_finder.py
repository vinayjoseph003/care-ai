# ==============================================================================
#  tests/test_doctor_finder.py — Unit tests for Doctor Finder privacy & config
# ==============================================================================

import importlib.util
import os
import sys
import unittest

try:
    from core.doctor_finder import OVERPASS_ENDPOINTS
except ModuleNotFoundError:
    df_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core", "doctor_finder.py"))
    spec = importlib.util.spec_from_file_location("core.doctor_finder", df_path)
    df_mod = importlib.util.module_from_spec(spec)
    sys.modules["core.doctor_finder"] = df_mod
    spec.loader.exec_module(df_mod)
    OVERPASS_ENDPOINTS = df_mod.OVERPASS_ENDPOINTS


class TestDoctorFinderEndpoints(unittest.TestCase):
    def test_no_mail_ru_endpoint(self):
        """Assert no endpoint in OVERPASS_ENDPOINTS contains 'mail.ru'."""
        for endpoint in OVERPASS_ENDPOINTS:
            self.assertNotIn(
                "mail.ru",
                endpoint.lower(),
                f"Forbidden endpoint detected: {endpoint}"
            )


if __name__ == "__main__":
    unittest.main()
