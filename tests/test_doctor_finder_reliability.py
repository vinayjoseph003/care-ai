# ==============================================================================
#  tests/test_doctor_finder_reliability.py
#
#  Offline, deterministic test suite for Doctor Finder reliability & error handling.
#  Mocks urllib.request.urlopen — absolutely NO network calls.
# ==============================================================================

import io
import json
import socket
import sys
import unittest
import urllib.error
import urllib.parse
import urllib.request
from unittest.mock import MagicMock, patch

# Ensure offline mock-only tests can run without optional heavy dependencies
for _mod in ("scipy", "scipy.sparse"):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = MagicMock()

from core.doctor_finder import (
    _query_overpass,
    find_nearby_doctors,
)
try:
    from core.doctor_finder import OverpassUnavailable
except ImportError:
    OverpassUnavailable = None

import ui.doctor_finder_panel as df_panel


def _make_http_response(status_code: int, payload: dict) -> MagicMock:
    """Create a mock HTTP response context manager."""
    resp = MagicMock()
    body_bytes = json.dumps(payload).encode("utf-8")
    resp.read.return_value = body_bytes
    resp.status = status_code
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


class TestDoctorFinderReliability(unittest.TestCase):

    # ── STEP 1 (a): Headers and Request format ─────────────────────────

    @patch("urllib.request.urlopen")
    def test_request_headers_and_body(self, mock_urlopen):
        """
        STEP 1 (a):
        The Request carries User-Agent 'CareAI-student-project/1.0 (+https://github.com/vinayjoseph003/care-ai)',
        Accept: 'application/json', and a URL-encoded form body.
        """
        mock_urlopen.return_value = _make_http_response(200, {"elements": []})

        query = "[out:json];node(around:5000,12.97,77.59);out;"
        _query_overpass(query)

        self.assertTrue(mock_urlopen.called, "urllib.request.urlopen must be called")
        req = mock_urlopen.call_args[0][0]
        self.assertIsInstance(req, urllib.request.Request)

        # Check User-Agent
        ua = req.get_header("User-agent")
        self.assertEqual(
            ua,
            "CareAI-student-project/1.0 (+https://github.com/vinayjoseph003/care-ai)",
            "User-Agent header must match OSM wiki requirement",
        )

        # Check Accept: application/json
        accept = req.get_header("Accept")
        self.assertEqual(accept, "application/json", "Accept header must be application/json")

        # Check URL-encoded form body
        self.assertIsNotNone(req.data, "Request must have a body")
        parsed_body = urllib.parse.parse_qs(req.data.decode("utf-8"))
        self.assertIn("data", parsed_body, "Body must be URL-encoded form containing 'data'")
        self.assertEqual(parsed_body["data"][0], query)

    # ── STEP 1 (b): Retry on 502 then 200 ──────────────────────────────

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_retry_on_502_then_success(self, mock_urlopen, mock_sleep):
        """
        STEP 1 (b):
        HTTP 502 then 200 -> returns results (patch time.sleep).
        """
        err_502 = urllib.error.HTTPError(
            url="https://overpass-api.de/api/interpreter",
            code=502,
            msg="Bad Gateway",
            hdrs={},
            fp=io.BytesIO(b"Bad Gateway"),
        )
        success_resp = _make_http_response(
            200,
            {
                "elements": [
                    {
                        "type": "node",
                        "id": 12345,
                        "lat": 12.971,
                        "lon": 77.591,
                        "tags": {
                            "name": "Apollo Clinic",
                            "amenity": "doctors",
                        },
                    }
                ]
            },
        )
        calls = [err_502, success_resp]

        def fake_urlopen(req, *args, **kwargs):
            if calls:
                item = calls.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item
            return _make_http_response(200, {"elements": []})

        mock_urlopen.side_effect = fake_urlopen

        result = find_nearby_doctors("General Practitioner", 12.97, 77.59)

        # Must have slept with backoff (2s on first retry)
        mock_sleep.assert_any_call(2)

        # Must have returned the facility
        self.assertIsNone(result.get("error_kind"))
        self.assertIsNone(result.get("error"))
        self.assertEqual(len(result.get("results", [])), 1)
        self.assertEqual(result["results"][0]["name"], "Apollo Clinic")

    # ── STEP 1 (c): Service unavailable on persistent 502/503/504 or 406/429 ──

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_service_unavailable_retries_exhausted_502(self, mock_urlopen, mock_sleep):
        """
        STEP 1 (c):
        502 on every attempt (at most 2 retries with 2s, 4s backoff) ->
        find_nearby_doctors returns error_kind 'service_unavailable' (distinct from empty),
        and UI shows busy message with existing links.
        """
        err_502 = urllib.error.HTTPError(
            url="https://overpass-api.de/api/interpreter",
            code=502,
            msg="Bad Gateway",
            hdrs={},
            fp=io.BytesIO(b"Bad Gateway"),
        )
        # Always fail with 502
        mock_urlopen.side_effect = err_502

        result = find_nearby_doctors("Cardiologist", 12.97, 77.59)

        # Must return error_kind "service_unavailable" distinct from empty
        self.assertEqual(result.get("error_kind"), "service_unavailable")
        self.assertEqual(result.get("results"), [])
        self.assertIn("busy right now", result.get("error", "").lower())

        # Check retry delays: initial attempt + retry 1 (sleep 2) + retry 2 (sleep 4) = 3 attempts total
        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list if c[0][0] in (2, 4)]
        self.assertEqual(sleep_calls, [2, 4], "Must retry at most twice with 2s and 4s backoff")

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_service_unavailable_no_retry_on_406_and_429(self, mock_urlopen, mock_sleep):
        """
        STEP 1 (c):
        406 or 429 on the first attempt -> NO retry, find_nearby_doctors returns
        error_kind 'service_unavailable'.
        """
        for code, msg in [(406, "Not Acceptable"), (429, "Too Many Requests")]:
            mock_urlopen.reset_mock()
            mock_sleep.reset_mock()

            err = urllib.error.HTTPError(
                url="https://overpass-api.de/api/interpreter",
                code=code,
                msg=msg,
                hdrs={},
                fp=io.BytesIO(msg.encode()),
            )
            mock_urlopen.side_effect = err

            result = find_nearby_doctors("Cardiologist", 12.97, 77.59)

            self.assertEqual(
                result.get("error_kind"),
                "service_unavailable",
                f"HTTP {code} must return error_kind 'service_unavailable'",
            )
            self.assertEqual(result.get("results"), [])

            # Must NOT retry (no 2s or 4s sleep)
            backoff_sleeps = [c[0][0] for c in mock_sleep.call_args_list if c[0][0] in (2, 4)]
            self.assertEqual(backoff_sleeps, [], f"HTTP {code} must NOT trigger backoff retries")
            self.assertEqual(mock_urlopen.call_count, 1, f"HTTP {code} must only be attempted once")

    def test_ui_service_unavailable_shows_busy_message_and_links(self):
        """
        STEP 1 (c):
        When error_kind is 'service_unavailable', UI shows
        'The map service is busy right now. Try again in a minute, or search on Google Maps/Practo'
        with the existing links.
        """
        import streamlit as st

        mock_bot = MagicMock()
        response = {"triage": "SEE_DOCTOR", "specialist": "Cardiologist"}

        fake_state = {
            "df_css_injected": True,
            "df_state": "done",
            "df_results": {
                "specialist": "Cardiologist",
                "results": [],
                "radius_used_km": 5,
                "error_kind": "service_unavailable",
                "error": "The map service is busy right now. Try again in a minute, or search on Google Maps/Practo",
            },
        }

        with patch.object(df_panel, "_HAS_JS_EVAL", True), \
             patch.object(st, "session_state", fake_state), \
             patch.object(st, "markdown") as mock_markdown, \
             patch.object(st, "button", return_value=False):

            df_panel.render_doctor_finder(mock_bot, response)

            markdown_texts = [call[0][0] for call in mock_markdown.call_args_list]
            all_rendered = "\n".join(markdown_texts)

            # Check busy message
            expected_msg = "The map service is busy right now. Try again in a minute, or search on Google Maps/Practo"
            self.assertIn(
                expected_msg,
                all_rendered,
                "UI must show 'The map service is busy right now...' message",
            )

            # Check existing links are shown
            self.assertIn("practo.com", all_rendered.lower())
            self.assertIn("google.com/maps", all_rendered.lower())

    # ── STEP 1 (d): HTTP 200 with zero elements ────────────────────────

    @patch("urllib.request.urlopen")
    def test_empty_results_on_200_returns_no_results(self, mock_urlopen):
        """
        STEP 1 (d):
        HTTP 200 with zero elements -> existing 'No nearby facilities found'
        message with error_kind 'no_results'.
        """
        mock_urlopen.return_value = _make_http_response(200, {"elements": []})

        result = find_nearby_doctors("Cardiologist", 12.97, 77.59)

        self.assertEqual(result.get("error_kind"), "no_results")
        self.assertEqual(result.get("results"), [])
        self.assertIn("no nearby facilities found", result.get("error", "").lower())

    # ── STEP 1 (a): Timeout, socket.timeout, URLError, non-JSON ───────

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_exceptions_and_invalid_json_yield_service_unavailable(self, mock_urlopen, mock_sleep):
        """
        STEP 1 (a):
        urlopen raising TimeoutError, socket.timeout, URLError, and a 200 response
        whose body is not JSON each yield error_kind 'service_unavailable'
        (no exception escapes find_nearby_doctors).
        Also asserts OverpassUnavailable custom exception is raised by _query_overpass.
        """
        self.assertIsNotNone(
            OverpassUnavailable,
            "OverpassUnavailable custom exception must be defined in core.doctor_finder",
        )

        non_json_resp = MagicMock()
        non_json_resp.read.return_value = b"<html>502 Bad Gateway</html>"
        non_json_resp.status = 200
        non_json_resp.__enter__.return_value = non_json_resp
        non_json_resp.__exit__.return_value = False

        cases = [
            ("TimeoutError", TimeoutError("Request timed out")),
            ("socket.timeout", socket.timeout("Socket timed out")),
            ("URLError", urllib.error.URLError("DNS resolution failed")),
            ("Non-JSON 200", non_json_resp),
        ]

        for name, failure_trigger in cases:
            with self.subTest(case=name):
                mock_urlopen.reset_mock()
                mock_sleep.reset_mock()
                if isinstance(failure_trigger, Exception):
                    mock_urlopen.side_effect = failure_trigger
                else:
                    mock_urlopen.side_effect = None
                    mock_urlopen.return_value = failure_trigger

                # 1. find_nearby_doctors must not let any exception escape and return service_unavailable
                try:
                    result = find_nearby_doctors("Cardiologist", 12.97, 77.59)
                except Exception as ex:
                    self.fail(f"find_nearby_doctors let exception escape for {name}: {ex}")

                self.assertEqual(
                    result.get("error_kind"),
                    "service_unavailable",
                    f"{name} must yield error_kind 'service_unavailable'",
                )
                self.assertEqual(result.get("results"), [])

                # 2. _query_overpass must convert the non-success into OverpassUnavailable
                with self.assertRaises(OverpassUnavailable, msg=f"_query_overpass must raise OverpassUnavailable on {name}"):
                    _query_overpass("[out:json];node(1);out;")

    # ── STEP 1 (b): Specialist Normalization & Fallback ───────────────

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_specialist_normalization_and_fallback(self, mock_urlopen, mock_sleep):
        """
        STEP 1 (b):
        Specialist normalization:
        - 'General Practitioner / GP', 'GP', 'gp', 'General Physician' -> doctors+healthcare tags
        - 'Cardiologist (heart)' behaves as today (cardiology speciality)
        - Unmapped specialist falls back to doctors + clinic + hospital tags (not hospital only)
        - Displayed specialist label is not altered
        """
        def capture_queries(specialist_str):
            queries = []
            mock_urlopen.reset_mock()

            def fake_urlopen(req, *args, **kwargs):
                body = urllib.parse.parse_qs(req.data.decode("utf-8"))
                queries.append(body.get("data", [""])[0])
                return _make_http_response(200, {"elements": []})

            mock_urlopen.side_effect = fake_urlopen
            res = find_nearby_doctors(specialist_str, 12.97, 77.59)
            return res, queries

        gp_specialists = ["General Practitioner / GP", "GP", "gp", "General Physician"]
        for spec in gp_specialists:
            with self.subTest(specialist=spec):
                res, queries = capture_queries(spec)
                all_queries = "\n".join(queries)
                self.assertIn('"amenity"="doctors"', all_queries, f"{spec} must query amenity=doctors")
                self.assertIn('"healthcare"="doctor"', all_queries, f"{spec} must query healthcare=doctor")

        # 'Cardiologist (heart)' behaves as today
        res_cardio, queries_cardio = capture_queries("Cardiologist (heart)")
        all_cardio = "\n".join(queries_cardio)
        self.assertIn('"healthcare:speciality"="cardiology"', all_cardio)
        self.assertEqual(res_cardio.get("specialist"), "Cardiologist", "Displayed label should retain cleaned base")

        # Unmapped specialist falls back to doctors + clinic + hospital tags (not hospital only)
        res_unmapped, queries_unmapped = capture_queries("Holistic Quantum Healer")
        all_unmapped = "\n".join(queries_unmapped)
        self.assertIn('"amenity"="doctors"', all_unmapped, "Unmapped specialist must fall back to doctors")
        self.assertIn('"amenity"="clinic"', all_unmapped, "Unmapped specialist must fall back to clinic")
        self.assertIn('"amenity"="hospital"', all_unmapped, "Unmapped specialist must fall back to hospital")

    def test_ui_no_duplicate_manual_links(self):
        """
        Verify that _manual_links is not rendered more than once in any outcome state.
        """
        import streamlit as st
        mock_bot = MagicMock()
        response = {"triage": "SEE_DOCTOR", "specialist": "Cardiologist"}

        for outcome, state_res in [
            ("service_unavailable", {"specialist": "Cardiologist", "results": [], "error_kind": "service_unavailable", "error": "busy"}),
            ("no_results", {"specialist": "Cardiologist", "results": [], "error_kind": "no_results", "error": "none"}),
            ("results", {"specialist": "Cardiologist", "results": [{"name": "Clinic", "facility_type": "Clinic", "distance_km": 1.0, "distance_str": "1.0 km", "lat": 12.9, "lon": 77.5, "osm_id": 1}], "error_kind": None, "error": None}),
        ]:
            with self.subTest(outcome=outcome):
                fake_state = {
                    "df_css_injected": True,
                    "df_state": "done",
                    "df_results": state_res,
                }
                with patch.object(df_panel, "_HAS_JS_EVAL", True), \
                     patch.object(st, "session_state", fake_state), \
                     patch.object(st, "markdown") as mock_markdown, \
                     patch.object(st, "button", return_value=False):

                    df_panel.render_doctor_finder(mock_bot, response)

                    markdown_texts = [call[0][0] for call in mock_markdown.call_args_list]
                    manual_link_count = sum(1 for text in markdown_texts if "df-manual-links" in text)
                    self.assertLessEqual(
                        manual_link_count,
                        1,
                        f"Manual links must not be rendered more than once in {outcome} state",
                    )


if __name__ == "__main__":
    unittest.main()
