# SPDX-License-Identifier: LGPL-2.1-or-later
"""Portable parser/admission regressions; synthetic records are not optimizer results."""
import copy
import json
import pathlib
import unittest
import compare_readme as comparison

ROOT = pathlib.Path(__file__).resolve().parent


def protocol():
    return json.loads((ROOT / "readme_protocol.json").read_text())


def record(p, value=1.0):
    timing = p["timing_reliability"]
    def block(seconds):
        return {"sample_times_ns": [1000.0], "stable_outputs": True,
                "validated_outputs": 1, "elapsed_seconds": seconds}
    rows = []
    for name, n in comparison.CASE_DIMENSIONS.items():
        result = {"params": [value] * n, "errors": [1.0] * n,
                  "covariance": [float(i == j) for i in range(n) for j in range(n)],
                  "fval": 1.0, "edm": 1e-8, "nfcn": 10, "actual_calls": 10,
                  "is_valid": True, "has_covariance": True,
                  "above_max_edm": False, "reached_call_limit": False}
        rows.append({"name": name, "samples": p["samples"], "evals": 1,
                     "stable_outputs": True, "sample_times_ns": [1000.0] * p["samples"],
                     "continuous_warmup": block(timing["fixed_warmup_seconds"]),
                     "timing_blocks": [block(timing["block_min_seconds"])
                                       for _ in range(timing["blocks"])], "result": result})
    return {"rows": rows, "timing_reliability": {
        key: timing[key] for key in ("fixed_warmup_seconds", "blocks", "block_min_seconds")}}


def summarize(p, records):
    return comparison.summarize([comparison.compare_pair(j, c, p) for j, c in records], p)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.p = protocol()
        self.records = [(record(self.p), record(self.p)) for _ in range(self.p["rounds"])]

    def admitted(self, records=None, p=None):
        return all(r["benchmark_admitted"] for r in
                   summarize(p or self.p, self.records if records is None else records)["cases"].values())

    def test_complete_positive_control(self):
        self.assertTrue(self.admitted())

    def test_changed_warmup_duration_rejects_short_records(self):
        p = copy.deepcopy(self.p)
        p["timing_reliability"]["fixed_warmup_seconds"] = 2.0
        self.assertFalse(self.admitted(p=p))

    def test_changed_block_duration_rejects_short_records(self):
        p = copy.deepcopy(self.p)
        p["timing_reliability"]["block_min_seconds"] = 2.0
        self.assertFalse(self.admitted(p=p))

    def test_changed_block_count_rejects_missing_block(self):
        p = copy.deepcopy(self.p)
        p["timing_reliability"]["blocks"] = 7
        self.assertFalse(self.admitted(p=p))

    def test_nondefault_complete_records_pass(self):
        p = copy.deepcopy(self.p)
        p["timing_reliability"].update(fixed_warmup_seconds=2.0, block_min_seconds=2.0, blocks=7)
        records = [(record(p), record(p)) for _ in range(p["rounds"])]
        self.assertTrue(self.admitted(records, p))

    def test_missing_block_rejected(self):
        self.records[0][0]["rows"][0]["timing_blocks"].pop()
        self.assertFalse(self.admitted())

    def test_configuration_metadata_mismatch_rejected(self):
        self.records[0][0]["timing_reliability"]["blocks"] += 1
        self.assertFalse(self.admitted())

    def test_single_noisy_block_rejected(self):
        self.records[0][0]["rows"][0]["timing_blocks"][0]["sample_times_ns"] = [2000.0]
        r = summarize(self.p, self.records)["cases"]["quad_4d"]
        self.assertFalse(r["benchmark_admitted"])
        self.assertEqual(r["status"], "TIMING_FAIL")

    def test_ratio_drift_rejected_even_within_language_limits(self):
        for row in self.records[0][0]["rows"]:
            for b in row["timing_blocks"]:
                b["sample_times_ns"] = [1050.0]
        for row in self.records[0][1]["rows"]:
            for b in row["timing_blocks"]:
                b["sample_times_ns"] = [950.0]
        r = summarize(self.p, self.records)["cases"]["quad_4d"]
        self.assertTrue(all(v <= 0.1 for v in r["language_block_spread_fraction"].values()))
        self.assertFalse(r["benchmark_admitted"])

    def test_joint_process_output_drift_rejects_within_pair_agreement(self):
        self.records[1] = (record(self.p, 1.00001), record(self.p, 1.00001))
        r = summarize(self.p, self.records)["cases"]["quad_4d"]
        self.assertTrue(r["strict_parity"])
        self.assertFalse(r["strict_numeric_admitted"])
        self.assertFalse(r["benchmark_admitted"])
        self.assertEqual(r["status"], "PROCESS_OUTPUT_VARIATION")

    def test_incomplete_pairs_rejected(self):
        self.assertFalse(self.admitted(self.records[:-1]))

    def test_numerical_failure_is_not_timing_success(self):
        self.records[0][0]["rows"][0]["result"]["params"][0] += 0.1
        self.assertFalse(self.admitted())

    def test_configuration_rejects_invalid_timing_values(self):
        for field in ("fixed_warmup_seconds", "block_min_seconds"):
            for value in (True, False, 0, -1, float("nan"), float("inf"), "0.5"):
                with self.subTest(field=field, value=value):
                    p = copy.deepcopy(self.p)
                    p["timing_reliability"][field] = value
                    with self.assertRaises(ValueError):
                        comparison.validate_protocol(p)
        for value in (True, 0, -1, 1.5, "7"):
            p = copy.deepcopy(self.p)
            p["timing_reliability"]["blocks"] = value
            with self.assertRaises(ValueError):
                comparison.validate_protocol(p)

    def test_unsupported_solver_and_case_contract_rejected(self):
        for field, value in (("strategy", 1), ("tol", 1.0), ("evals", 2), ("warmups", True),
                             ("cases", list(reversed(self.p["cases"]))), ("order", [])):
            p = copy.deepcopy(self.p)
            p[field] = value
            with self.assertRaises(ValueError):
                comparison.validate_protocol(p)

    def test_sampling_parameters_reach_both_commands(self):
        p = copy.deepcopy(self.p)
        p["timing_reliability"].update(fixed_warmup_seconds=2.0, block_min_seconds=0.25, blocks=7)
        for language in ("julia", "cpp"):
            command = comparison.collector_command(language, ROOT.parent, "cpp", "julia", "out", "inputs", p)
            self.assertEqual(command[-3:], ["2.0", "7", "0.25"])


if __name__ == "__main__":
    unittest.main()
