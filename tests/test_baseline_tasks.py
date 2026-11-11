from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from plvr3b.baseline_tasks import (
    TOOLS, build_smoke_cases, evaluate, manifest, prompt_messages,
    public_input, ref, sequence, validate_oracles,
)


class SmokeTaskTests(unittest.TestCase):
    def setUp(self):
        self.cases = build_smoke_cases()

    def test_all_thirty_independently_specified_oracles_execute(self):
        self.assertEqual(len(self.cases), 30)
        self.assertEqual(len({case.id for case in self.cases}), 30)
        self.assertEqual(len(validate_oracles(self.cases)), 30)
        self.assertEqual({call["tool"] for case in self.cases for call in case.oracle["calls"]}, set(TOOLS))
        self.assertEqual(len(TOOLS), 18)
        self.assertEqual(sum(len(case.oracle["calls"]) == 1 for case in self.cases), 18)
        self.assertGreaterEqual(sum('"$ref": "c0.' in json.dumps(case.oracle) for case in self.cases), 6)

    def test_inventory_and_booking_results_are_not_self_scored(self):
        reserved = evaluate(json.dumps(self.cases[7].oracle), self.cases[7])
        self.assertEqual(reserved["result"]["available"], 5)
        self.assertEqual(reserved["result"]["reserved"], 5)
        booking = evaluate(json.dumps(self.cases[15].oracle), self.cases[15])
        self.assertEqual(booking["result"]["booking_id"], "B3")
        self.assertEqual(booking["final_state"]["bookings"]["B3"]["destination"], "Madurai")

    def test_generator_is_repeatable_and_split_is_smoke_only(self):
        first, second = manifest(self.cases), manifest(build_smoke_cases())
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual({case["split"] for case in first["cases"]}, {"smoke-development"})
        descriptor = json.loads((Path(__file__).resolve().parents[1] / "data/manifests/smoke-v1.json").read_text())
        canonical = (json.dumps(first, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), descriptor["corpus_sha256"])
        self.assertEqual({tool["name"] for tool in descriptor["tool_catalog"]}, set(TOOLS))

    def test_prompt_has_only_public_fields_and_no_oracle(self):
        for case in self.cases:
            public = public_input(case)
            self.assertEqual(set(public), {"request", "available_tools"})
            text = json.dumps(prompt_messages(public))
            self.assertNotIn("expected_state", text)
            self.assertNotIn("oracle", text)
            self.assertNotIn("expected_actions", text)
            self.assertEqual({tool["name"] for tool in public["available_tools"]}, set(case.available_tools))
            self.assertEqual(set(case.available_tools), set(TOOLS))

    def test_format_failures_are_strict_and_no_json_repair_occurs(self):
        for raw in ("no", '```json\n{}\n```', '{"op":1,"op":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e400}'):
            with self.subTest(raw=raw):
                report = evaluate(raw, self.cases[0])
                self.assertEqual(report["status"], "format_failure")
                self.assertTrue(report["violations"])

    def test_invalid_structure_unknown_tool_and_arguments_are_contract_failures(self):
        invalid = [[], {}, {"op": "Parallel", "calls": [], "return": None}]
        for name, args in (("unknown", {}), ("lookup_customer", {}), ("lookup_customer", {"customer_id": "C1", "extra": 1})):
            invalid.append(sequence((name, args)))
        for program in invalid:
            with self.subTest(program=program):
                self.assertEqual(evaluate(json.dumps(program), self.cases[0])["status"], "contract_failure")

    def test_wrong_known_identifier_is_a_semantic_failure(self):
        report = evaluate(json.dumps(sequence(("lookup_customer", {"customer_id": "C2"}))), self.cases[0])
        self.assertEqual(report["status"], "semantic_failure")
        self.assertEqual(report["events"][0]["result"]["name"], "Ben")

    def test_forward_missing_and_duplicate_references_fail_contracts(self):
        case = self.cases[18]
        for reference in ("c9.customer_id", "c0.unknown", "c0.customer_id.extra"):
            program = deepcopy(case.oracle)
            program["calls"][1]["arguments"]["customer_id"] = ref(reference)
            self.assertEqual(evaluate(json.dumps(program), case)["status"], "contract_failure")
        program = deepcopy(case.oracle)
        program["calls"][1]["id"] = "c0"
        self.assertEqual(evaluate(json.dumps(program), case)["status"], "contract_failure")

    def test_booleans_are_not_integer_quantities(self):
        program = sequence(("reserve_inventory", {"product_id": "P1", "quantity": True}))
        self.assertEqual(evaluate(json.dumps(program), self.cases[7])["status"], "contract_failure")

    def test_state_resets_and_partial_execution_survives_contract_failure(self):
        case = self.cases[7]
        before = deepcopy(case.initial_state)
        first = evaluate(json.dumps(case.oracle), case)
        second = evaluate(json.dumps(case.oracle), case)
        self.assertEqual(first, second)
        self.assertEqual(case.initial_state, before)
        program = deepcopy(case.oracle)
        program["calls"].append({"id": "c1", "tool": "unknown", "arguments": {}})
        failed = evaluate(json.dumps(program), case)
        self.assertEqual(failed["status"], "contract_failure")
        self.assertEqual(len(failed["events"]), 1)
        self.assertEqual(failed["final_state"]["products"]["P1"]["available"], 5)
        self.assertEqual(case.initial_state, before)

    def test_extra_read_fails_even_when_output_and_state_match(self):
        case = self.cases[0]
        program = sequence(("lookup_customer", {"customer_id": "C1"}), ("lookup_customer", {"customer_id": "C1"}))
        report = evaluate(json.dumps(program), case)
        self.assertEqual(report["result"], case.expected_result)
        self.assertEqual(report["final_state"], case.initial_state)
        self.assertEqual(report["status"], "semantic_failure")

    def test_call_ids_do_not_determine_semantic_success(self):
        case = self.cases[18]
        program = deepcopy(case.oracle)
        program["calls"][0]["id"] = "order"
        program["calls"][1]["id"] = "person"
        program["calls"][1]["arguments"]["customer_id"] = ref("order.customer_id")
        program["return"] = ref("person")
        self.assertEqual(evaluate(json.dumps(program), case)["status"], "success")

    def test_required_dependency_cannot_be_replaced_by_a_literal_guess(self):
        case = self.cases[18]
        program = deepcopy(case.oracle)
        program["calls"][1]["arguments"]["customer_id"] = "C1"
        report = evaluate(json.dumps(program), case)
        self.assertEqual(report["result"], case.expected_result)
        self.assertEqual(report["status"], "semantic_failure")

    def test_required_dependency_rejects_reversed_calls_with_correct_values(self):
        case = self.cases[18]
        program = sequence(
            ("lookup_customer", {"customer_id": "C1"}),
            ("lookup_order", {"order_id": "O1"}),
        )
        program["return"] = ref("c0")
        report = evaluate(json.dumps(program), case)
        self.assertEqual(report["result"], case.expected_result)
        self.assertEqual(report["status"], "semantic_failure")

    def test_independent_calls_can_be_reordered(self):
        case = self.cases[28]
        program = sequence(
            ("lookup_order", {"order_id": "O1"}),
            ("lookup_customer", {"customer_id": "C2"}),
        )
        program["return"] = ref("c0")
        self.assertEqual(evaluate(json.dumps(program), case)["status"], "success")
