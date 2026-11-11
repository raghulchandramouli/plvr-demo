from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from plvr3b.baseline_tasks import TOOLS, initial_state
from plvr3b.contracts import graph, parse_program, parse_request, registry, validate_request
from plvr3b.executor import canonicalize, execute_calls, ordered_program, program_data, run_program, signature, unordered_program
from plvr3b.phase2_fixtures import build_fixtures, fixture_data, grounded
from plvr3b.primitives import Decompose, GetOrder, Match
from plvr3b.scoring import score_fixture
from plvr3b.search import enumerate_programs, fingerprint, run_gate
from plvr3b.types import Budget, Context, ContractError, Foreach, PrimitiveNode, ResultRef, Scratchpad, Sequence, TaskList, ToolCall, ToolCallList

ROOT = Path(__file__).resolve().parents[1]


class StructuredContractTests(unittest.TestCase):
    def setUp(self):
        self.fixtures = build_fixtures()
        self.tools = registry()

    def rejected(self, payload, code):
        with self.assertRaises(ContractError) as caught:
            parse_request(payload, self.tools)
        self.assertEqual(caught.exception.violations[0].code, code)

    def test_nested_transformation_operands_fail_before_recursion(self):
        payload = deepcopy(self.fixtures[-1].payload)
        operand = payload['tasks'][0]['arguments']['quantity']['input']
        for _ in range(1100):
            operand = {'transform': 'parse_integer', 'input': operand}
        payload['tasks'][0]['arguments']['quantity']['input'] = operand
        self.rejected(payload, 'invalid_transform')

    def test_oversized_integer_literal_is_a_contract_failure(self):
        payload = deepcopy(self.fixtures[7].payload)
        payload['tasks'][0]['arguments']['quantity']['value'] = 1 << 20000
        self.rejected(payload, 'invalid_value')

    def test_registry_reuses_every_existing_function_and_schema(self):
        self.assertEqual(set(self.tools.specs), set(TOOLS))
        for name, spec in TOOLS.items():
            self.assertIs(self.tools.specs[name].function, spec.function)
            self.assertEqual(set(dict(self.tools.specs[name].arguments)), set(spec.arguments))
        with self.assertRaises(TypeError):
            self.tools.specs['fake'] = None

    def test_parse_is_deterministic_and_does_not_mutate_input(self):
        payload = self.fixtures[0].payload
        before = deepcopy(payload)
        self.assertEqual(parse_request(payload), parse_request(payload))
        self.assertEqual(payload, before)
        self.assertEqual(set(Context.__dataclass_fields__), {'request', 'registry', 'budget'})

    def test_missing_structured_unknown_and_missing_inputs_fail_explicitly(self):
        self.rejected({'id': 'prose', 'text': 'Find my order'}, 'needs_semantic_extraction')
        payload = deepcopy(self.fixtures[0].payload)
        del payload['text']
        self.rejected(payload, 'missing_field')
        payload = deepcopy(self.fixtures[0].payload)
        payload['tasks'][0]['intent'] = 'invented_tool'
        self.rejected(payload, 'unsupported_intent')
        payload = deepcopy(self.fixtures[0].payload)
        payload['tasks'][0]['arguments'] = {}
        self.rejected(payload, 'argument_fields')

    def test_provenance_bounds_and_actual_source_are_checked(self):
        payload = deepcopy(self.fixtures[0].payload)
        arg = payload['tasks'][0]['arguments']['customer_id']
        arg['value'] = 'C2'
        self.rejected(payload, 'ungrounded_value')
        arg['source']['end'] = 100000
        self.rejected(payload, 'invalid_span')
        payload = deepcopy(self.fixtures[0].payload)
        payload['tasks'][0]['arguments']['customer_id'] = 'C1'
        self.rejected(payload, 'missing_provenance')
        request = parse_request(self.fixtures[0].payload)
        task = request.tasks.items[0]
        value = task.arguments[0][1]
        forged = replace(value, source=replace(value.source, request_id='other'))
        with self.assertRaises(ContractError) as caught:
            validate_request(replace(request, tasks=TaskList((replace(task, arguments=(('customer_id', forged),)),))), self.tools)
        self.assertEqual(caught.exception.violations[0].code, 'invalid_provenance')

    def test_reference_kind_and_fields_are_checked_without_executing_tools(self):
        payload = deepcopy(self.fixtures[18].payload)
        payload['tasks'][1]['arguments']['customer_id'] = {'ref': 'c0.product_id'}
        self.rejected(payload, 'incompatible_reference_type')
        payload['tasks'][1]['arguments']['customer_id'] = {'ref': 'c0.made_up'}
        self.rejected(payload, 'unknown_result_field')
        payload['tasks'][1]['arguments']['customer_id'] = {'ref': 'absent.customer_id'}
        self.rejected(payload, 'unknown_reference')

    def test_boolean_quantity_and_disallowed_values_fail(self):
        payload = deepcopy(self.fixtures[7].payload)
        payload['text'] += ' true 0'
        payload['tasks'][0]['arguments']['quantity'] = grounded(payload['text'], True)
        self.rejected(payload, 'wrong_argument_type')
        payload['tasks'][0]['arguments']['quantity'] = grounded(payload['text'], 0)
        self.rejected(payload, 'invalid_value')
        payload = deepcopy(self.fixtures[17].payload)
        payload['text'] += ' 2026-02-30'
        payload['tasks'][0]['arguments']['date'] = grounded(payload['text'], '2026-02-30')
        self.rejected(payload, 'invalid_value')

    def test_duplicate_tasks_cycles_and_unknown_dependencies_fail(self):
        payload = deepcopy(self.fixtures[18].payload)
        payload['tasks'][1]['id'] = payload['tasks'][0]['id']
        self.rejected(payload, 'duplicate_task')
        payload = deepcopy(self.fixtures[18].payload)
        payload['tasks'][0]['depends_on'] = ['c1']
        self.rejected(payload, 'dependency_cycle')
        payload['tasks'][0]['depends_on'] = ['absent']
        self.rejected(payload, 'unknown_dependency')

    def test_task_budget_and_unsupported_transformation_fail(self):
        payload = deepcopy(self.fixtures[0].payload)
        payload['tasks'] *= 7
        self.rejected(payload, 'task_budget')
        payload = deepcopy(self.fixtures[-1].payload)
        payload['tasks'][0]['arguments']['quantity']['transform'] = 'eval_python'
        self.rejected(payload, 'unsupported_transform')

    def test_integer_transform_rejects_oversized_operand_before_conversion(self):
        payload = deepcopy(self.fixtures[-1].payload)
        digits = '9' * 5000
        payload['text'] = f'Reserve {digits} units of P1 and return its inventory.'
        arguments = payload['tasks'][0]['arguments']
        arguments['product_id'] = grounded(payload['text'], 'P1')
        arguments['quantity']['input'] = grounded(payload['text'], digits)
        self.rejected(payload, 'invalid_transform')

    def test_typed_integer_transform_has_the_same_operand_bound(self):
        request = parse_request(self.fixtures[-1].payload)
        task = request.tasks.items[0]
        arguments = dict(task.arguments)
        transform = arguments['quantity']
        digits = '9' * 5000
        text = f'Reserve {digits} units of P1 and return its inventory.'
        source = replace(transform.operand.source, start=8, end=8 + len(digits))
        arguments['quantity'] = replace(transform, operand=replace(transform.operand, value=digits, source=source))
        product = arguments['product_id']
        product_start = text.index('P1')
        arguments['product_id'] = replace(product, source=replace(product.source, start=product_start, end=product_start + 2))
        request = replace(request, text=text, tasks=TaskList((replace(task, arguments=tuple(arguments.items())),)))
        with self.assertRaises(ContractError) as caught:
            validate_request(request, self.tools)
        self.assertEqual(caught.exception.violations[0].code, 'invalid_transform')

    def test_empty_collection_is_valid_and_missing_return_is_not(self):
        request = parse_request(self.fixtures[-2].payload)
        self.assertEqual(request.tasks.items, ())
        payload = deepcopy(self.fixtures[0].payload)
        payload['return'] = None
        self.rejected(payload, 'missing_return')


class PrimitiveAndExecutionTests(unittest.TestCase):
    def setUp(self):
        self.fixtures = build_fixtures()
        self.tools = registry()
        self.request = parse_request(self.fixtures[18].payload)
        self.context = Context(self.request, self.tools, Budget())

    def test_34_oracles_run_against_independently_declared_expectations(self):
        self.assertEqual(len(self.fixtures), 34)
        for fixture in self.fixtures:
            with self.subTest(fixture=fixture.id):
                original = deepcopy(fixture.initial_state)
                execution = run_program(ordered_program(), parse_request(fixture.payload), self.tools, fixture.initial_state)
                self.assertTrue(score_fixture(execution, fixture)['success'])
                self.assertEqual(execution.final_state, fixture.expected_state)
                self.assertEqual(fixture.initial_state, original)

    def test_validate_rejects_dropped_tasks_wrong_order_and_fabricated_binding(self):
        tasks = self.request.tasks
        self.assertEqual(Decompose().validate(self.request, TaskList(tasks.items[:1]), self.context)[0].code, 'decompose_contract')
        self.assertEqual(GetOrder().validate(tasks, TaskList(tuple(reversed(tasks.items))), self.context)[0].code, 'get_order_contract')
        task = tasks.items[0]
        bad = ToolCall(task.id, 'lookup_customer', task.arguments)
        self.assertEqual(Match().validate(task, bad, self.context)[0].code, 'match_contract')
        wrong = replace(task, arguments=())
        self.assertEqual(Match().validate(wrong, ToolCall(wrong.id, wrong.intent, ()), self.context)[0].code, 'match_contract')
        self.assertEqual(GetOrder().validate('not tasks', tasks, self.context)[0].code, 'wrong_input_type')

    def test_get_order_changes_dependency_execution_not_literal_values(self):
        fixture = self.fixtures[30]
        request = parse_request(fixture.payload)
        unordered = run_program(unordered_program(), request, self.tools, fixture.initial_state)
        ordered = run_program(ordered_program(), request, self.tools, fixture.initial_state)
        self.assertEqual(unordered.violations[0].code, 'unresolved_dependency')
        self.assertEqual(len(unordered.events), 0)
        self.assertTrue(score_fixture(ordered, fixture)['success'])
        self.assertEqual([e['tool'] for e in ordered.events], ['lookup_order', 'lookup_customer'])
        self.assertEqual(ordered.events[1]['arguments'], {'customer_id': 'C1'})
        self.assertEqual(ordered.events[1]['argument_provenance']['customer_id'], {'task_id': 'c0', 'field': 'customer_id'})
        self.assertEqual(ordered.result_provenance['c1']['provenance']['email'], {'task_id': 'c1', 'field': 'email'})

    def test_ordered_fixture_cost_differs_by_one_actual_oracle_invocation(self):
        fixture = self.fixtures[18]
        direct = run_program(unordered_program(), self.request, self.tools, fixture.initial_state)
        sorted_run = run_program(ordered_program(), self.request, self.tools, fixture.initial_state)
        self.assertTrue(score_fixture(direct, fixture)['success'])
        self.assertEqual(len(direct.primitive_events), 3)
        self.assertEqual(len(sorted_run.primitive_events), 4)

    def test_mutation_candidates_reset_state_and_reference_dynamic_result(self):
        fixture = self.fixtures[22]
        request = parse_request(fixture.payload)
        first = run_program(ordered_program(), request, self.tools, fixture.initial_state)
        second = run_program(ordered_program(), request, self.tools, fixture.initial_state)
        self.assertEqual(first.final_state, second.final_state)
        self.assertNotIn('R2', fixture.initial_state['returns'])
        self.assertEqual(first.events[1]['arguments']['return_id'], 'R2')
        self.assertEqual(first.events[1]['argument_provenance']['return_id'], {'task_id': 'c0', 'field': 'return_id'})

    def test_foreach_empty_and_bounded_iterations_tool_and_primitive_budgets(self):
        empty = self.fixtures[-2]
        result = run_program(ordered_program(), parse_request(empty.payload), self.tools, empty.initial_state)
        self.assertEqual(result.events, [])
        self.assertEqual([e['name'] for e in result.primitive_events], ['Decompose', 'Get-Order'])
        for budget, code in [(Budget(max_iterations=1), 'iteration_budget'), (Budget(max_tool_calls=1), 'tool_budget'), (Budget(max_primitive_calls=1), 'primitive_budget')]:
            execution = run_program(ordered_program(), self.request, self.tools, initial_state(), budget)
            self.assertEqual(execution.violations[0].code, code)

    def test_duplicate_extra_missing_and_fabricated_calls_fail_before_mutation(self):
        calls = tuple(Match().run(task, self.context) for task in self.request.tasks.items)
        for items, code in [(calls + (calls[0],), 'duplicate_call'), (calls + (replace(calls[0], id='extra'),), 'unrequested_call'), (calls[:1], 'missing_call'), ((replace(calls[0], tool='lookup_customer'), calls[1]), 'binding_contract')]:
            scratch = Scratchpad(self.request, self.tools, Budget(), initial_state())
            with self.assertRaises(ContractError) as caught:
                execute_calls(ToolCallList(items), self.context, scratch)
            self.assertEqual(caught.exception.violations[0].code, code)
            self.assertEqual(scratch.events, [])

    def test_tool_output_schema_is_checked_before_reference_use(self):
        request = parse_request(self.fixtures[0].payload)
        specs = dict(self.tools.specs)
        specs['lookup_customer'] = replace(specs['lookup_customer'], function=lambda state, **kwargs: {'invented': 'value'})
        execution = run_program(ordered_program(), request, replace(self.tools, specs=specs), initial_state())
        self.assertEqual(execution.violations[0].code, 'tool_result_contract')
        self.assertEqual(execution.result_provenance, {})

    def test_tool_precondition_failure_retains_completed_events(self):
        fixture = self.fixtures[18]
        state = deepcopy(fixture.initial_state)
        del state['customers']['C1']
        execution = run_program(ordered_program(), self.request, self.tools, state)
        self.assertEqual(execution.violations[0].code, 'tool_precondition')
        self.assertEqual(len(execution.events), 1)
        self.assertEqual(execution.events[0]['tool'], 'lookup_order')

    def test_explicit_transform_is_preserved_and_executed(self):
        fixture = self.fixtures[-1]
        execution = run_program(ordered_program(), parse_request(fixture.payload), self.tools, fixture.initial_state)
        self.assertEqual(execution.events[0]['arguments']['quantity'], 3)
        self.assertEqual(execution.events[0]['argument_provenance']['quantity']['name'], 'parse_integer')
        self.assertEqual(execution.events[0]['argument_provenance']['quantity']['operand']['value'], '03')


class FiniteGrammarTests(unittest.TestCase):
    def test_program_parser_bounds_depth_before_recursing(self):
        raw = {'op': 'Primitive', 'name': 'Match'}
        for _ in range(1100):
            raw = {'op': 'Foreach', 'child': raw}
        with self.assertRaises(ContractError) as caught:
            parse_program(raw)
        self.assertEqual(caught.exception.violations[0].code, 'program_depth')
        self.assertEqual(parse_program(program_data(ordered_program())), ordered_program())

    def test_program_parser_bounds_sequence_width(self):
        raw = {'op': 'Sequence', 'children': [{'op': 'Primitive', 'name': 'Match'}] * 5}
        with self.assertRaises(ContractError) as caught:
            parse_program(raw)
        self.assertEqual(caught.exception.violations[0].code, 'program_width')

    def test_enumeration_is_typed_and_canonical_not_syntax_inflation(self):
        programs, records = enumerate_programs()
        self.assertEqual(len(records), 1554)
        self.assertEqual(sum(r['type_valid'] for r in records), 3)
        self.assertEqual(len(programs), 2)
        self.assertTrue(all(signature(p) == ('Request', 'ToolCallList') for _, p in programs))
        repeated = Sequence((PrimitiveNode('Decompose'), PrimitiveNode('Get-Order'), PrimitiveNode('Get-Order'), Foreach(PrimitiveNode('Match'))))
        nested = Sequence((PrimitiveNode('Decompose'), Sequence((PrimitiveNode('Get-Order'), Foreach(PrimitiveNode('Match'))))))
        self.assertEqual(canonicalize(repeated), ordered_program())
        self.assertEqual(canonicalize(nested), ordered_program())
        self.assertEqual(canonicalize(Sequence((PrimitiveNode('Match'),))), PrimitiveNode('Match'))

    def test_bad_compositions_nested_loops_and_unsupported_parallel_are_rejected(self):
        for program in (Sequence((PrimitiveNode('Match'), PrimitiveNode('Decompose'))), Foreach(Foreach(PrimitiveNode('Match')))):
            with self.assertRaises(ContractError):
                signature(program)
        for op in ('Parallel', 'Loop', 'Identity'):
            with self.assertRaises(ContractError) as caught:
                parse_program({'op': op})
            self.assertEqual(caught.exception.violations[0].code, 'unsupported_operator')

    def test_exhaustive_cli_artifacts_and_small_space_verdict_are_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / 'first'
            second = Path(directory) / 'second'
            one = run_gate(ROOT, first)
            two = run_gate(ROOT, second)
            self.assertEqual(one, two)
            self.assertEqual(one['useful_workflow_count'], 1)
            self.assertEqual(one['canonical_program_count'], 2)
            self.assertEqual(one['distinct_behavior_count'], 2)
            self.assertFalse(one['guided_search_justified'])
            self.assertEqual(one['verdict'], 'exhaustive_only')
            self.assertEqual(sorted(c['successes'] for c in one['candidates']), [32, 34])
            self.assertEqual(len(list((first / 'traces').glob('*.json'))), 68)
            self.assertEqual((first / 'fixtures.json').read_bytes(), (second / 'fixtures.json').read_bytes())
            self.assertEqual(json.loads((first / 'metadata.json').read_text())['model_calls'], 0)
            with self.assertRaises(FileExistsError):
                run_gate(ROOT, first)


if __name__ == '__main__':
    unittest.main()
