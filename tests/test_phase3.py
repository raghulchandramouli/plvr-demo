from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from plvr3b.cache import RuntimeCache, VersionStamp
from plvr3b.comparison import evaluate_candidate, run_arm, run_comparison
from plvr3b.contracts import parse_request, registry
from plvr3b.executor import ordered_program, program_data, run_program, signature
from plvr3b.phase2_fixtures import build_fixtures
from plvr3b.planning import SearchLimits, bind_parameters, resolve_types, typed_tree
from plvr3b.primitives import Decompose
from plvr3b.scoring import aggregate_rank, align_calls, call_terms, ranking_terms, selection_key, value_terms
from plvr3b.search import enumerate_programs, fingerprint
from plvr3b.types import Budget, ContractError, OPERATION_SPECS, PrimitiveNode, Sequence, TaskList

ROOT = Path(__file__).resolve().parents[1]
STAMP = VersionStamp(*['version-one'] * 7)
DOMAINS = {name: [{}] for name in OPERATION_SPECS}


class BackwardConstructionTests(unittest.TestCase):
    def test_goal_driven_construction_matches_independent_reference_without_helpers(self):
        with patch('plvr3b.executor.ordered_program', side_effect=AssertionError('template leaked')), patch('plvr3b.executor.unordered_program', side_effect=AssertionError('template leaked')):
            plans, proof = resolve_types()
        reference, _ = enumerate_programs()
        self.assertEqual({fingerprint(program_data(p)) for p in plans}, {key for key, _ in reference})
        self.assertEqual(len(plans), 2)
        self.assertEqual(proof['emitted_layer_plans'], 3)
        self.assertEqual({p.required_inputs for p in plans}, {('Request', 'ToolRegistry')})
        self.assertTrue(all(signature(p) == ('Request', 'ToolCallList') for p in plans))
        tree = typed_tree(plans[1])
        self.assertEqual(set(tree), {'operation', 'name', 'parameters', 'required_inputs', 'output_type', 'constraints', 'children'})
        self.assertEqual(tree['output_type'], 'ToolCallList')
        self.assertNotIn('TaskList', tree['required_inputs'])

    def test_catalog_and_available_inputs_control_derivation(self):
        without_order = {k: v for k, v in OPERATION_SPECS.items() if k != 'Get-Order'}
        self.assertEqual(len(resolve_types(catalog=without_order)[0]), 1)
        self.assertEqual(resolve_types(supplied={'Request'})[0], ())
        self.assertEqual(resolve_types(goal='UnknownResult')[0], ())
        self.assertEqual(resolve_types(limits=SearchLimits(layers=1))[0], ())

    def test_modified_catalog_signatures_and_contexts_reject_explicitly(self):
        for field, value in [('output_type', 'Wrong'), ('context_inputs', ('Other',)), ('version', 'changed')]:
            catalog = dict(OPERATION_SPECS)
            catalog['Get-Order'] = replace(catalog['Get-Order'], **{field: value})
            with self.assertRaises(ContractError) as caught:
                resolve_types(catalog=catalog)
            self.assertEqual(caught.exception.violations[0].code, 'unsupported_catalog')

    def test_parameter_binding_is_separate_and_invalid_values_reject_before_run(self):
        plans, _ = resolve_types()
        bound = bind_parameters(plans, DOMAINS)
        self.assertEqual(len(bound), 2)
        with self.assertRaises(ContractError):
            bind_parameters(plans, {**DOMAINS, 'Get-Order': [{'direction': 'reverse'}]})
        with self.assertRaises(ContractError):
            signature(PrimitiveNode('Match', (('temperature', 1),)))

    def test_layer_plan_caps_and_beam_caps_bind(self):
        plans, proof = resolve_types(limits=SearchLimits(max_layer_plans=2))
        self.assertEqual(len(plans), 1)
        self.assertEqual(proof['stop_reason'], 'plan_budget')
        fixtures = build_fixtures()
        reference, _ = enumerate_programs()
        arm = run_arm('guided', fixtures, registry(), Budget(), SearchLimits(beam_width=1, per_key_cap=1), STAMP, DOMAINS, reference)
        self.assertEqual(arm['beam_before'], 2)
        self.assertEqual(arm['beam_after'], 1)
        self.assertEqual(arm['selected_successes'], 34)
        capped = run_arm('random', fixtures, registry(), Budget(), SearchLimits(max_candidates=1), STAMP, DOMAINS, reference, 0)
        self.assertEqual(capped['evaluated_candidates'], 1)
        self.assertEqual(capped['fixture_evaluations'], 34)
        self.assertEqual(capped['stop_reason'], 'candidate_budget')


class RankingTests(unittest.TestCase):
    def test_content_alignment_handles_insertions_and_partial_bindings(self):
        wanted = [{'tool': 'lookup_customer', 'arguments': {'customer_id': 'C1'}}, {'tool': 'lookup_order', 'arguments': {'order_id': 'O1'}}]
        errors, atoms, alignment = align_calls(list(reversed(wanted)), wanted)
        self.assertEqual((errors, sum(atoms)), (0, 0))
        extra = {'tool': 'lookup_product', 'arguments': {'product_id': 'P1'}}
        errors, atoms, _ = align_calls([extra] + wanted, wanted)
        self.assertEqual(errors, 1)
        self.assertEqual(sum(v*v for v in atoms), 1)
        partial = call_terms({'tool': 'create_return', 'arguments': {'order_id': 'O1', 'reason': 'wrong'}}, {'tool': 'create_return', 'arguments': {'order_id': 'O1', 'reason': 'damaged'}})
        wrong = call_terms(extra, wanted[0])
        self.assertEqual(partial, (0, [0.25]))
        self.assertGreater(wrong[1][0], partial[1][0])

    def test_alignment_preserves_same_functions_before_type_scoring(self):
        wanted = [{'tool': 'lookup_customer', 'arguments': {'customer_id': 'C1'}}, {'tool': 'lookup_order', 'arguments': {'order_id': 'O1'}}]
        predicted = [{'tool': 'lookup_customer', 'arguments': {'customer_id': True}}, wanted[1]]
        errors, atoms, alignment = align_calls(predicted, wanted)
        self.assertEqual(errors, 1)
        self.assertEqual(alignment, [{'predicted': 0, 'expected': 0}, {'predicted': 1, 'expected': 1}])
        self.assertEqual(atoms, [0.5, 0.0])

    def test_value_types_alignment_and_extra_objects_do_not_disappear(self):
        self.assertEqual(value_terms(True, 1), (1, [1.0]))
        left = [{'order_id': 'O2', 'v': 2}, {'order_id': 'O1', 'v': 1}]
        right = list(reversed(left))
        self.assertEqual(sum(value_terms(left, right)[1]), 0)
        self.assertGreater(sum(value_terms([1, 2], [2, 1])[1]), 0)
        self.assertGreater(sum(value_terms({'required': 1, 'extra': 2}, {'required': 1})[1]), 0)

    def test_cumulative_l2_rank_and_fixed_deviation(self):
        one = {'unmet_requirements': 0, 'type_errors': 1, 'value_error_squared': 9}
        two = {'unmet_requirements': 0, 'type_errors': 2, 'value_error_squared': 16}
        self.assertEqual(aggregate_rank([one, two], {'Request', 'ToolRegistry'}, {'Request', 'ToolRegistry'}), (0, 3, 5, 0))
        self.assertEqual(aggregate_rank([one], {'Request'}, {'Request', 'ToolRegistry'})[-1], 1)
        self.assertLess((0, 0, 99, 0), (0, 1, 0, 0))
        self.assertLess((0, 99, 99, 0), (1, 0, 0, 0))
        self.assertLess(selection_key({'rank': [0,0,0,0], 'size': 2, 'label': 'z'}), selection_key({'rank': [0,0,0,0], 'size': 3, 'label': 'a'}))

    def test_empty_success_and_failed_partial_work_get_correct_unmet_flag(self):
        fixtures = build_fixtures()
        empty = fixtures[-2]
        execution = run_program(ordered_program(), parse_request(empty.payload), registry(), empty.initial_state)
        self.assertEqual(ranking_terms(execution, empty)['unmet_requirements'], 0)
        fixture = fixtures[30]
        plans, _ = resolve_types()
        failure = run_program(plans[0], parse_request(fixture.payload), registry(), fixture.initial_state)
        terms = ranking_terms(failure, fixture)
        self.assertEqual(terms['unmet_requirements'], 1)
        self.assertGreater(terms['type_errors'], 0)
        self.assertGreater(terms['value_error_squared'], 0)


class VersionedCacheTests(unittest.TestCase):
    def setUp(self):
        self.fixture = build_fixtures()[0]
        self.request = parse_request(self.fixture.payload)
        self.tools = registry()
        self.cache = RuntimeCache(STAMP)

    def execute(self, **kwargs):
        return run_program(ordered_program(), kwargs.get('request', self.request), kwargs.get('tools', self.tools), kwargs.get('state', self.fixture.initial_state), kwargs.get('budget', Budget()), self.cache)

    def test_keyword_only_tool_defaults_invalidate_execution_identity(self):
        def lookup(state, customer_id, *, name='First'):
            return {**state['customers'][customer_id], 'name': name}

        specs = dict(self.tools.specs)
        specs['lookup_customer'] = replace(specs['lookup_customer'], function=lookup)
        tools = replace(self.tools, specs=specs)
        self.assertEqual(self.execute(tools=tools).result['name'], 'First')
        lookup.__kwdefaults__['name'] = 'Second'
        self.assertEqual(self.execute(tools=tools).result['name'], 'Second')
        self.assertEqual(self.cache.stats['execution_hits'], 0)

    def test_execution_hit_replays_without_primitive_runs_and_copies_mutable_results(self):
        with patch.object(Decompose, 'run', wraps=Decompose().run) as physical:
            first = self.execute()
            misses = self.cache.stats['primitive_misses']
            first.final_state['customers']['C1']['name'] = 'corrupted'
            second = self.execute()
            self.assertEqual(physical.call_count, 1)
            self.assertEqual(self.cache.stats['primitive_misses'], misses)
            self.assertEqual(self.cache.stats['execution_hits'], 1)
            self.assertEqual(second.final_state['customers']['C1']['name'], 'Asha')
            self.assertTrue(all(e['execution_cache_replay'] for e in second.primitive_events))
            second.result['name'] = 'changed'
            self.assertEqual(self.execute().result['name'], 'Asha')

    def test_primitive_prefix_hits_validate_and_do_not_alias(self):
        plans, _ = resolve_types()
        first = run_program(plans[0], self.request, self.tools, self.fixture.initial_state, cache=self.cache)
        second = self.execute()
        self.assertGreater(self.cache.stats['primitive_hits'], 0)
        self.assertEqual(first.result, second.result)
        self.assertTrue(any(e.get('cache_hit') for e in second.primitive_events))

    def test_adapter_and_every_version_identity_force_actual_misses(self):
        for field in STAMP.__dataclass_fields__:
            with self.subTest(field=field):
                self.cache = RuntimeCache(STAMP)
                self.execute()
                prior = self.cache.stats['primitive_misses']
                self.cache.update_versions(replace(STAMP, **{field: 'changed'}))
                self.execute()
                self.assertGreater(self.cache.stats['primitive_misses'], prior)
                self.assertEqual(self.cache.stats['execution_hits'], 0)
                self.assertEqual(self.cache.stats['invalidations'], 1)

    def test_request_provenance_budget_state_and_registry_change_keys(self):
        self.execute()
        original_misses = self.cache.stats['execution_misses']
        changed_state = deepcopy(self.fixture.initial_state)
        changed_state['customers']['C1']['name'] = 'Asha changed'
        self.assertEqual(self.execute(state=changed_state).result['name'], 'Asha changed')
        self.execute(budget=Budget(max_primitive_calls=15))
        payload = deepcopy(self.fixture.payload)
        payload['id'] = 'other_request'
        self.execute(request=parse_request(payload))
        specs = dict(self.tools.specs)
        specs['lookup_customer'] = replace(specs['lookup_customer'], function=lambda state, **kwargs: {'customer_id': 'C1', 'name': 'Different', 'email': 'x@example.test'})
        changed_tools = replace(self.tools, specs=specs)
        self.assertEqual(self.execute(tools=changed_tools).result['name'], 'Different')
        self.assertEqual(self.cache.stats['execution_misses'], original_misses + 4)
        self.assertEqual(self.cache.stats['execution_hits'], 0)

    def test_cached_primitive_output_is_validated_on_hit(self):
        self.execute()
        key = next(event['key'] for event in self.cache.events if event['kind'] == 'primitive')
        self.cache.primitive[key] = TaskList(())
        self.cache.execution.clear()
        failed = self.execute()
        self.assertEqual(failed.status, 'contract_failure')
        self.assertEqual(failed.violations[0].code, 'decompose_contract')
        self.assertEqual(failed.events, [])

    def test_schema_and_runtime_representation_changes_cannot_reuse_entries(self):
        self.execute()
        specs = dict(self.tools.specs)
        specs['lookup_customer'] = replace(specs['lookup_customer'], result_fields=specs['lookup_customer'].result_fields + (('extra', 'string'),))
        failed = self.execute(tools=replace(self.tools, specs=specs))
        self.assertEqual(failed.violations[0].code, 'tool_result_contract')
        self.assertEqual(self.cache.stats['execution_hits'], 0)
        model = yaml.safe_load((ROOT / 'configs/model.yaml').read_text())
        original = VersionStamp.from_root(ROOT, model)
        changed = deepcopy(model)
        changed['dtype'] = 'float16'
        changed['runtime']['weight_dtype'] = 'float16'
        updated = VersionStamp.from_root(ROOT, changed)
        self.assertEqual(updated.model_revision, original.model_revision)
        self.assertNotEqual(updated.runtime_policy, original.runtime_policy)
        self.cache.update_versions(original)
        self.execute()
        before = self.cache.stats['primitive_misses']
        self.cache.update_versions(updated)
        self.execute()
        self.assertGreater(self.cache.stats['primitive_misses'], before)

    def test_changed_expected_answers_recompute_scores_from_cached_execution(self):
        first = evaluate_candidate(ordered_program(), [self.fixture], self.tools, Budget(), self.cache, {'Request', 'ToolRegistry'})
        changed = replace(self.fixture, expected_result={'bad': 'answer'})
        second = evaluate_candidate(ordered_program(), [changed], self.tools, Budget(), self.cache, {'Request', 'ToolRegistry'})
        self.assertEqual(first['rank'], (0,0,0,0))
        self.assertNotEqual(second['rank'], first['rank'])
        self.assertEqual(second['cache_delta']['execution_hits'], 1)


class ComparisonTests(unittest.TestCase):
    def test_all_arms_find_winner_with_equal_cold_budgets_and_fixed_seed_replay(self):
        fixtures = build_fixtures()
        reference, _ = enumerate_programs()
        arms = [run_arm(name, fixtures, registry(), Budget(), SearchLimits(), STAMP, DOMAINS, reference if name == 'exhaustive' else ()) for name in ('exhaustive', 'guided')]
        arms += [run_arm('random', fixtures, registry(), Budget(), SearchLimits(), STAMP, DOMAINS, reference, seed) for seed in (0,1,2)]
        self.assertEqual({a['selected_id'] for a in arms}, {fingerprint(program_data(ordered_program()))})
        self.assertTrue(all(a['selected_successes'] == 34 for a in arms))
        self.assertEqual({a['fixture_manifest_hash'] for a in arms}, {arms[0]['fixture_manifest_hash']})
        self.assertEqual({a['fixture_evaluations'] for a in arms}, {68})
        self.assertTrue(all(a['cache_start'] == 'cold' for a in arms))
        self.assertEqual(arms[2], run_arm('random', fixtures, registry(), Budget(), SearchLimits(), STAMP, DOMAINS, reference, 0))
        self.assertEqual({a['cache_stats']['primitive_misses'] for a in arms}, {115})
        self.assertEqual({a['cache_stats']['primitive_hits'] for a in arms}, {81})

    def test_random_caps_apply_after_uniform_full_pool_sampling(self):
        fixtures = build_fixtures()[:1]
        reference, _ = enumerate_programs()
        first_choices = set()
        for seed in range(6):
            arm = run_arm('random', fixtures, registry(), Budget(), SearchLimits(max_layer_plans=1), STAMP, DOMAINS, reference, seed)
            self.assertEqual(arm['proposed_canonical_candidates'], 2)
            self.assertEqual(arm['evaluated_candidates'], 1)
            self.assertEqual(arm['max_fixture_evaluations'], 1)
            first_choices.add(arm['candidate_order'][0])
        self.assertEqual(len(first_choices), 2)
        for method in ('exhaustive', 'guided', 'random'):
            with self.assertRaises(ContractError):
                run_arm(method, fixtures, registry(), Budget(), SearchLimits(layers=1), STAMP, DOMAINS, reference, 0)

    def test_cli_artifacts_retain_all_scores_and_trace_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'compare'
            with patch('plvr3b.executor.ordered_program', side_effect=AssertionError('template leaked')):
                summary = run_comparison(ROOT, output)
            self.assertEqual(summary['status'], 'verified')
            self.assertFalse(summary['guided_search_advantage'])
            self.assertEqual(len(summary['arms']), 5)
            self.assertEqual(summary['canonical_program_count'], 2)
            self.assertEqual(len(list((output / 'arms').glob('*/*.json'))), 15)
            metadata = json.loads((output / 'metadata.json').read_text())
            self.assertEqual(metadata['preflight_fixture_evaluations'], 68)
            self.assertFalse(metadata['paper_matched'])
            with self.assertRaises(FileExistsError):
                run_comparison(ROOT, output)


if __name__ == '__main__':
    unittest.main()
