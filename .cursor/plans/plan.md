# PLVR-3B

### A shared-backbone reproduction study of Program Learning with Verifiable Rewards

Reproduce the core mechanisms of [the PLVR paper](https://arxiv.org/html/2608.28421v1) using one shared [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) backbone, typed programs, verifiable contracts, and primitive-specific LoRA adapters.

> **Status: implementation plan.** The repository structure, configuration, and milestones below describe work to build. No implementation, benchmark results, or performance improvements are claimed yet.

## 1. Research question and scope

**Can typed, contract-checked program search make a 3B language model more reliable than the same model using ordinary prompting or a fixed workflow, under controlled computational budgets?**

This project investigates three sources of improvement separately:

1. Breaking a task into specialized operations.
2. Learning how to compose those operations into a program.
3. Fine-tuning the operations using verified training examples.

### What counts as reproduction?

| Milestone | What it establishes |
| --- | --- |
| Mechanism reproduction | Typed execution, contracts, backward search, scoring, and the search/retraining cycle work correctly. |
| Experimental reproduction | The adaptation is evaluated on relevant benchmarks against controlled baselines. |
| Exact numerical replication | Requires matching the original architecture, data, evaluation, and compute; this project does not initially claim this. |

The shared 3B backbone replaces the paper's multi-model primitive library. This is a deliberate architectural adaptation, not a claim that the original system has been reproduced unchanged.

Start with tool-calling. Add coding and conversational benchmarks only after the core mechanism is validated.

### Initial non-goals

- Beating frontier models or reproducing headline scores immediately.
- Training a foundation model from scratch.
- Connecting experimental agents to real accounts or destructive tools.
- Building distributed training infrastructure before profiling demonstrates a need.

## 2. Model and execution strategy

| Item | Choice |
| --- | --- |
| Shared backbone | `Qwen/Qwen2.5-3B-Instruct` |
| Model size | Approximately 3.09B total parameters; not 3B active parameters in a larger MoE. |
| Initial specialization | Different prompts for different primitives. |
| Later specialization | A separate LoRA adapter for each neural primitive. |
| Local development | 48 GB Apple-silicon Mac, using MLX LM where supported. |
| Optional cloud training | Transformers + PEFT on a rented GPU. |
| Primary evaluation backend | Select one backend and keep it fixed for the main comparisons. |

Load the backbone once and select the appropriate primitive adapter for each operation. Keep the original backbone unchanged. Do not merge adapters into the baseline checkpoint.

One backbone does **not** mean one model call. A candidate program may invoke Qwen several times. Measure the entire system's calls, tokens, memory, and latency.

Pin the model revision, tokenizer revision, library versions, and decoding settings. Review the model license and dataset licenses before distributing checkpoints or data.

## 3. System architecture

```text
User request + tool registry + environment context
                        |
                        v
             Structured input parser
                        |
                        v
                  Typed input state
                        |
                        v
       Candidate program selected through search
       +-----------------------------------------+
       | Decompose -> Get-Order -> Match -> Check |
       |                                         |
       | Neural operations:                      |
       | shared Qwen + selected prompt/adapter   |
       |                                         |
       | Deterministic operations:               |
       | type checks, contracts, bindings, loops |
       +-----------------------------------------+
                        |
                        v
               Validated tool calls
                        |
                        v
          Mock execution + task evaluator
                        |
                        v
          Traces, failures, scores, and costs
```

This diagram shows an illustrative candidate, not a permanently hard-coded workflow. Program search must be able to propose alternative compositions.

### Learning versus evaluation

During learning, labeled search examples help score and select reusable programs.

During held-out evaluation, freeze the selected program, adapters, prompts, and settings. The program can branch on its inputs, but it must not receive reference answers or use hidden-test feedback to select a better program for that test instance.

## 4. Primitive library

### First implementation: three neural primitives

| Primitive | Responsibility | Validation |
| --- | --- | --- |
| `Decompose` | Extract atomic tasks and references to supporting source text. | Schema validity, source grounding, and task coverage against references. |
| `Get-Order` | Identify task dependencies and propose a valid execution order. | Cycle detection and consistency with declared dependencies. |
| `Match` | Select a tool and bind its arguments. | Tool existence, required arguments, types, constraints, and reference correctness. |

### Expanded implementation

| Primitive | Planned role |
| --- | --- |
| `Check-Prerequisites` | Check whether required conditions are satisfied before an operation. |
| `Elaborate` | Expand terms using an explicit, controlled dictionary. |
| `Denote` | Convert supported language into the formal representation required by the task. |
| `Init-Scratchpad` | Initialize structured working state; deterministic. |
| `Identity` | Preserve an input unchanged; deterministic. |
| `Wrap` | Convert an object into a one-element collection; deterministic. |
| `Append` | Add or combine collection entries under declared semantics; deterministic. |

Audit exact signatures and behavior against the target paper before claiming primitive-level fidelity. Record deviations in `assumptions.md`.

### Shared primitive interface

```python
# Interface sketch, not an implemented API.
result = primitive.run(input_state, parameters)

violations = primitive.validate(
    input_state=input_state,
    output_state=result,
)
```

Keep execution, contract validation, and task scoring separate.

**Valid JSON is not the same as a valid tool call, and a valid tool call is not necessarily the correct action.** A wrong booking ID can pass a string-type check. A valid dependency graph can still omit a real dependency.

## 5. Typed state and execution contracts

Start with these explicit types:

```text
Request
SourceSpan
Task
TaskList
DependencyGraph
ToolRegistry
ToolCall
ToolCallList
Scratchpad
ContractViolation
```

Retain provenance wherever possible: which request span, previous tool result, or explicit transformation supports each value?

Use deterministic parsing for already structured inputs. Introduce neural extraction only when the input actually requires semantic interpretation.

For extracted text, retain source-span references. Make transformations such as date resolution explicit rather than silently inventing normalized values.

### Contracts to implement

- Required fields, accepted types, and allowed values.
- Valid tool names and argument bindings.
- Resolvable references to earlier tool outputs.
- Task ordering and prerequisite constraints.
- Maximum loop iterations and tool-call budgets.
- Explicit behavior for missing information and unsupported requests.

Resolve references to earlier tool results at execution time. Do not require the model to guess results that are not available yet.

Use mock environments, reset state between candidate executions, and reject unsafe parallel mutation. Define fallback behavior carefully so failed attempts do not silently leave state changes behind.

### Oracle-first testing

Before using model outputs, implement deterministic oracle primitives for a small set of fixtures.

Test valid programs, incompatible types, missing inputs, unresolved references, empty collections, duplicate calls, dependency cycles, bounded loops, and unwanted extra actions.

**If oracle primitives cannot solve a known-solvable task, fix the runtime before fine-tuning Qwen.**

## 6. Program representation and search

Represent programs as abstract syntax trees, not unrestricted generated strings.

Each node should store its operation, children, parameters, required inputs, output types, and execution constraints.

Start with `Sequence` and `Foreach`. Add `Parallel`, `Branch`, `Fallback`, and bounded `Loop` after basic execution works.

### Backward type reasoning

A simplified example:

```text
Decompose:
    Request -> TaskList

Match:
    Task x ToolRegistry -> ToolCall

Foreach(Match):
    TaskList x ToolRegistry -> ToolCallList

Combined program:
    Request x ToolRegistry -> ToolCallList
```

Starting from the desired `ToolCallList`, search identifies operations that can produce it and recursively derives their input requirements.

Intermediate values produced inside a composition are not external requirements. If `Decompose` produces `TaskList`, the full program should not also demand that the user supply `TaskList`.

Preserve shared context, such as `ToolRegistry`, explicitly.

### Search loop

```text
Propose candidate program structures
                  |
                  v
Reject incompatible or unsatisfiable type compositions
                  |
                  v
Execute surviving candidates on labeled search examples
                  |
                  v
Score outputs, contract failures, and execution failures
                  |
                  v
Keep promising candidates within the beam
                  |
                  v
Repeat within the declared search budget
```

Types constrain what is executable. They do not establish semantic correctness. Execution and task evaluation determine whether a candidate is useful.

### Starting and expanded configurations

| Setting | Initial debugging configuration | Expanded configuration |
| --- | ---: | ---: |
| Program layers | 2 | 4 |
| Maximum composition depth | 2 | 3 |
| Beam width | 4 | 8 |
| Per-key candidate cap | 4 | 8 |
| Maximum emitted layer plans | 20 | 60 |

These are implementation targets from the project plan. Audit how they map to the paper's definitions before labeling a run paper-matched.

### Scoring

Separate non-executable candidates, type errors, and semantic/value errors. Implement cumulative, type-first ranking and document its exact ordering and normalization.

An intended ranking structure to audit against the paper is:

```text
(unmet_requirements, type_errors, value_error, deviation)
```

Declare partial-credit rules, object alignment, extra-call penalties, and tie-breaking in `assumptions.md`.

When ranking candidates cumulatively, evaluate them on the same relevant examples. Do not compare one candidate scored on five easy examples with another scored on fifty mixed examples.

Reference answers belong in the scorer, not in the runtime inputs to neural primitives.

### Caching

Cache primitive outputs using at least:

```text
input state
primitive identity and implementation version
model revision
adapter revision
prompt version
decoding settings
```

Cache execution results only when the environment state and execution semantics are also captured. Invalidate affected entries after adapter updates.

### Random-search control

Implement random program sampling early. Hold the primitive library, valid grammar, evaluator, and execution budget fixed when comparing it with guided search.

Document the sampling distribution. Randomly generating syntax trees does not automatically sample valid programs uniformly.

## 7. Data plan and leakage prevention

Start with a controlled mock-tool environment containing approximately 15-20 functions.

```python
# Example tool schemas to implement against mock data.
lookup_customer(customer_id)
lookup_order(order_id)
check_inventory(product_id)
check_return_eligibility(order_id)
create_return(order_id, reason)
lookup_booking(booking_id)
search_alternatives(destination, date)
```

Include conditional tasks, missing information, conflicting instructions, and situations where executing no tool is correct.

Example request:

> Return the damaged item from order A123, but only if it is eligible.

### Proposed starting allocations

| Split | Initial size | Purpose |
| --- | ---: | --- |
| Primitive training | 1,000 examples per initial neural primitive | Train individual operations. |
| Primitive validation | 200 examples per initial neural primitive | Select adapters and inspect generalization. |
| Program search | 100 examples | Discover reusable programs. |
| Program development | 100 examples | Select search settings and final candidates. |
| Locked synthetic evaluation | 500 examples | Measure held-out task success. |
| External benchmark evaluation | Preselected categories and examples | Evaluate beyond the synthetic testbed. |

The first three primitives therefore start with approximately 3,000 training and 600 validation examples. These are proposed project allocations, not counts reported by the paper.

Split by scenario, task template, and tool family where appropriate. Do not place paraphrases of the same underlying example across training and test sets.

Keep an in-distribution held-out split and, where practical, a separate transfer split with unseen tool families. Label the distinction explicitly.

### Generate structure first, language second

Construct a structured task with a known solution, then express it in natural language.

- For ordering, generate a dependency graph and derive a valid order programmatically.
- For matching, sample tool schemas and arguments before writing the request.
- For decomposition, combine known atomic intents while retaining their original labels.

Paraphrasing can add variety, but verify that meaning and labels are preserved. Teacher-model outputs are proposed labels, not automatic ground truth.

Do not use evaluation answers, hidden test results, or held-out benchmark examples for primitive training, program search, prompt changes, or hyperparameter selection.

## 8. Primitive fine-tuning

Train one adapter per neural primitive from the same unchanged backbone.

| Setting | Proposed starting value |
| --- | --- |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Learning rate | `1e-4`, with a development-only comparison against `5e-5` |
| Sequence length | 2,048 tokens initially |
| Effective batch size | 16 |
| Epoch limit | Up to 2, selecting checkpoints by validation |
| Loss masking | Assistant-target tokens only |

These are proposed adapter-training settings, not the paper's original training recipe. Check backend-specific parameter semantics rather than copying settings blindly between PEFT and MLX.

Choose a microbatch size that fits measured memory usage and use gradient accumulation as needed. Track truncation; do not silently discard essential tool definitions or target content.

Select checkpoints using semantic correctness as well as schema validity. For dependency ordering, accept any valid topological order instead of requiring one arbitrary textual ordering.

## 9. Alternating search and retraining

Once frozen-primitive search works, implement the complete cycle:

```text
Freeze primitive adapters
        -> Search for program structures
        -> Inspect failures on training/search data
        -> Construct verified correction examples
        -> Retrain affected adapters
        -> Invalidate stale caches
        -> Rescore candidate programs
        -> Repeat within a fixed budget
```

A required output type does not supply the correct semantic answer. Do not turn failed outputs or inferred types into training labels without a trustworthy construction or verification process.

Keep development data for selection and the locked test set for final evaluation. Declare stopping rules before inspecting final results.

## 10. Baselines and ablations

| Experimental arm | Question |
| --- | --- |
| Plain Qwen | What can the unchanged model do directly? |
| Fixed hand-written workflow | Does ordinary decomposition already provide the benefit? |
| Random search + unadapted primitives | What does the program space provide without guided search or training? |
| Guided search + unadapted primitives | Does guided search help without fine-tuning? |
| Random search + adapted primitives | How much comes from stronger components? |
| Guided search + adapted primitives | What does the full adaptation achieve? |
| Token-budget-matched iterative Qwen | Could similar inference effort achieve the gain without program synthesis? |

Keep adapter weights fixed within each random-versus-guided comparison.

Use the same output parsing and comparable structured-generation assistance across arms. Otherwise, cleaner formatting can be mistaken for better reasoning.

Match candidate-execution budgets and report actual tokens and runtime. Equal candidate counts are not automatically equal compute.

Start with three independent search seeds; use five for the main comparison when affordable. Rerun search itself rather than only reordering a fixed pool of candidates.

Record search/training cost separately from per-task inference cost. Report the number of deployments required to amortize setup cost only when the measurements support that calculation.

## 11. Evaluation roadmap

### Stage A: synthetic tool-calling

Validate the runtime and search using controlled mock tools. Report these as synthetic-testbed results, not BFCL or tau2-bench scores.

### Stage B: BFCL

Start with preselected single-turn function-calling categories. Pin the dataset revision, example IDs, evaluation code, and scoring configuration.

Use the established evaluator where applicable. Keep these examples out of primitive training and program selection.

Reference: [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html).

### Stage C: LiveCodeBench

Add coding only after tool-calling is stable. Explicitly pin the benchmark release and date range; a cumulative release alone does not identify a narrow evaluation window.

Audit the target paper's evaluation window before claiming a comparable result.

Run generated code in isolated environments with resource limits, no credentials, and no unnecessary network access. Do not expose hidden-test feedback to program search.

Reference: [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench).

### Stage D: tau2-bench

Start with one domain and expand later. Pin the benchmark revision, task set, policies, user simulator, prompts, and simulator settings.

Use task-success evaluation rather than assuming one exact tool-call sequence is the only valid solution. Report simulator usage and cost separately.

Reference: [tau2-bench](https://github.com/sierra-research/tau2-bench).

Completing BFCL alone is a useful tool-calling reproduction milestone, not a reproduction of every experiment in the paper.

## 12. Metrics and logging

Track strict task success, schema validity, contract failures, semantic error categories, calls and tokens per task, median and tail latency, peak memory, search cost, training cost, and total expenditure.

Report uncertainty across tasks and independent search runs. For comparisons on the same tasks, retain paired outcomes.

Count model failures and timeouts under a declared failure policy. Report infrastructure failures and any permitted reruns separately. Do not silently exclude failed runs from averages.

### Minimum per-example record

```text
example_id
split and dataset revision
model and tokenizer revisions
adapter revision
prompt version
program identifier
search seed
raw model output
parsed output
intermediate execution trace
validation failures
task success
input and output tokens
model and tool call counts
elapsed time
```

### Results template

| Method | Task success | Contract failure rate | Output tokens/task | Median latency | Status |
| --- | --- | --- | --- | --- | --- |
| Plain Qwen | - | - | - | - | Not run |
| Fixed workflow | - | - | - | - | Not run |
| Random search, unadapted | - | - | - | - | Not run |
| Guided search, unadapted | - | - | - | - | Not run |
| Random search, adapted | - | - | - | - | Not run |
| Guided search, adapted | - | - | - | - | Not run |
| Budget-matched iterative Qwen | - | - | - | - | Not run |

## 13. Six-week implementation plan

Suggested allocation: approximately 15-20 focused hours per week. Progress depends on completion gates, not the calendar.

| Week | Work | Completion gate |
| --- | --- | --- |
| 1 | Model wrapper, mock tools, split manifests, baseline evaluator, and logging. | A rerunnable baseline with inspectable failures. |
| 2 | Typed objects, primitive interfaces, contracts, executor, and oracle fixtures. | Known-valid programs run; deliberately invalid programs fail correctly. |
| 3 | Backward search, beam ranking, caching, and random-search control. | Search discovers a known-solvable oracle program without receiving its structure. |
| 4 | Verified primitive data and separate LoRA adapters. | Held-out primitive evaluation is complete; adapters are selected using validation. |
| 5 | Search/retraining cycle and controlled ablations. | Search effects can be separated from fine-tuning effects. |
| 6 | Locked evaluation, initial external benchmark integration, analysis, and release artifacts. | Another person can reproduce the reported configuration and results. |

### Weekly deliverables

**Week 1:** baseline result file, failure taxonomy, and evaluation entry point.

**Week 2:** typed executor, trace format, and conformance tests.

**Week 3:** guided-versus-random search results with frozen primitives.

**Week 4:** adapter checkpoints, data manifests, and per-primitive reports.

**Week 5:** ablation table and documented retraining cycle.

**Week 6:** final report, reproducibility instructions, and benchmark manifests. Additional coding and conversational coverage can extend beyond this initial schedule.

A validated null result is informative. Do not change the locked evaluation merely to obtain a positive result.

## 14. Hardware and budget

Use the 48 GB Mac first. Rent GPUs only after profiling identifies a useful bottleneck to move.

Raw BF16 weights for approximately 3.09B parameters occupy:

```text
3.09 billion parameters x 2 bytes = approximately 6.18 GB
```

This is decimal GB for weights only. Activations, caches, optimizer state, temporary allocations, and framework overhead require additional memory.

Start with short sequences and small microbatches. Measure peak usage rather than assuming that a particular training configuration fits.

MLX LM is the proposed local route; Transformers + PEFT is the proposed cloud route. Do not assume adapter formats or numerical behavior are interchangeable without validating conversion and outputs.

### Budget by measurement

```text
Total project cost =
    local compute cost
  + GPU hourly rate x GPU hours
  + CPU, memory, and storage charges
  + teacher-model usage
  + benchmark simulator usage
```

Illustration only: an assumed GPU rate of $2/hour makes 10 GPU-hours $20 before additional charges. Check actual provider prices before spending.

Search may cost more than adapter training. For example:

```text
100 programs
x 100 examples
x 3 model calls per execution
x 500 generated tokens per call
= 15,000,000 generated tokens before caching
```

Profile a small search run before increasing candidate counts, task counts, or program depth. Set explicit cost and execution limits.

## 15. Planned repository structure

```text
plvr-3b/
|-- README.md
|-- assumptions.md
|-- configs/
|   |-- model.yaml
|   |-- search.yaml
|   |-- training.yaml
|   `-- evaluation.yaml
|-- data/
|   |-- generators/
|   |-- manifests/
|   `-- raw/
|-- src/
|   `-- plvr3b/
|       |-- model.py
|       |-- types.py
|       |-- primitives/
|       |-- contracts.py
|       |-- executor.py
|       |-- search.py
|       |-- scoring.py
|       |-- cache.py
|       `-- training.py
|-- scripts/
|   |-- run_baseline.py
|   |-- run_search.py
|   |-- train_primitive.py
|   `-- evaluate.py
|-- tests/
|   |-- test_types.py
|   |-- test_contracts.py
|   |-- test_executor.py
|   `-- test_search.py
`-- experiments/
```

These paths are a proposed scaffold, not a claim that runnable scripts already exist. Add installation and execution commands after implementing and testing the entry points.

Pin dependencies and save the validated environment configuration. Keep generated datasets, checkpoints, credentials, and large traces out of Git unless intentionally published with appropriate permissions.

### Record in `assumptions.md`

- Target paper version and relevant sections.
- Author implementation availability and revision, when applicable.
- Exact primitive signatures, operator semantics, and input parser behavior.
- Loss definitions, candidate ranking, object alignment, and tie-breaking.
- Search limits, random sampling procedure, and stopping rules.
- Dataset construction, split rules, and contamination checks.
- Shared-backbone and adapter differences from the original design.
- Evaluation exclusions, failure handling, and unresolved ambiguities.

## 16. First implementation checklist

- [ ] Load Qwen locally and generate one structured mock tool call.
- [ ] Define the first mock tool schemas and deterministic tool implementations.
- [ ] Create a 30-example smoke-test set separate from locked evaluation.
- [ ] Parse outputs and distinguish formatting, contract, and semantic failures.
- [ ] Save a complete trace for every example.
- [ ] Implement typed execution with oracle primitives.
- [ ] Add backward search and the random-search control.
- [ ] Only then generate larger datasets and fine-tune adapters.

**First milestone:** run Qwen on 30 mock tool-calling examples, validate its outputs, and save enough evidence to explain every success and failure.

## 17. Definition of done

The initial study is complete when:

- [ ] The runtime and search pass oracle-based correctness tests.
- [ ] Model, data, prompts, adapters, and evaluation revisions are pinned.
- [ ] A reusable program is selected without accessing held-out answers.
- [ ] Fixed-workflow, random-search, guided-search, and budget-matched baselines are evaluated.
- [ ] Primitive training and program search contributions are reported separately.
- [ ] Failures, uncertainty, latency, token usage, memory, and cost are included.
- [ ] Reproduction differences and limitations are explicit.
- [ ] Another person can rerun the documented experiment.

The goal is not merely to claim that Qwen scored higher. It is to establish **which gains came from specialization, which came from program search, what they cost, and where a shared 3B backbone remains insufficient**.

## References

- [Target PLVR paper, arXiv v1](https://arxiv.org/html/2608.28421v1)
- [Qwen2.5-3B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [PEFT model and adapter documentation](https://huggingface.co/docs/peft/package_reference/peft_model)
- [MLX LM](https://github.com/ml-explore/mlx-lm)
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)
- [tau2-bench](https://github.com/sierra-research/tau2-bench)
- [Modal pricing](https://modal.com/pricing)