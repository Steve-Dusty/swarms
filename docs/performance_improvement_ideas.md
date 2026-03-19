# Swarms Multi-Agent Architecture: Performance Improvement Ideas

A comprehensive catalog of performance, reliability, and feature improvement ideas for the core multi-agent structures in the Swarms framework.

---

## Table of Contents

- [SequentialWorkflow](#sequentialworkflow)
- [ConcurrentWorkflow](#concurrentworkflow)
- [HierarchicalSwarm](#hierarchicalswarm)
- [AgentRearrange](#agentrearrange)
- [Cross-Cutting Improvements](#cross-cutting-improvements)

---

## SequentialWorkflow

**File:** `swarms/structs/sequential_workflow.py`
**Engine:** `swarms/structs/agent_rearrange.py`

### Context & Memory

| # | Improvement | Description |
|---|------------|-------------|
| 1 | **Sliding window context** | Instead of passing the full conversation history to every agent via `get_str()`, pass only the last N messages or a summary. Token usage currently grows quadratically — agent 5 sees all prior outputs. |
| 2 | **Context summarization between steps** | Insert an automatic summarizer agent that compresses prior outputs before handing off, reducing token bloat in long chains. |
| 3 | **Cache `get_str()` output** | The conversation is serialized to string on every agent call. Cache the result and only rebuild the delta. *(GitHub Issue: [#1460](https://github.com/kyegomez/swarms/issues/1460))* |

### Execution

| # | Improvement | Description |
|---|------------|-------------|
| 4 | **Pipeline parallelism** | When running batched tasks, agent 2 could start on task 1's output while agent 1 processes task 2. Currently tasks are fully serialized. |
| 5 | **Async-native execution** | `run_async` currently wraps synchronous `run()` in `asyncio.to_thread`. A true async path with `await agent.arun()` would reduce thread overhead. |
| 6 | **Early termination** | Add a confidence/quality check between steps so the chain can stop early if output is already sufficient (e.g., skip the reviewer if the writer's output scores high). |

### Overhead Reduction

| # | Improvement | Description |
|---|------------|-------------|
| 7 | **Cache flow validation** | `validate_flow()` runs on every call with O(n) agent name checks. Parse and validate once at init, invalidate only on flow change. *(GitHub Issue: [#1461](https://github.com/kyegomez/swarms/issues/1461))* |
| 8 | **Lazy output formatting** | `history_output_formatter()` processes the entire history at the end. For `"final"` output type, skip serializing everything and just return the last message. |
| 9 | **Eliminate redundant `any_to_str()` calls** | Every agent response goes through `any_to_str()` conversion even when it's already a string. *(GitHub Issue: [#1462](https://github.com/kyegomez/swarms/issues/1462))* |

### Reliability

| # | Improvement | Description |
|---|------------|-------------|
| 10 | **Checkpoint and resume** | Save conversation state after each agent step so a failed workflow can resume from the last successful agent instead of restarting from scratch. |

---

## ConcurrentWorkflow

**File:** `swarms/structs/concurrent_workflow.py`

### Thread & Execution Model

| # | Improvement | Description |
|---|------------|-------------|
| 1 | **Async with `uvloop`/`asyncio`** | Replace `ThreadPoolExecutor` with native async execution. The codebase already has `run_agents_concurrently_uvloop` but ConcurrentWorkflow doesn't use it. LLM API calls are I/O-bound, making async ideal. |
| 2 | **Adaptive worker pool sizing** | Currently hardcoded at 95% of CPU cores. For 3 agents, that's wasteful; for 200 agents, it's a bottleneck. Scale workers to `min(agent_count, cpu_cores)`. |
| 3 | **Connection pooling per LLM provider** | Multiple agents hitting the same API endpoint should share HTTP connections to reduce TLS handshake overhead. |

### Result Handling

| # | Improvement | Description |
|---|------------|-------------|
| 4 | **Stream-first results** | Use `as_completed()` in dashboard mode too (currently uses `wait()` which blocks until the slowest agent finishes). Return partial results as each agent completes. |
| 5 | **Result streaming to disk** | For large outputs from many agents, stream results to disk instead of holding everything in memory. Current approach holds all outputs in RAM. |
| 6 | **Parallel result aggregation** | Adding results to the conversation happens sequentially after all agents finish. Add them as each future completes. |

### Dashboard & Monitoring

| # | Improvement | Description |
|---|------------|-------------|
| 7 | **Reduce dashboard refresh overhead** | The 0.1s throttle on dashboard updates causes ~100+ dict mutations/sec on large swarms. Batch status updates and refresh at 1-2 FPS instead. |
| 8 | **Decouple dashboard from execution thread** | Run dashboard rendering in a separate thread so it never blocks agent execution. |

### Batching

| # | Improvement | Description |
|---|------------|-------------|
| 9 | **Parallel batch execution** | `batch_run()` processes tasks sequentially, running agents concurrently only within each task. Process multiple tasks concurrently too. |
| 10 | **Rate-limit-aware scheduling** | Add API rate limit awareness so the executor can throttle submissions per provider instead of hitting 429s and retrying. |

### Error Handling

| # | Improvement | Description |
|---|------------|-------------|
| 11 | **Timeout per agent** | No per-agent timeout exists. A single hanging agent blocks the entire workflow forever. Add configurable per-agent timeouts with `future.result(timeout=X)`. |
| 12 | **Retry with backoff** | Failed agents (especially on transient API errors) should retry with exponential backoff before reporting failure. |

---

## HierarchicalSwarm

**File:** `swarms/structs/hiearchical_swarm.py`

### Director Optimization

| # | Improvement | Description |
|---|------------|-------------|
| 1 | **Lightweight director model** | The director just assigns tasks; it doesn't need the same heavy model as workers. Use a faster/cheaper model (e.g., Haiku) for task distribution. |
| 2 | **Skip planning phase when unnecessary** | `planning_enabled=True` adds a full extra LLM call. For simple tasks or single-loop runs, skip it. |
| 3 | **Cache director decisions** | For repeated similar tasks, cache the director's task distribution plan and reuse it. |
| 4 | **Batch director calls across loops** | In multi-loop mode, the director could plan multiple iterations ahead instead of one loop at a time. |

### Agent Execution

| # | Improvement | Description |
|---|------------|-------------|
| 5 | **O(1) agent lookup** | `call_single_agent` does O(n) linear search for agents by name. Use a dict keyed by `agent_name` (the sequential/concurrent workflows already do this). *(GitHub Issue: [#1458](https://github.com/kyegomez/swarms/issues/1458))* |
| 6 | **Parallel execution by default** | Sequential is the default but most hierarchical tasks are independent. Make parallel the default and fall back to sequential only when dependencies exist. *(GitHub Issue: [#1458](https://github.com/kyegomez/swarms/issues/1458))* |
| 7 | **Dependency-aware scheduling** | Add a DAG-based scheduler so the director can specify dependencies between orders. Independent tasks run in parallel; dependent ones wait. |
| 8 | **Per-agent timeout and cancellation** | In parallel mode, `as_completed()` waits for all futures. Add timeouts so a slow agent doesn't block the whole swarm. |

### Feedback Loop

| # | Improvement | Description |
|---|------------|-------------|
| 9 | **Conditional feedback** | Run the judge/feedback agent only when output quality is uncertain. If all agent outputs exceed a quality threshold, skip the feedback loop and save an LLM call. |
| 10 | **Incremental feedback** | Instead of reviewing all outputs at once, provide feedback as each agent completes. This lets later agents benefit from earlier feedback within the same loop. |
| 11 | **Reduce feedback agent context** | The feedback director receives the full conversation. Pass only the current loop's outputs + the original task. |

### Context Management

| # | Improvement | Description |
|---|------------|-------------|
| 12 | **Conversation pruning between loops** | History grows with every loop. Summarize or prune earlier loops' outputs to stay within token limits. Currently risks exceeding context windows on multi-loop runs. |
| 13 | **Selective context per agent** | Not every worker needs the full conversation. Pass only the director's order + relevant prior outputs instead of `conversation.get_str()`. |

### Structural

| # | Improvement | Description |
|---|------------|-------------|
| 14 | **Sub-hierarchies** | Allow a worker agent to itself be a HierarchicalSwarm, enabling tree-structured task decomposition for complex problems. |
| 15 | **Agent pool with auto-selection** | Instead of the director manually naming agents, let it describe the capability needed and auto-match to the best available agent. |

---

## AgentRearrange

**File:** `swarms/structs/agent_rearrange.py`

### Context & Token Efficiency

| # | Improvement | Description |
|---|------------|-------------|
| 1 | **Quadratic token growth** | Every agent receives the full conversation via `get_str()` (lines 509, 572). With 5 agents each producing 1K tokens, agent 5 processes ~5K input tokens. Add a sliding window or summary mode that compresses prior outputs. |
| 2 | **Cache `get_str()` result** | `conversation.get_str()` rebuilds the full string on every call. Cache it and invalidate only when new messages are added. *(GitHub Issue: [#1460](https://github.com/kyegomez/swarms/issues/1460))* |
| 3 | **Selective context per agent** | Not every agent needs all prior outputs. Allow the flow syntax to specify which upstream outputs an agent should see (e.g., `agent3[agent1]` = only see agent1's output). |

### Validation & Parsing Overhead

| # | Improvement | Description |
|---|------------|-------------|
| 4 | **Parse flow once, not per-run** | `validate_flow()` and `self.flow.split("->")` re-parse the flow string on every `_run()` call. Parse it once at init into a structured execution plan. *(GitHub Issue: [#1461](https://github.com/kyegomez/swarms/issues/1461))* |
| 5 | **Drop redundant `any_to_str()` calls** | Line 577 runs `any_to_str(current_task)` on every agent response. If `agent.run()` already returns a string, this is wasted work. Check type first. *(GitHub Issue: [#1462](https://github.com/kyegomez/swarms/issues/1462))* |

### Execution Model

| # | Improvement | Description |
|---|------------|-------------|
| 6 | **True async execution** | `run_async` just wraps the synchronous `run()` in `asyncio.to_thread`. A native async path using `await agent.arun()` would avoid thread pool overhead and scale better. |
| 7 | **Pipeline parallelism for batch_run** | `batch_run` processes tasks sequentially in a list comprehension. Agent 2 could start on task 1 while agent 1 works on task 2. *(GitHub Issue: [#1463](https://github.com/kyegomez/swarms/issues/1463))* |
| 8 | **batch_run should use ThreadPoolExecutor** | The comment says "process batch using concurrent execution" but it's actually a sequential list comprehension. *(GitHub Issue: [#1463](https://github.com/kyegomez/swarms/issues/1463))* |

### Error Handling & Resilience

| # | Improvement | Description |
|---|------------|-------------|
| 9 | **`_run()` swallows exceptions** | `_catch_error` catches the exception, logs it, and returns it — but `_run` doesn't re-raise. A failed workflow silently returns `None`. *(GitHub Issue: [#1464](https://github.com/kyegomez/swarms/issues/1464))* |
| 10 | **`run()` double-logs on failure** | `run()` calls `_run()` which calls `_catch_error()`, then `run()` also calls `_catch_error()` again. Two telemetry writes + two log entries for the same error. *(GitHub Issue: [#1465](https://github.com/kyegomez/swarms/issues/1465))* |
| 11 | **No per-agent timeout** | A single hanging agent blocks the entire workflow forever. Add a configurable timeout per agent step. |
| 12 | **Checkpoint & resume** | If agent 4 of 5 fails, the entire workflow restarts. Save conversation state after each step so execution can resume from the last successful agent. |

### Telemetry Overhead

| # | Improvement | Description |
|---|------------|-------------|
| 13 | **`to_dict()` called twice per run** | `run()` calls `log_agent_data(self.to_dict())` both before and after execution. `to_dict()` serializes the entire object including all agents. Move telemetry to a background thread or make it opt-in. |
| 14 | **`to_dict()` is expensive** | It iterates all `__dict__` items and attempts `json.dumps()` on each to test serializability. This is O(n) JSON serialization attempts per attribute, called twice per run. |

### Concurrency Safety

| # | Improvement | Description |
|---|------------|-------------|
| 15 | **Race condition in concurrent workflow** | `_run_concurrent_workflow` has agents read `self.conversation.get_str()` and write results back to `self.conversation` without locking. Add a lock or snapshot the conversation before concurrent execution. |
| 16 | **Shared conversation across `concurrent_run`** | `concurrent_run()` runs multiple tasks on the same instance sharing `self.conversation`. Parallel tasks corrupt each other's history. Each task should get its own conversation copy. |

### Flow Syntax & Flexibility

| # | Improvement | Description |
|---|------------|-------------|
| 17 | **Support conditional branching** | Add support for conditional routing (e.g., `agent1 -> agent2 ? agent3` where the choice depends on agent1's output). |
| 18 | **Support agent repetition in flow** | Formally support loops like `agent1 -> agent2 -> agent1` for iterative refinement patterns. *(GitHub Issue: [#1466](https://github.com/kyegomez/swarms/issues/1466))* |
| 19 | **Weighted concurrent execution** | When agents run concurrently, there's no priority. Allow weighting so critical agents get thread priority or earlier result collection. |

---

## Cross-Cutting Improvements

Improvements that apply to all multi-agent architectures.

| # | Area | Improvement | Description |
|---|------|------------|-------------|
| 1 | **Observability** | OpenTelemetry tracing | Add tracing spans per agent call for latency profiling across all swarm types. |
| 2 | **Token Budgeting** | Per-agent token limits | Set per-agent token limits to prevent one agent from consuming the entire context window. |
| 3 | **Caching** | LLM response cache | Cache identical LLM calls (same prompt = same result) across agents and runs. |
| 4 | **Warm-up** | Pre-initialize connections | Pre-initialize agent connections/models before workflow starts to avoid cold-start latency on the first agent. |
| 5 | **Serialization** | Structured message objects | Use structured message objects instead of string concatenation for inter-agent communication. |

---

## GitHub Issues Tracking

| Issue | Title | Architecture |
|-------|-------|-------------|
| [#1458](https://github.com/kyegomez/swarms/issues/1458) | O(1) agent lookup + parallel execution by default | HierarchicalSwarm |
| [#1460](https://github.com/kyegomez/swarms/issues/1460) | Cache `Conversation.get_str()` with invalidation | AgentRearrange / All |
| [#1461](https://github.com/kyegomez/swarms/issues/1461) | Parse flow once at init, not per-run | AgentRearrange |
| [#1462](https://github.com/kyegomez/swarms/issues/1462) | Drop redundant `any_to_str()` calls | AgentRearrange |
| [#1463](https://github.com/kyegomez/swarms/issues/1463) | Pipeline parallelism + fix `batch_run` ThreadPoolExecutor | AgentRearrange |
| [#1464](https://github.com/kyegomez/swarms/issues/1464) | `_run()` silently swallows exceptions | AgentRearrange |
| [#1465](https://github.com/kyegomez/swarms/issues/1465) | `run()` double-logs errors on failure | AgentRearrange |
| [#1466](https://github.com/kyegomez/swarms/issues/1466) | Support agent repetition in flow | AgentRearrange |
