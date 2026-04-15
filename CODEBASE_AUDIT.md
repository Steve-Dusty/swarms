# Swarms Codebase Audit — Deletion & Cleanup Candidates

> Generated: 2026-04-14
> Scope: `swarms/agents/`, `swarms/structs/`, `swarms/utils/`, `swarms/tools/`, `swarms/prompts/`

---

## 1. DEAD CODE — Safe to Delete (Zero imports outside own file)

These files have no callers anywhere in the codebase, are not exported, and provide no value.

| File | Lines | Contents | Why Delete |
|---|---|---|---|
| `swarms/structs/agent_roles.py` | 35 | Single `Literal` type with ~30 role strings | Never imported, not exported, zero callers |
| `swarms/structs/concat.py` | 28 | `concat_strings()` — wraps `str.join()` | Trivially replaced by stdlib, zero callers |
| `swarms/structs/swarm_id.py` | 6 | `f"swarm-{uuid4().hex}"` wrapper | Zero callers, one-liner not worth a module |
| `swarms/structs/safe_loading.py` | 258 | `SafeLoaderUtils` serialization class | Zero imports anywhere in `swarms/` |
| `swarms/utils/xml_utils.py` | 77 | XML parsing helpers | Only referenced from `history_output_formatter.py` indirectly; no direct callers |
| `swarms/tools/func_to_str.py` | 43 | `function_to_str()`, `functions_to_str()` | Zero imports found anywhere |
| `swarms/tools/function_util.py` | 35 | `process_tool_docs()` | Zero imports found anywhere |
| `swarms/tools/tool_type.py` | 7 | `ToolType` union type | Only used inside `base_tool.py` — inline it there |
| `swarms/prompts/tests.py` | 95 | `TEST_WRITER_SOP_PROMPT()` string | Zero imports, not exported |

---

## 2. EXPERIMENTAL / ABANDONED — Review Before Deleting

These files show signs of being experiments that were never finished or promoted to the public API.

| File | Lines | Contents | Signal |
|---|---|---|---|
| `swarms/structs/swarm_templates.py` | 1351 | Template swarm pattern implementations | Only 1 import (mixture_of_agents.py) — massive file for minimal usage |
| `swarms/structs/various_alt_swarms.py` | 1103 | Alternative swarm implementations | Only 1 import (base_swarm.py) — large unexplained file |
| `swarms/structs/collaborative_utils.py` | 77 | Agent communication helpers | Only 2 imports, not exported |
| `swarms/agents/ape_agent.py` | 36 | `auto_generate_prompt()` | Only used in CLI, not public API |
| `swarms/agents/auto_chat_agent.py` | 52 | REPL-style interactive chat loop | Only used in CLI, appears incomplete |
| `swarms/agents/auto_generate_swarm_config.py` | 447 | Parse markdown YAML into swarm config | Only used in CLI, experimental |

---

## 3. UTILS — Unused or Only Re-exported

| File | Lines | Status |
|---|---|---|
| `swarms/utils/get_cpu_cores.py` | 50 | Only 2 imports; not exported; could be inlined at call sites |
| `swarms/utils/types.py` | 13 | `ReturnTypes` Literal — used in 3-4 files but not exported; should be consolidated into a shared types module or inlined |

---

## 4. PROMPTS — Unused Domain Prompts

The `swarms/prompts/` directory has 60+ files. Most are domain-specific and fine to keep, but these stand out as clearly unused:

| File | Contents | Why Flag |
|---|---|---|
| `swarms/prompts/tests.py` | Test-writing SOP prompt | Zero imports anywhere |
| `swarms/prompts/tools.py` | Tool-use prompts (if present) | Verify import count — likely unused |

> **Recommendation**: Run a bulk grep across all prompt files to confirm which ones have zero callers outside their own file. Any with zero callers and no docs reference are candidates.

---

## 5. LARGE FILES WITH SUSPICIOUSLY LOW USAGE

These are large, self-contained files that only appear once or twice in the codebase — worth investigating whether they're actively maintained or effectively abandoned.

| File | Lines | Import Count | Notes |
|---|---|---|---|
| `swarms/structs/swarm_templates.py` | 1351 | 1 | Single caller in `mixture_of_agents.py` |
| `swarms/structs/various_alt_swarms.py` | 1103 | 1 | Single caller in `base_swarm.py` |
| ~~`swarms/structs/safe_loading.py`~~ | ~~258~~ | ~~0~~ | **RETRACTED** — actively used in `agent.py` (`SafeLoaderUtils`, `SafeStateManager`) |
| `swarms/agents/auto_generate_swarm_config.py` | 447 | 1 | CLI only |

---

## 6. WHAT IS NOT EXPORTED BUT SHOULD BE (Not Deletion — Fix)

These files are heavily used internally but missing from their `__init__.py`. Not deletion candidates — they need to be *added* to exports.

| File | Import Count | Missing From |
|---|---|---|
| `swarms/utils/formatter.py` | 70+ files | `utils/__init__.py` |
| `swarms/utils/loguru_logger.py` | 70+ files | `utils/__init__.py` |
| `swarms/utils/workspace_utils.py` | 15+ files | `utils/__init__.py` |
| `swarms/structs/planner_worker_swarm.py` | multiple examples + docs | `structs/__init__.py` |
| `swarms/structs/tree_swarm.py` | 5+ examples + docs | `structs/__init__.py` |
| `swarms/structs/transforms.py` | docs + examples | `structs/__init__.py` |
| `swarms/structs/deep_discussion.py` | 3 callers + example | `structs/__init__.py` |
| `swarms/structs/image_batch_processor.py` | example + test | `structs/__init__.py` |
| `swarms/structs/csv_to_agent.py` | example file | `structs/__init__.py` |
| `swarms/structs/hierarchical_structured_communication_framework.py` | 16+ matches + 4 docs | `structs/__init__.py` |

---

## Recommended Deletion Order

### Phase 1 — No-risk, delete immediately
```
swarms/structs/agent_roles.py
swarms/structs/concat.py
swarms/structs/swarm_id.py
swarms/tools/func_to_str.py
swarms/tools/function_util.py
swarms/tools/tool_type.py
swarms/prompts/tests.py
```

### Phase 2 — Verify then delete
```
swarms/utils/xml_utils.py          (confirm no callers besides history_output_formatter)
swarms/utils/get_cpu_cores.py      (inline at the 2 call sites)
swarms/utils/types.py              (consolidate into existing types or inline)
swarms/structs/collaborative_utils.py  (confirm only 2 callers, inline logic)
```

### Phase 3 — Investigate before deciding
```
swarms/structs/swarm_templates.py        (1351 lines, 1 import — actively used?)
swarms/structs/various_alt_swarms.py     (1103 lines, 1 import — actively used?)
swarms/agents/ape_agent.py              (CLI only — keep CLI or cut?)
swarms/agents/auto_chat_agent.py        (CLI only — keep CLI or cut?)
swarms/agents/auto_generate_swarm_config.py  (CLI only — keep CLI or cut?)
```

---

## Summary Count

| Category | Files | Est. Lines Removed |
|---|---|---|
| Phase 1 — Safe delete | 8 | ~550 |
| Phase 2 — Verify + delete | 4 | ~175 |
| Phase 3 — Investigate | 5 | ~3,000+ |
| **Total** | **17** | **~3,725** |
