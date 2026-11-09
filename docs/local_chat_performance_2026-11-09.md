# Local chat performance validation — 2026-11-09

The strongest measured improvement is disabling thinking for routine chat and
tool routing on `nemotron-cascade-2:30b`: median latency fell from 4.73 seconds
to 1.90 seconds in this small test. This does not establish equivalent quality
for complex code, scientific interpretation, or multi-step workflows.

## Environment and method

- NVIDIA GB10; Ollama 0.20.2; model Q4_K_M, 31.6B total parameters.
- `ollama ps` showed 100% GPU residency and a 262,144-token context.
- Used the actual 39,631-character tool-routing system prompt and the plugin's
  sampling settings, adding seed 42 for repeatability. Model execution is still
  not fully deterministic, as the results demonstrate.
- Two brief image-analysis questions, each repeated twice per configuration:
  Gaussian versus median denoising, and thresholding uneven illumination.
- Empty viewer, memory, and intent context. Measured the HTTP/model path only;
  did not exercise the GUI, large datasets, or execute generated actions.
- Configurations ran sequentially: default, thinking off, then 16K/thinking off.
  Prefix-cache and loading state therefore differ. Four calls per configuration
  are exploratory evidence, not a production latency distribution.
- Raw results: [local_chat_benchmark_2026-11-09.json](local_chat_benchmark_2026-11-09.json).

## Measurements

| Configuration | Median wall time | Range | Valid action JSON |
| --- | ---: | ---: | ---: |
| Existing defaults, 262K resident context | 4.73 s | 2.70–8.36 s | 4/4 |
| `think: false`, same context | 1.90 s | 1.79–2.09 s | 4/4 |
| `think: false`, `num_ctx: 16384` | 1.94 s | 1.47–10.42 s | 3/4 |

The first default call spent 3.96 seconds processing the prompt; subsequent
calls spent about 0.52 seconds. Excluding that first call, the default median
was 4.46 seconds: thinking off was still about 57% faster. Defaults produced
246–889 hidden reasoning characters per call; thinking off produced none.

Changing to 16K caused a reload: the first call spent 3.51 seconds loading and
5.41 seconds evaluating the prompt. Its subsequent median was 1.92 seconds,
essentially the same as thinking off at the original context. One response was
not parseable as the expected action JSON. This test does not establish that
the smaller context caused the formatting failure, or improves throughput.

## Evidence from real usage

The existing telemetry contains genuine slow requests:

- 2026-11-09: 48.43 seconds wall time, 9,182 input tokens, 461 generated tokens;
  prompt evaluation 5.21 seconds, generation 7.41 seconds, Ollama total 48.38
  seconds. About 35.77 seconds is outside those two measured phases. Loading
  or server scheduling is plausible, but historical telemetry omitted
  `load_duration`, so the cause cannot be established from this event.
- 2026-05-28: 50.90 seconds wall time, 3,257 generated tokens and only 516
  returned content characters; generation alone took 49.44 seconds. This is
  consistent with hidden reasoning dominating the wait, as directly observed
  in the new benchmark.
- Historical model comparisons are uncontrolled. For example, this machine's
  three `qwen2.5:7b` turns had a 2.24-second median, but that sample does not
  establish comparable task accuracy or justify replacing the current model.

## Recommended changes, in priority order

1. Add a fast/reasoning control. For the tested Nemotron model, send top-level
   `think: false` for routine requests. Keep reasoning available for difficult
   coding and analysis. Check model support before applying this to other
   models: some use named levels instead. `think` belongs outside `options`.
2. Record `load_duration`, thinking character count, and `done_reason` in
   production telemetry. Time viewer-context preparation separately. These
   distinguish cold loads, hidden reasoning, long inputs, and truncation.
3. Keep preloading and chat requests consistent in context and residency.
   Preload currently requests `keep_alive: "30m"`; chat does not explicitly set
   it. Explicitly configure the same residency policy for both. Avoid frequent
   model/context switches and distinguish the first request from warm requests.
4. Reduce the routing system prompt through task-specific tool descriptions.
   It currently accounts for almost all input in simple requests. Preserve its
   stable prefix to retain caching, and validate tool selection, follow-ups,
   and code safety before replacing the full prompt. This primarily targets
   uncached prompt processing; warm prompt evaluation was already about 0.52 s.
5. Add streaming for earlier feedback. The client currently uses
   `stream: false`; streaming can improve perceived responsiveness but does
   not eliminate reasoning or generation time. Execute actions only after the
   complete response passes parsing and validation.
6. Consider a bounded output budget per task, with explicit handling for
   `done_reason: "length"`. An aggressive global `num_predict` limit can cut
   off reasoning before the final JSON or truncate generated code.

Do not prioritize GPU replacement or a smaller context based on this test.
The existing hardware already handles these warm routine requests quickly.
No production defaults were changed. The original 262K resident context and
30-minute preload residency were restored after testing.

## Reproduction

Run inside the plugin's Python environment:

```bash
python scripts/benchmark_local_chat.py --model nemotron-cascade-2:30b --output /tmp/default.json
python scripts/benchmark_local_chat.py --model nemotron-cascade-2:30b --think off --output /tmp/fast.json
python scripts/benchmark_local_chat.py --model nemotron-cascade-2:30b --think off --context 16384 --output /tmp/16k.json
```

The context option changes the model's resident context for the test; restore
your original configuration afterward. Output JSON checks here verify a
recognized action field, not semantic correctness or safe code execution.

Ollama documents [model-specific thinking controls](https://docs.ollama.com/capabilities/thinking)
and [context, residency, and GPU inspection](https://docs.ollama.com/faq).
