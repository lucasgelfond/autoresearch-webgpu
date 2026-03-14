# autoresearch-webgpu

Browser-based autonomous ML research, powered by jax-js and WebGPU.

A user opens a webpage, clicks "Start", and watches an LLM researcher (Claude)
run experiments on a small GPT model — training, evaluating, tweaking
hyperparameters, and improving val_bpb — all live in the browser on their M4.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Browser Tab                     │
│                                                  │
│  ┌────────────┐  ┌───────────────────────────┐  │
│  │  Dashboard  │  │   WebGPU Training Engine  │  │
│  │            │  │                           │  │
│  │ loss curve  │  │  jax-js GPT model         │  │
│  │ experiment  │  │  adamw optimizer          │  │
│  │ history     │  │  ~1-3M params             │  │
│  │ config diff │  │  256-tok context          │  │
│  │ LLM log     │  │  30-60s per run           │  │
│  │ text sample │  │                           │  │
│  └──────┬─────┘  └────────────┬──────────────┘  │
│         │                     │                  │
│  ┌──────┴─────────────────────┴──────────────┐  │
│  │         Research Controller                │  │
│  │                                            │  │
│  │  Claude API → config JSON → train → eval  │  │
│  │  keep best / revert → repeat              │  │
│  └────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Constraints

- **Target hardware**: Apple M4 (unified memory, Metal via WebGPU)
- **No server-side compute**: all training happens client-side
- **Model size**: ~1-3M parameters (what trains meaningfully in 30-60s on M4)
- **Context length**: 256 tokens (no Flash Attention → O(n²) memory)
- **Vocab size**: 2048 tokens (small BPE, shipped as static asset)
- **Training data**: ~10-50MB pre-tokenized text, shipped as static .bin files
- **LLM calls**: Claude API from a thin server endpoint (keeps API key safe)

## Simplifications vs. real autoresearch

| Real autoresearch           | This project                         |
|-----------------------------|--------------------------------------|
| 50M params on H100          | 1-3M params on M4 WebGPU             |
| Muon + AdamW optimizer      | AdamW only (from @jax-js/optax)      |
| Flash Attention 3           | nn.dotProductAttention (standard)    |
| 2048 context, 8192 vocab    | 256 context, 2048 vocab              |
| bfloat16 mixed precision    | float32 (f16 if stable on Metal)     |
| Value residuals, ResFormer  | Dropped (marginal at small scale)    |
| Sliding window attention    | Full causal attention                |
| LLM modifies Python source  | LLM returns config JSON              |
| Per-layer scaling (λ)       | Standard residual connections        |
| 5-min training runs         | 30-60s training runs                 |

## What we keep from autoresearch

- RoPE positional embeddings
- RMSNorm (via nn.standardize)
- Pre-norm transformer blocks
- relu² MLP activation
- Logit softcap (15 * tanh(x/15))
- Cross-entropy loss → bits-per-byte evaluation
- The core research loop: train → eval → keep/revert

## Config-driven research

Instead of the LLM modifying source code, it returns a config object:

```typescript
type ExperimentConfig = {
  // architecture
  nLayer: number       // 2-8
  nEmbd: number        // 64-256
  nHead: number        // 2-8
  mlpRatio: number     // 2-6
  activation: 'relu_sq' | 'gelu' | 'silu'
  useRoPE: boolean
  softcapValue: number // 0 = disabled, 15 = default

  // optimization
  lr: number
  weightDecay: number
  warmupRatio: number
  cooldownRatio: number
  batchSize: number    // 4-32
  seqLen: number       // 64-256

  // training
  trainSeconds: number // 30-90
}
```

## Tech stack

- **SvelteKit** — UI framework (matches jax-js website)
- **@jax-js/jax** + **@jax-js/optax** — training engine
- **Claude API** — research agent (called from server endpoint)
- **D3 / Observable Plot** — loss curves and charts
- **Tailwind CSS** — styling

---

## PR plan

Each PR builds on the previous. Ordered for incremental reviewability.

### PR 1: `feat(scaffold): sveltekit project with jax-js`

Set up the project skeleton.

- `pnpm create svelte@latest` with TypeScript
- Add dependencies: `@jax-js/jax`, `@jax-js/optax`, `tailwindcss`
- Single page at `/` with WebGPU initialization check
- Show device info (adapter name, limits) or "WebGPU not supported" message
- Basic layout shell with header

**Files:**
```
package.json
svelte.config.js
src/app.html
src/app.css
src/routes/+layout.svelte
src/routes/+page.svelte
src/lib/webgpu.ts          # init() wrapper, device check
```

### PR 2: `feat(model): GPT model and forward pass`

Implement the config-driven GPT in jax-js.

- `ExperimentConfig` type definition
- `DEFAULT_CONFIG` with sensible defaults for M4
- `initParams(config, key)` → flat params dict
- `forward(params, config, inputIds)` → logits
- Components: token embedding, RoPE, RMSNorm, causal attention
  (via `nn.dotProductAttention`), relu² MLP, softcap
- `lossFn(params, config, inputIds, targetIds)` → scalar loss
- Simple smoke test: random input → forward pass → loss is finite

**Files:**
```
src/lib/model/config.ts     # ExperimentConfig type + DEFAULT_CONFIG
src/lib/model/gpt.ts        # initParams, forward, lossFn
```

### PR 3: `feat(data): tokenizer and data loading`

Ship a small dataset for in-browser training.

- Script (`scripts/prepare-data.ts`) that:
  - Takes a text file, trains a small BPE tokenizer (2048 vocab)
  - Tokenizes text into fixed-length chunks
  - Writes `static/data/tokens.bin` (Uint16Array) and
    `static/data/tokenizer.json` (vocab + merges)
- Browser-side loader: fetch .bin, wrap as batched iterator
- `makeBatches(tokens, batchSize, seqLen)` → yields `{input, target}` pairs
- `Tokenizer` class for encode/decode (needed for text sampling)
- Ship a ~10MB default dataset (e.g., tiny subset of OpenWebText)

**Files:**
```
scripts/prepare-data.ts
src/lib/data/loader.ts      # fetch + batch iterator
src/lib/data/tokenizer.ts   # BPE encode/decode
static/data/tokens.bin      # pre-tokenized training data
static/data/tokenizer.json  # vocabulary
```

### PR 4: `feat(train): training loop with live metrics`

Wire up forward pass + optimizer into a training loop.

- `trainRun(config, data, callbacks)` — the core training function
  - Initialize params with `initParams`
  - Create AdamW optimizer from `@jax-js/optax`
  - LR schedule: linear warmup → constant → linear cooldown
  - Training loop: forward → grad → optimizer step
  - Call `callbacks.onStep({step, loss, elapsed})` each step
  - Call `callbacks.onDone({valBpb, totalSteps, elapsed})` at end
- `evaluateBpb(params, config, valData)` — bits-per-byte metric
- Runs for `config.trainSeconds` wall-clock seconds, then evaluates
- Yields to browser event loop periodically (requestAnimationFrame)

**Files:**
```
src/lib/train/loop.ts       # trainRun, evaluateBpb
src/lib/train/schedule.ts   # LR schedule helpers
```

### PR 5: `feat(ui): dashboard with live loss chart`

Build the training dashboard UI.

- "Start Training" button that runs a single experiment
- Live loss curve (updates every step via callbacks)
- Config editor panel (JSON or form inputs)
- Training status: step count, elapsed time, tokens/sec
- Val BPB display after training completes
- Text sampling: generate tokens from trained model, show output
- Use Observable Plot (or a simple canvas chart) for loss curve

**Files:**
```
src/routes/+page.svelte          # main dashboard
src/lib/components/LossChart.svelte
src/lib/components/ConfigEditor.svelte
src/lib/components/TrainingStatus.svelte
src/lib/components/TextSample.svelte
src/lib/sample.ts                # greedy/temperature sampling
```

### PR 6: `feat(research): autonomous experiment loop with Claude`

The research controller — the "auto" in autoresearch.

- Server endpoint `POST /api/research` that proxies to Claude API
- System prompt describing the experiment setup, current results history,
  and instructions to return a new `ExperimentConfig` JSON
- `ResearchController` class:
  1. Maintain experiment history: `{config, valBpb, reasoning}[]`
  2. Call Claude with history → receive new config + reasoning
  3. Run training with new config
  4. Compare val_bpb to current best
  5. Keep or discard (no git — just track best config in memory)
  6. Loop
- "Start Research" button (vs manual "Start Training")
- Research log panel showing Claude's reasoning for each experiment
- Config diff view: highlight what changed between experiments

**Files:**
```
src/routes/api/research/+server.ts  # Claude API proxy
src/lib/research/controller.ts      # ResearchController
src/lib/research/prompt.ts          # system prompt + formatting
src/lib/components/ResearchLog.svelte
src/lib/components/ConfigDiff.svelte
```

### PR 7: `feat(ui): experiment history and leaderboard`

Polish the research experience.

- Experiment timeline: scrollable list of all runs
- Each entry shows: config diff, val_bpb, duration, keep/discard badge
- Leaderboard: rank experiments by val_bpb (lower = better)
- Click any experiment to see its full config + loss curve
- Persistent storage: save experiment history to IndexedDB
- Resume research across page reloads
- "Export results" button (download as JSON/TSV)

**Files:**
```
src/lib/components/ExperimentTimeline.svelte
src/lib/components/Leaderboard.svelte
src/lib/storage.ts                  # IndexedDB persistence
```

---

## Open questions

1. **Dataset**: What text corpus to ship? Needs to be small (~10MB tokenized),
   public domain, and interesting enough that BPB differences are meaningful.
   Candidates: TinyStories, a subset of Wikipedia, or Project Gutenberg.

2. **f16 on Metal**: jax-js supports f16 on WebGPU, but Metal's f16 behavior
   may have quirks. Need to test early whether f16 training is stable on M4
   or if we should stick with f32.

3. **Training speed**: Need to benchmark early (PR 4) to calibrate model size.
   If a 1M-param model trains too slowly, we may need to go smaller. If it's
   fast, we can go bigger. The config-driven approach makes this easy to adjust.

4. **jax-js missing ops**: No `lax.scan` — we unroll transformer layers in a JS
   for-loop (fine for 2-8 layers). No scatter — we avoid ops that need it and
   use gather + masking instead. These are non-blocking.

5. **Memory management**: jax-js uses manual `.ref`/`.dispose()` instead of
   GC. The training loop needs careful reference tracking to avoid leaks across
   many iterations. This is the trickiest implementation detail.
