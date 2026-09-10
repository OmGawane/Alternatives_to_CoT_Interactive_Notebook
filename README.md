# Beyond the Scratchpad: Alternatives to Chain-of-Thought Reasoning

**DataForge 2026 — Pathway Track submission.** An interactive Jupyter notebook exploring why Chain-of-Thought (CoT) reasoning is expensive by architecture, not by necessity — and how a fixed-size recurrent latent-state design in the spirit of Pathway's **BDH-CQ** sidesteps that cost, illustrated on a small, fully-verified toy task.

## The problem

Transformer LLMs "show their work" by generating long strings of natural-language reasoning tokens. That verbosity isn't free:

| Bottleneck | Cause | Consequence |
|---|---|---|
| Quadratic attention | Every new token attends to all prior tokens — O(N²) | KV cache balloons, memory pressure grows |
| Serial latency | Tokens are generated one at a time, autoregressively | Hardware idles at each token boundary |
| Token-tax cost | Every reasoning step = more tokens = more $ | Cost scales with verbosity, not problem difficulty |

Pathway's BDH-CQ argues these costs are architectural, not fundamental to intelligence: it reasons via iterative computation inside a fixed-size recurrent latent state, updated with local Hebbian-style rules, and only decodes a final answer. On the public ARC-AGI-1 benchmark, Pathway reports BDH-CQ scoring 29.5% (pass@2) at ≈$0.0007/task — about 11× cheaper per task than a comparison model that scored higher, 34.2%.

## What this notebook actually does

This isn't a slide deck dressed up as code. Everything in it is checkable:

1. **Tokenization** — a small dependency-free BPE tokenizer, trained live in the notebook, used to tokenize real text (not a stand-in for the rest of the demo — its output feeds directly into the cost accounting later).
2. **A real, verifiable toy task** — a grid-reflection puzzle (infer a rule from one example, apply it to a new grid). Small and deterministic on purpose, so correctness can be checked with `np.array_equal`, not asserted.
3. **Two solvers, same task:**
   - **Solver A (CoT-style):** verbalizes a full cell-by-cell reasoning trace in natural language, then tokenizes it — a real, growing token count, not a placeholder.
   - **Solver B (BDH-CQ-style):** represents the grid as a fixed-size latent vector and applies the rule as a single direct operation on that state — no verbalized trace.
   - Both solvers' answers are verified correct **before** any cost is computed.
4. **An auditable, shared cost model** — every constant is labeled:
   - **DERIVED** — computed from a stated, standard transformer-inference formula
   - **ASSUMPTION** — an editable parameter (hardware throughput, $/GPU-hour, fixed call overhead); change it and the numbers change accordingly
   - **CITED** — taken directly from Pathway's published ARC-AGI-1 results, not computed by this notebook

   The same formula and assumptions are applied to *both* solvers, so any gap in the results comes from the shape of the work (growing token trace vs. fixed-size state update), not from a friendlier constant chosen for one side.
5. **A scaling sweep** across grid sizes, re-verifying correctness at every size before plotting cost/latency.
6. **Grounding in Pathway's real published numbers** (clearly separated from this notebook's own toy-task results).
7. **An explicit "Honest Limitations" section** — stating plainly that the toy task is trivial, Solver B is an analogy rather than a reimplementation of BDH-CQ's actual neuron/synapse formalism, the hardware/pricing constants are illustrative assumptions, and that Pathway's own benchmark shows a real accuracy tradeoff (29.5% vs 34.2%), not a strictly-dominant result.
8. **An interactive playground** to pick a grid size and see both solvers run live, side by side.

## Why the labeling system

A demo that only shows numbers favoring its own conclusion isn't evidence, it's marketing. Every quantity here is traceable to either a formula you can read, a parameter you can edit, or a citation you can check — so you can disagree with an assumption and see exactly how the conclusion moves.

## Running it

No special setup is required — the notebook is dependency-light by design.

```bash
pip install jupyter numpy matplotlib
# optional, for live interactive widgets instead of re-run-the-cell interactivity:
pip install ipywidgets pandas

jupyter notebook BDH_CQ_CoT_Alternatives_Demo_final.ipynb
```

If `ipywidgets` isn't available, the notebook automatically falls back to plain function calls (e.g. `run_comparison(grid_size=10)`) — everything still runs, you just edit-and-rerun a cell instead of dragging a slider.

## Structure

| # | Section |
|---|---|
| 1 | Tokenization — a real, from-scratch BPE tokenizer |
| 2 | The toy task — grid-reflection puzzle |
| 3 | Solver A — verbalized, cell-by-cell CoT reasoning |
| 4 | Solver B — fixed-size latent iterative reasoning |
| 5 | Honest cost accounting — shared, auditable formulas |
| 6 | Scaling sweep — where the cost gap comes from |
| 7 | Grounding in Pathway's real ARC-AGI-1 result |
| 8 | Honest limitations |
| 9 | Interactive playground |
| 10 | Key takeaways & references |

## Key takeaway

Verbalizing reasoning as natural-language tokens imposes a real, formula-derivable cost tax that a fixed-size latent-state update does not pay — that mechanism is demonstrated and auditable here. Whether that architectural choice generalizes to genuinely hard, open-ended reasoning — and what accuracy it costs to get there — is an empirical question that Pathway's own published benchmark, not this notebook, is the actual evidence for.

## References

- Pathway, *"Introducing BDH-CQ"* — pathway.com/introducing-bdh-cq
- Pathway, *"The Missing Link Between the Transformer and Models of the Brain"* (BDH architecture paper)
- Pathway press release: *"Pathway's 150M-Parameter Model Breaks the ARC-AGI-1 Cost-Efficiency Frontier"*
- ARC-AGI-1 public benchmark

---

*Built for DataForge 2026 — Pathway Track.*
