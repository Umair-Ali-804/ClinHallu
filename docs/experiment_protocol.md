# Experiment protocol

## Adaptation study

| Condition | Training rows | Validation rows | Frozen test rows |
|---|---:|---:|---:|
| C1 | 10,000 | 4,000 | 1,000 |
| C2 | 11,000 | 4,000 | 1,000 |
| C3 | 12,000 | 4,000 | 1,000 |
| C4 | 13,000 | 4,000 | 1,000 |
| C5 | 14,000 | 4,000 | 1,000 |

Use seed 13 consistently unless the registered study protocol is deliberately
expanded. Each condition uses its matching YAML file.

## Ablations

B through F are cumulative, structurally distinct models. Each variant must be
trained from its own initialization and must not reuse full-model GAER head
weights. They may reuse the same Hugging Face base-model download cache.

```bash
clinhallu ablate --variants B C D E F --seed 13
```

Run state is stored in `run_state.json`; incomplete training resumes from
`latest_checkpoint.pt` automatically.
