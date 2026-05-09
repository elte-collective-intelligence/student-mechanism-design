# src/configs/eval/

Configuration files for architecture evaluation and ablation studies.

## Files

### `ablation.yaml`

Single source of truth for the architecture ablation experiment. Controls which
architectures, hyperparameter combinations, seeds, and graph sizes are evaluated.
Used by `src/eval/architecture_ablations.py`.

#### Key fields

| Field | Description |
|---|---|
| `architectures` | List of agent types to compare: `gnn`, `gat`, `transformer` |
| `sweep.n_layers` | Layer counts to sweep (GAT / Transformer only) |
| `sweep.hidden_dims` | Hidden dimension sizes to sweep |
| `sweep.n_heads` | Attention head counts to sweep (ignored for `gnn`) |
| `seeds` | Random seeds — must match the seeds used during training |
| `graph_sizes` | Named graph configurations (`small` / `large`) with node/edge counts and police setup |
| `n_eval_episodes` | Episodes to run per checkpoint |
| `checkpoint_dir` | Root directory where trained checkpoints are stored |
| `output_dir` | Directory for ablation results CSV and plots |
| `env.max_steps` | Episode step limit (must match training) |
| `env.seed_offset` | Added to seed when initialising the environment |

#### Checkpoint naming convention

Trained checkpoints **must** follow this convention so the ablation script can find them:

```
{checkpoint_dir}/{arch}_L{n_layers}_H{hidden_dim}_h{n_heads}_s{seed}_{graph_size}/MrX.pt
{checkpoint_dir}/{arch}_L{n_layers}_H{hidden_dim}_h{n_heads}_s{seed}_{graph_size}/Police.pt
```

Examples:
- `src/artifacts/checkpoints/gat_L3_H128_h4_s2_small/MrX.pt`
- `src/artifacts/checkpoints/transformer_L5_H256_h8_s0_large/Police.pt`
- `src/artifacts/checkpoints/gnn_L2_H128_h1_s1_small/MrX.pt`

The GNN baseline uses the fixed key `L2_H128_h1` regardless of sweep values because
`GNNModel` does not expose configurable layers or hidden dimensions.

#### Running the ablation

```bash
# Full grid (slow — run on a machine with checkpoints present)
python src/eval/architecture_ablations.py --config src/configs/eval/ablation.yaml

# Single architecture filter
python src/eval/architecture_ablations.py --config src/configs/eval/ablation.yaml --arch gat

# Dry run (prints grid without loading any checkpoints)
python src/eval/architecture_ablations.py --config src/configs/eval/ablation.yaml --dry_run
```

Results are saved to `{output_dir}/ablation_results.csv`.
