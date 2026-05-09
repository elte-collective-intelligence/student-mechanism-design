#!/usr/bin/env bash
# train_ablation_grid.sh — launch one training run per architecture ablation cell.
#
# Reads src/configs/eval/ablation.yaml for the sweep grid, then trains each
# (arch, n_layers, hidden_dim, n_heads, seed, graph_size) combination and saves
# the final MrX.pt / Police.pt to the ablation checkpoint directory.
#
# Usage:
#   ./scripts/train_ablation_grid.sh                         # full grid
#   ./scripts/train_ablation_grid.sh --arch gat              # single arch
#   ./scripts/train_ablation_grid.sh --arch gnn --size small # arch + graph size
#   ./scripts/train_ablation_grid.sh --dry_run               # print grid only
#
# Flags:
#   --arch <name>      Only train this architecture (gnn, gat, transformer)
#   --size <name>      Only train this graph size (small, large)
#   --seed <n>         Only train this seed
#   --epochs <n>       Override training epochs (default: read from experiment config)
#   --dry_run          Print the run grid without executing anything
#
# Prerequisites:
#   - Virtual environment activated with project dependencies installed
#   - src/configs/eval/ablation.yaml exists
#   - For Docker: run from the container with /app mounted

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
FILTER_ARCH=""
FILTER_SIZE=""
FILTER_SEED=""
EPOCH_OVERRIDE=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)   FILTER_ARCH="$2"; shift 2 ;;
        --size)   FILTER_SIZE="$2"; shift 2 ;;
        --seed)   FILTER_SEED="$2"; shift 2 ;;
        --epochs) EPOCH_OVERRIDE="$2"; shift 2 ;;
        --dry_run) DRY_RUN=true; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve Python interpreter
# ---------------------------------------------------------------------------
if [[ -f "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${PROJECT_ROOT}/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
else
    PYTHON="python3"
fi
echo "Using Python: $PYTHON"

# ---------------------------------------------------------------------------
# Read ablation config via Python (avoids bash YAML parsing)
# ---------------------------------------------------------------------------
ABLATION_CFG="${SRC_DIR}/configs/eval/ablation.yaml"
if [[ ! -f "$ABLATION_CFG" ]]; then
    echo "Error: $ABLATION_CFG not found."
    exit 1
fi

# Read grid values — use a temp Python script to avoid heredoc-in-subshell issues.
_TMP_PY=$(mktemp /tmp/ablation_read_XXXXXX.py)
cat > "$_TMP_PY" << 'EOF'
import yaml, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
gs = cfg["graph_sizes"]
sep = " "
lines = [
    'ARCHITECTURES="' + sep.join(cfg["architectures"]) + '"',
    'N_LAYERS_LIST="' + sep.join(map(str, cfg["sweep"]["n_layers"])) + '"',
    'HIDDEN_DIMS="'   + sep.join(map(str, cfg["sweep"]["hidden_dims"])) + '"',
    'N_HEADS_LIST="'  + sep.join(map(str, cfg["sweep"]["n_heads"])) + '"',
    'SEEDS="'         + sep.join(map(str, cfg["seeds"])) + '"',
    'GRAPH_SIZES="'   + sep.join(gs.keys()) + '"',
    'CHECKPOINT_DIR="' + cfg["checkpoint_dir"] + '"',
    'SMALL_NODES="'   + str(gs["small"]["graph_nodes"]) + '"',
    'SMALL_EDGES="'   + str(gs["small"]["graph_edges"]) + '"',
    'SMALL_POLICE="'  + str(gs["small"]["num_police_agents"]) + '"',
    'SMALL_MONEY="'   + str(gs["small"]["agent_money"]) + '"',
    'LARGE_NODES="'   + str(gs["large"]["graph_nodes"]) + '"',
    'LARGE_EDGES="'   + str(gs["large"]["graph_edges"]) + '"',
    'LARGE_POLICE="'  + str(gs["large"]["num_police_agents"]) + '"',
    'LARGE_MONEY="'   + str(gs["large"]["agent_money"]) + '"',
]
print("\n".join(lines))
EOF
eval "$($PYTHON "$_TMP_PY" "$ABLATION_CFG")"
rm -f "$_TMP_PY"

cd "$PROJECT_ROOT"
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

# Load WandB credentials if available
WANDB_API_KEY=null; WANDB_PROJECT=null; WANDB_ENTITY=null
WANDB_DATA="${SRC_DIR}/wandb_data.json"
if [[ -f "$WANDB_DATA" ]]; then
    WANDB_API_KEY=$($PYTHON -c "import json; d=json.load(open('$WANDB_DATA')); print(d.get('wandb_api_key','') or 'null')" 2>/dev/null)
    WANDB_PROJECT=$($PYTHON -c "import json; d=json.load(open('$WANDB_DATA')); print(d.get('wandb_project','') or 'null')" 2>/dev/null)
    WANDB_ENTITY=$($PYTHON -c "import json; d=json.load(open('$WANDB_DATA')); print(d.get('wandb_entity','') or 'null')" 2>/dev/null)
fi

# ---------------------------------------------------------------------------
# Helper: graph-size params
# ---------------------------------------------------------------------------
graph_size_params() {
    local size="$1"
    if [[ "$size" == "small" ]]; then
        echo "$SMALL_NODES $SMALL_EDGES $SMALL_POLICE $SMALL_MONEY"
    else
        echo "$LARGE_NODES $LARGE_EDGES $LARGE_POLICE $LARGE_MONEY"
    fi
}

# ---------------------------------------------------------------------------
# Count total runs for progress reporting
# ---------------------------------------------------------------------------
total=0
for arch in $ARCHITECTURES; do
    [[ -n "$FILTER_ARCH" && "$arch" != "$FILTER_ARCH" ]] && continue
    for n_layers in $N_LAYERS_LIST; do
        for hidden_dim in $HIDDEN_DIMS; do
            if [[ "$arch" == "gnn" ]]; then heads_iter="1"; else heads_iter="$N_HEADS_LIST"; fi
            for n_heads in $heads_iter; do
                for seed in $SEEDS; do
                    [[ -n "$FILTER_SEED" && "$seed" != "$FILTER_SEED" ]] && continue
                    for size in $GRAPH_SIZES; do
                        [[ -n "$FILTER_SIZE" && "$size" != "$FILTER_SIZE" ]] && continue
                        total=$((total + 1))
                    done
                done
            done
        done
    done
done
echo "Total runs to train: $total"
[[ "$DRY_RUN" == true ]] && echo "(dry run — no training will execute)"

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
run_idx=0
failed=0

for arch in $ARCHITECTURES; do
    [[ -n "$FILTER_ARCH" && "$arch" != "$FILTER_ARCH" ]] && continue

    for n_layers in $N_LAYERS_LIST; do
        for hidden_dim in $HIDDEN_DIMS; do

            if [[ "$arch" == "gnn" ]]; then
                heads_iter="1"
            else
                heads_iter="$N_HEADS_LIST"
            fi

            for n_heads in $heads_iter; do
                for seed in $SEEDS; do
                    [[ -n "$FILTER_SEED" && "$seed" != "$FILTER_SEED" ]] && continue

                    for size in $GRAPH_SIZES; do
                        [[ -n "$FILTER_SIZE" && "$size" != "$FILTER_SIZE" ]] && continue

                        run_idx=$((run_idx + 1))
                        run_name="${arch}_L${n_layers}_H${hidden_dim}_h${n_heads}_s${seed}_${size}"
                        ablation_ckpt_dir="${CHECKPOINT_DIR}/${run_name}"
                        mrx_ckpt="${ablation_ckpt_dir}/MrX.pt"

                        echo ""
                        echo "[$run_idx/$total] $run_name"

                        # Skip if already trained
                        if [[ -f "$mrx_ckpt" ]]; then
                            echo "  [skip] checkpoint already exists: $mrx_ckpt"
                            continue
                        fi

                        read -r g_nodes g_edges g_police g_money <<< "$(graph_size_params "$size")"
                        log_dir="${SRC_DIR}/configs/experiments/ablation_runs/logs/${run_name}"

                        if [[ "$DRY_RUN" == true ]]; then
                            echo "  [dry_run] arch=$arch layers=$n_layers dim=$hidden_dim heads=$n_heads seed=$seed"
                            echo "           graph_nodes=$g_nodes graph_edges=$g_edges police=$g_police money=$g_money"
                            echo "           checkpoint → $ablation_ckpt_dir"
                            continue
                        fi

                        # Write temporary agent config YAML
                        tmp_agent_cfg=$(mktemp /tmp/ablation_agent_XXXXXX.yaml)
                        trap "rm -f '$tmp_agent_cfg'" EXIT

                        case "$arch" in
                            gnn)
                                cat > "$tmp_agent_cfg" <<YAML
agent_type: gnn
gamma: 0.99
lr: 0.001
batch_size: 64
buffer_size: 10000
epsilon: 1.0
epsilon_decay: 0.995
epsilon_min: 0.01
YAML
                                ;;
                            gat)
                                cat > "$tmp_agent_cfg" <<YAML
agent_type: gat
hidden_dim: ${hidden_dim}
num_layers: ${n_layers}
heads: ${n_heads}
dropout: 0.2
edge_dim: 1
gamma: 0.99
lr: 0.001
batch_size: 64
buffer_size: 10000
epsilon: 1.0
epsilon_decay: 0.995
epsilon_min: 0.01
YAML
                                ;;
                            transformer)
                                cat > "$tmp_agent_cfg" <<YAML
agent_type: transformer
hidden_dim: ${hidden_dim}
num_layers: ${n_layers}
num_heads: ${n_heads}
dropout: 0.1
edge_dim: 1
use_positional_encoding: false
pe_dim: 8
gamma: 0.99
lr: 0.0003
batch_size: 64
buffer_size: 10000
epsilon: 1.0
epsilon_decay: 0.995
epsilon_min: 0.01
YAML
                                ;;
                        esac

                        # Write temporary experiment config YAML
                        tmp_exp_cfg=$(mktemp /tmp/ablation_exp_XXXXXX.yaml)
                        trap "rm -f '$tmp_exp_cfg' '$tmp_agent_cfg'" EXIT

                        epochs="${EPOCH_OVERRIDE:-50}"
                        cat > "$tmp_exp_cfg" <<YAML
agent_configurations:
  - num_police_agents: ${g_police}
    agent_money: ${g_money}

graph_nodes: ${g_nodes}
graph_edges: ${g_edges}
num_episodes: 10
num_eval_episodes: 5
epochs: ${epochs}

log_dir: logs/${run_name}
wandb_run_name: ablation_${run_name}
wandb_resume: false
random_seed: ${seed}
evaluate: false
log_configs: default
vis_configs: none
YAML

                        mkdir -p "$log_dir"

                        echo "  Training..."
                        $PYTHON "${SRC_DIR}/main.py" \
                            --config "$tmp_exp_cfg" \
                            --agent_configs "$tmp_agent_cfg" \
                            --wandb_api_key "$WANDB_API_KEY" \
                            --wandb_project "$WANDB_PROJECT" \
                            --wandb_entity "$WANDB_ENTITY" \
                            --wandb_run_name "ablation_${run_name}" \
                            --ablation_checkpoint_dir "$ablation_ckpt_dir" \
                            && echo "  Done → $ablation_ckpt_dir" \
                            || { echo "  FAILED: $run_name"; failed=$((failed + 1)); }

                        rm -f "$tmp_agent_cfg" "$tmp_exp_cfg"
                        trap - EXIT

                    done  # size
                done  # seed
            done  # n_heads
        done  # hidden_dim
    done  # n_layers
done  # arch

echo ""
echo "================================================"
echo "Grid complete. $run_idx runs attempted, $failed failed."
if [[ $failed -gt 0 ]]; then
    echo "Re-run with the same flags to retry failed combinations."
    exit 1
fi
