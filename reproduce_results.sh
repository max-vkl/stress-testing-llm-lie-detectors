#!/usr/bin/env bash
# Table 2, in paper order, with fixed zero-indexed layers from Appendix B.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'Execute this script instead of sourcing it.' >&2
  return 0
fi
set -euo pipefail
paper_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
paper_configs="$paper_root/deception_detection/scripts/configs/paper_main_llama"
paper_python="${PAPER_MAIN_PYTHON:-${UV_PROJECT_ENVIRONMENT:-$paper_root/.venv}/bin/python}"
paper_dry_run=false
paper_fresh=false
paper_stop_after=''
paper_runs=(ours marks_burger_truth marks_burger_lie goldowsky_dill_repe goldowsky_dill_roleplay zou_repe cooney_dyl cundy_doluschat macdiarmid_cp goldowsky_dill_sd)
usage() {
  cat <<'USAGE'
Usage: ./reproduce_results.sh [--dry-run] [--fresh] [--stop-after PROBE]

Run all ten Llama 3.3 70B configurations sequentially; stop on any failure.
--dry-run         Print commands without writing files or loading models.
--fresh           Retrain probes without reusing probe/activation caches.
--stop-after ours Complete the first probe only, with all 33 paper evaluations.

Run ./setup.sh first. Defaults: BF16, SDPA, activation batch 4, DYL batch 1.
PAPER_MAIN_PYTHON or UV_PROJECT_ENVIRONMENT selects the installed environment.
HF_HOME selects the model cache. DD_ACTS_BATCH and DD_DYL_BATCH_SIZE override batches.
Outputs: results/paper_main_llama/<run>/{logs,runs,main_table.csv,summary.json}.
USAGE
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) paper_dry_run=true; shift ;;
    --fresh) paper_fresh=true; shift ;;
    --stop-after) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; paper_stop_after="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
if [[ -n "$paper_stop_after" && " ${paper_runs[*]} " != *" $paper_stop_after "* ]]; then
  printf 'Unknown probe: %s\n' "$paper_stop_after" >&2; exit 2
fi
cd -- "$paper_root"
for paper_run in "${paper_runs[@]}"; do
  [[ -r "$paper_configs/$paper_run.yaml" ]] || { printf 'Missing config: %s\n' "$paper_run" >&2; exit 1; }
done
if [[ "$paper_dry_run" == false ]] && ! command -v "$paper_python" >/dev/null 2>&1; then
  echo 'Project Python not found. Run ./setup.sh first.' >&2; exit 1
fi
export DD_GPU_COMPACT_TRAIN=1 DD_LOCAL_FILES_ONLY=1 MPLBACKEND=Agg
export DD_ACTS_BATCH="${DD_ACTS_BATCH:-4}" DD_DYL_BATCH_SIZE="${DD_DYL_BATCH_SIZE:-1}"
export DD_DTYPE="${DD_DTYPE:-bf16}" DD_ATTN_IMPL="${DD_ATTN_IMPL:-sdpa}"
paper_output="$paper_root/results/paper_main_llama/$(date -u +%Y%m%d_%H%M%S)_$$"
[[ "$paper_dry_run" == true ]] || mkdir -p -- "$paper_output/logs"
printf 'Output directory: %s\n' "$paper_output"
run_step() {
  local step="$1"; shift
  printf '\n[%s] ' "$step"; printf '%q ' "$@"; printf '\n'
  if [[ "$paper_dry_run" == false ]]; then
    "$@" 2>&1 | tee "$paper_output/logs/$step.log"
  fi
}
run_step validate "$paper_python" -u -m deception_detection.scripts.validate_reproduction --require-assets --require-gpu
paper_extra=(--caching_off=False)
[[ "$paper_fresh" == false ]] || paper_extra=(--caching_off=True)
for paper_run in "${paper_runs[@]}"; do
  paper_config="$paper_configs/$paper_run.yaml"
  if [[ "$paper_run" == cooney_dyl ]]; then
    run_step prepare_dyl "$paper_python" -u -m deception_detection.scripts.prepare_dyl_followups \
      --model llama-70b-3.3 --config "$paper_config" --batch-size "$DD_DYL_BATCH_SIZE" --checkpoint-batches 5
  fi
  run_step "$paper_run" "$paper_python" -u -m deception_detection.scripts.experiment \
    run "$paper_config" --folder="$paper_output/runs" --name="$paper_run" "${paper_extra[@]}"
  [[ "$paper_run" != "$paper_stop_after" ]] || break
done
run_step aggregate "$paper_python" -u -m deception_detection.scripts.summarize_results "$paper_output"
if [[ "$paper_dry_run" == true ]]; then
  echo 'Dry run complete; no files written or experiments launched.'
elif [[ -n "$paper_stop_after" && "$paper_stop_after" != goldowsky_dill_sd ]]; then
  printf '\nPartial suite completed through %s. Results: %s\n' "$paper_stop_after" "$paper_output"
else
  printf '\nAll ten experiments completed. Results: %s\n' "$paper_output"
fi
