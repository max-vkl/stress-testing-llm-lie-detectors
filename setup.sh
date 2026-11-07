#!/usr/bin/env bash
# Install the locked environment and prepare the immutable reproduction assets.
set -euo pipefail
setup_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
  -h|--help) echo 'Usage: ./setup.sh [--dependencies-only]'; echo 'Default: install dependencies, download pinned Llama weights and full DolusChat.'; exit 0 ;;
  ''|--dependencies-only) ;;
  *) echo 'Usage: ./setup.sh [--dependencies-only]' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Too many arguments' >&2; exit 2; }
cd -- "$setup_root"
if ! command -v uv >/dev/null 2>&1; then
  export UV_INSTALL_DIR="$setup_root/.tools"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$UV_INSTALL_DIR:$PATH"
fi
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$setup_root/.venv}"
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 MPLBACKEND=Agg
uv sync --frozen --python 3.12
setup_python="$UV_PROJECT_ENVIRONMENT/bin/python"
"$setup_python" -m deception_detection.scripts.validate_reproduction
if [[ "${1:-}" != --dependencies-only ]]; then
  "$setup_python" -u -m deception_detection.scripts.prepare_reproduction
  "$setup_python" -m deception_detection.scripts.validate_reproduction --require-assets
fi
printf '\nSetup complete. Run ./reproduce_results.sh\n'
