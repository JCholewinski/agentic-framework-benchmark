#!/usr/bin/env bash
# Start serwera vLLM na instancji Vast.ai z parametrami z configu.
#
#   bash scripts/start_vllm.sh <config> [--set klucz=wartość ...]
#   np.  bash scripts/start_vllm.sh configs/pilot.yaml --set max_model_len=49152
#
# vLLM ma własne środowisko, oddzielne od frameworków (inne wersje torch itd.). Kolejność szukania:
#   1. $VLLM_BIN, jeśli ustawione;
#   2. `vllm` w PATH (np. instancja z obrazu Dockera vllm/vllm-openai:v<wersja>), jeśli wersja się zgadza;
#   3. venv w $VLLM_ENV (domyślnie /workspace/vllm-env), tworzony tu przy pierwszym uruchomieniu.
# Log serwera trafia do logs/vllm_<data>.log (potrzebny dla scripts/check_vllm.py).
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${1:?Podaj config, np. configs/pilot.yaml}"
shift

export HF_HOME="${HF_HOME:-/workspace/hf}"   # wagi (~20 GB) na dysku instancji, nie w repo

mapfile -d '' ARGS < <(uv run --frozen python -m bench.vllm_server "$CONFIG" "$@")
WANT="$(uv run --frozen python -c "from bench.config import load_config; print(load_config('$CONFIG').vllm.version)")"

vllm_version() { "$1" --version 2>/dev/null | tail -1 | tr -d '[:space:]'; }

if [[ -z "${VLLM_BIN:-}" ]]; then
    if command -v vllm >/dev/null 2>&1 && [[ "$(vllm_version "$(command -v vllm)")" == "$WANT" ]]; then
        VLLM_BIN="$(command -v vllm)"
    else
        VLLM_ENV="${VLLM_ENV:-/workspace/vllm-env}"
        if [[ ! -x "$VLLM_ENV/bin/vllm" ]]; then
            echo ">> Instaluję vLLM $WANT w $VLLM_ENV"
            uv venv --python 3.12 "$VLLM_ENV"
            uv pip install --python "$VLLM_ENV/bin/python" "vllm==$WANT"
        fi
        VLLM_BIN="$VLLM_ENV/bin/vllm"
    fi
fi

HAVE="$(vllm_version "$VLLM_BIN")"
if [[ "$HAVE" != "$WANT" ]]; then
    echo "!! vLLM '$HAVE', a config wymaga $WANT" >&2
    exit 1
fi

mkdir -p logs
LOG="logs/vllm_$(date +%Y%m%d_%H%M%S).log"
echo ">> vLLM $HAVE  |  log: $LOG"
echo ">> vllm serve ${ARGS[*]}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

"$VLLM_BIN" serve "${ARGS[@]}" 2>&1 | tee "$LOG"
