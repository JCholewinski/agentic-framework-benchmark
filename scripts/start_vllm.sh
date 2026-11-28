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
unset VIRTUAL_ENV   # szablony Vast.ai aktywują /venv/main; uv i tak używa .venv projektu

mapfile -d '' ARGS < <(uv run --frozen python -m bench.vllm_server "$CONFIG" "$@")
WANT="$(uv run --frozen python -c "from bench.config import load_config; print(load_config('$CONFIG').vllm.version)")"

arg_value() {
    local i
    for ((i = 0; i < ${#ARGS[@]} - 1; i++)); do
        if [[ "${ARGS[i]}" == "$1" ]]; then echo "${ARGS[i+1]}"; fi
    done
}
HOST="$(arg_value --host)"
PORT="$(arg_value --port)"

# --- Kontrola przed startem: GPU i port muszą być wolne -------------------------------------
# (Szablon vLLM od Vast.ai sam startuje własny serwer z innym modelem i zajmuje GPU oraz port 8000.)
GPU_APPS="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
if [[ -n "$GPU_APPS" ]]; then
    echo "!! Na GPU działają już inne procesy:" >&2
    echo "$GPU_APPS" | sed 's/^/     /' >&2
    echo "   Co to jest:   ps -eo pid,ppid,cmd | grep -i vllm | grep -v grep" >&2
    echo "   Szablon Vast.ai: supervisorctl stop vllm, potem kill <PID serwera i EngineCore>" >&2
    exit 1
fi
if ! python3 -c "import socket,sys; socket.socket().bind(('$HOST', $PORT))" 2>/dev/null; then
    echo "!! Port $HOST:$PORT jest zajęty. Kto go trzyma:  ss -ltnp | grep ':$PORT '" >&2
    echo "   Zmień vllm.port w configu albo uruchom z --set port=<inny>." >&2
    exit 1
fi

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
