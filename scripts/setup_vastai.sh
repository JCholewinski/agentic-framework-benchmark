#!/usr/bin/env bash
# Przygotowanie środowiska projektu na instancji Vast.ai.
# Uruchamiać z katalogu głównego repo po `git clone`:  bash scripts/setup_vastai.sh
# vLLM ma osobne środowisko (krok 2), żeby jego zależności (torch itd.) nie mieszały się z frameworkami.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo ">> Instaluję uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ">> uv $(uv --version)"
# --frozen: instalujemy DOKŁADNIE wersje z uv.lock; nie rozwiązujemy zależności na nowo.
uv sync --frozen

echo ">> Sprawdzam importy frameworków"
uv run python - <<'EOF'
from importlib.metadata import version
import langgraph, crewai, google.adk, agent_framework  # noqa: F401
for pkg in ["langgraph", "langchain-openai", "crewai", "agent-framework-core",
            "agent-framework-orchestrations", "google-adk", "litellm", "openai"]:
    print(f"   {pkg:32s} {version(pkg)}")
EOF

echo ">> Gotowe"
