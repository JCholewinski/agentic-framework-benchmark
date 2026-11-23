# agent-bench

Porównanie frameworków agentowych (LangGraph, Microsoft Agent Framework, CrewAI, Google ADK)
z mniejszym modelem językowym uruchamianym lokalnie (Qwen3.8-27B, 4-bit, vLLM) na zadaniach z LiveCodeBench.

Pełny opis projektu, decyzje i obecny etap: [CLAUDE.md](CLAUDE.md).

## Gdzie co działa

- **Lokalnie** tylko piszemy kod.
- **Na instancji Vast.ai** (RTX 4090) uruchamiamy wszystko: vLLM, proxy, runner, piaskownicę i ocenę.

## Środowisko

Zależności są przypięte w `pyproject.toml` i `uv.lock` ([uv](https://docs.astral.sh/uv/)). Python 3.12.

Na instancji Vast.ai:

```bash
git clone <adres repo> agent-bench && cd agent-bench
bash scripts/setup_vastai.sh
```

Zmiana lub dodanie zależności (lokalnie), potem commit `pyproject.toml` i `uv.lock`:

```bash
uv add --no-sync --bounds exact <pakiet>
```

## Struktura

```
configs/      konfiguracje eksperymentu (model, adresy, limity)
data/         ID zadań pilotażowych i próby głównej
bench/        wspólne elementy: zadania, piaskownica, ocena, prompty, proxy, runner
frameworks/   implementacje konfiguracji (po jednej na framework i tryb)
scripts/      skrypty uruchomieniowe (setup instancji, start vLLM)
results/      wyniki przebiegów (JSONL)
analysis/     analiza wyników
```
