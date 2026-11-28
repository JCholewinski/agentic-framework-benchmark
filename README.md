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

## Instancja Vast.ai

- GPU: 1× RTX 3090 (24 GB).
- Obraz: `vllm/vllm-openai:v0.34.0`, tryb uruchomienia SSH (obraz domyślnie sam startuje serwer, a my startujemy go skryptem).
- Filtr „Max CUDA” ≥ 13.0: obraz vLLM 0.34.0 jest zbudowany na CUDA 13.0, co wymaga sterownika NVIDIA ≥ 580.
- Dysk ≥ 80 GB (wagi ~20 GB, obraz, środowiska, dane), RAM ≥ 32 GB, CPU ≥ 8 rdzeni (piaskownica), szybkie łącze (pobieranie wag).
- Model `RedHatAI/Qwen3.8-27B-INT4` jest publiczny, więc token Hugging Face nie jest potrzebny.
- Uwaga na **szablon vLLM od Vast.ai**: przy starcie sam uruchamia vLLM z innym modelem (np. `Qwen/Qwen3.5-9B`),
  który zajmuje ~21 GB VRAM, a jego proxy (`caddy`) trzyma port 8000. Dlatego nasz vLLM działa na porcie 8100.
  Serwer z szablonu trzeba zatrzymać po każdym starcie instancji (`start_vllm.sh` to wykrywa):

  ```bash
  supervisorctl stop vllm
  ps -eo pid,ppid,cmd | grep -i vllm | grep -v grep    # zostają: `vllm serve ...` i VLLM::EngineCore
  kill <PID vllm serve> <PID EngineCore>
  ```

## Serwer vLLM i test modelu (krok 2)

Najprościej użyć instancji z obrazu Dockera `vllm/vllm-openai:v0.34.0`; bez niego skrypt sam zainstaluje
vLLM w osobnym środowisku (`/workspace/vllm-env`). Wagi modelu (~20 GB) trafiają do `/workspace/hf`.

```bash
tmux new -s vllm
bash scripts/start_vllm.sh configs/pilot.yaml            # opcjonalnie: --set max_model_len=49152
```

W drugim oknie, gdy serwer zgłosi gotowość:

```bash
uv run python scripts/check_vllm.py configs/pilot.yaml --vllm-log logs/vllm_<data>.log
```

Raport trafia do `results/checks/`.

## Zależności

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
