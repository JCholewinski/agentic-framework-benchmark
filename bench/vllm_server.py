"""Budowanie argumentów `vllm serve` z konfiguracji (jedno źródło prawdy: configs/*.yaml).

Użycie (wywołuje to scripts/start_vllm.sh):
    python -m bench.vllm_server configs/pilot.yaml [--set klucz=wartość ...]

Wypisuje argumenty rozdzielone bajtem NUL, żeby skrypt bash mógł je bezpiecznie wczytać do tablicy.
`--set` nadpisuje pole sekcji `vllm` na potrzeby jednego pomiaru, np. `--set max_model_len=49152`.
"""

from __future__ import annotations

import argparse
import json
import sys

from bench.config import Config, VllmCfg, load_config


def vllm_args(cfg: Config) -> list[str]:
    v = cfg.vllm
    args = [
        cfg.model.hf_repo,
        "--served-model-name", cfg.model.served_name,
        "--host", cfg.endpoints.host,
        "--port", str(v.port),
        "--max-model-len", str(v.max_model_len),
        "--gpu-memory-utilization", str(v.gpu_memory_utilization),
        "--kv-cache-dtype", v.kv_cache_dtype,
        "--max-num-seqs", str(v.max_num_seqs),
        "--enable-auto-tool-choice",
        "--tool-call-parser", v.tool_call_parser,
        "--reasoning-parser", v.reasoning_parser,
        # Tryb myślenia ustawiany po stronie serwera: identyczny dla wszystkich frameworków.
        "--default-chat-template-kwargs", json.dumps({"enable_thinking": cfg.model.enable_thinking}),
        # Wartości domyślne z generation_config.json modelu nie mogą wpływać na wyniki:
        # sampling ustawiamy jawnie w każdym zapytaniu (docelowo wymusza go proxy).
        "--generation-config", "vllm",
    ]
    if v.language_model_only:
        args.append("--language-model-only")
    if v.enforce_eager:
        args.append("--enforce-eager")
    return args


def _apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    if not overrides:
        return cfg
    data = cfg.vllm.model_dump()
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or key not in data:
            sys.exit(f"Nieznane nadpisanie: {item!r} (dozwolone klucze: {', '.join(data)})")
        data[key] = yaml_scalar(raw)
    return cfg.model_copy(update={"vllm": VllmCfg.model_validate(data)})


def yaml_scalar(raw: str):
    import yaml

    return yaml.safe_load(raw)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--set", action="append", default=[], metavar="KLUCZ=WARTOŚĆ")
    a = p.parse_args()
    cfg = _apply_overrides(load_config(a.config), a.set)
    sys.stdout.write("\0".join(vllm_args(cfg)))


if __name__ == "__main__":
    main()
