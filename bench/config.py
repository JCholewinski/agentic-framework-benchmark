"""Ładowanie konfiguracji eksperymentu z pliku YAML (configs/*.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    # Literówka w kluczu YAML ma być błędem, a nie cicho ignorowanym polem.
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelCfg(_Strict):
    served_name: str
    hf_repo: str
    enable_thinking: bool


class GenerationCfg(_Strict):
    temperature: float
    top_p: float
    max_tokens: int
    seed: int


class EndpointsCfg(_Strict):
    vllm_base_url: str
    proxy_base_url: str
    api_key: str

    def base_url_for(self, run_id: str) -> str:
        """Adres, którego framework używa w danym przebiegu (run_id w ścieżce, patrz bench/proxy.py)."""
        return self.proxy_base_url.format(run_id=run_id)


class LimitsCfg(_Strict):
    max_iterations: int
    run_timeout_s: int
    max_llm_calls: int


class SandboxCfg(_Strict):
    exec_timeout_s: int
    memory_limit_mb: int
    max_output_chars: int


class TasksCfg(_Strict):
    file: str


class OutputCfg(_Strict):
    results_dir: str


class Config(_Strict):
    model: ModelCfg
    generation: GenerationCfg
    endpoints: EndpointsCfg
    limits: LimitsCfg
    sandbox: SandboxCfg
    tasks: TasksCfg
    output: OutputCfg


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))
