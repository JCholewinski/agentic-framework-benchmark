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


class VllmCfg(_Strict):
    version: str
    port: int
    max_model_len: int
    gpu_memory_utilization: float
    kv_cache_dtype: str
    max_num_seqs: int
    language_model_only: bool
    enforce_eager: bool
    tool_call_parser: str
    reasoning_parser: str


class GenerationCfg(_Strict):
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    max_tokens: int
    seed: int

    def sampling_params(self) -> dict:
        """Parametry samplingu w formacie API vLLM (top_k nie należy do standardu OpenAI)."""
        return self.model_dump()


class EndpointsCfg(_Strict):
    host: str
    proxy_port: int
    api_key: str


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
    vllm: VllmCfg
    generation: GenerationCfg
    endpoints: EndpointsCfg
    limits: LimitsCfg
    sandbox: SandboxCfg
    tasks: TasksCfg
    output: OutputCfg

    @property
    def vllm_base_url(self) -> str:
        """Bezpośredni adres vLLM (tylko dla proxy i testów serwera, nie dla frameworków)."""
        return f"http://{self.endpoints.host}:{self.vllm.port}/v1"

    def proxy_base_url(self, run_id: str) -> str:
        """Adres, którego framework używa w danym przebiegu (run_id w ścieżce, patrz bench/proxy.py)."""
        return f"http://{self.endpoints.host}:{self.endpoints.proxy_port}/run/{run_id}/v1"


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))
