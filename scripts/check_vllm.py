"""Test serwera vLLM przed pilotażem (krok 2). Uruchamiać na instancji, gdy działa scripts/start_vllm.sh:

    uv run python scripts/check_vllm.py configs/pilot.yaml --vllm-log logs/vllm_<data>.log

Sprawdza (każdy test --reps razy, bo pojedyncza próba niewiele mówi):
  thinking_off   odpowiedź bez bloku <think> i bez reasoning_content (myślenie wyłączone na serwerze)
  tool_call      model wywołuje narzędzie z poprawnym JSON-em argumentów i używa wyniku narzędzia
  json_plain     werdykt JSON tylko z instrukcji w prompcie (tak będą odpowiadać agenci)
  json_schema    werdykt JSON z response_format (structured outputs vLLM)
  long_context   zapytanie na ~90% max_model_len: działa i odnajduje informację z początku
  overflow       zapytanie ponad max_model_len: jaki błąd zwraca vLLM (runner ma go rozpoznawać)
Do tego zużycie VRAM (nvidia-smi) i linie o pamięci z logu vLLM.
Raport: results/checks/vllm_check_<data>.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import httpx
import openai
from openai import OpenAI

from bench.config import Config, load_config

RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "Run a Python program and return its standard output.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Complete Python program."}},
            "required": ["code"],
        },
    },
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["pass", "fail"]}, "feedback": {"type": "string"}},
    "required": ["verdict", "feedback"],
    "additionalProperties": False,
}

VERDICT_TASK = (
    "You are a code verifier. Task: read an integer n and print n*2.\n"
    "Program:\n```python\nn = int(input())\nprint(n + 2)\n```\n"
    "Sample: input `3`, expected output `6`.\n"
)


class Checker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.endpoints.vllm_base_url, api_key=cfg.endpoints.api_key, timeout=600)
        g = cfg.generation
        self.sampling = {"temperature": g.temperature, "top_p": g.top_p, "presence_penalty": g.presence_penalty, "seed": g.seed}
        self.extra = {"top_k": g.top_k}

    def chat(self, messages, max_tokens: int | None = None, **kw):
        t0 = time.perf_counter()
        r = self.client.chat.completions.create(
            model=self.cfg.model.served_name, messages=messages,
            max_tokens=max_tokens or self.cfg.generation.max_tokens,
            extra_body=self.extra, **self.sampling, **kw,
        )
        return r, time.perf_counter() - t0

    @staticmethod
    def _usage(r) -> dict:
        return {"prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens}

    @staticmethod
    def _reasoning(msg) -> str:
        extra = msg.model_extra or {}
        return extra.get("reasoning_content") or extra.get("reasoning") or ""

    # --- testy: każdy zwraca dict z kluczem "ok" ---------------------------------------------

    def thinking_off(self):
        r, dt = self.chat([{"role": "user", "content": "What is 17 * 23? Answer with the number only."}], max_tokens=256)
        m = r.choices[0].message
        content = m.content or ""
        return {"ok": "<think>" not in content and not self._reasoning(m) and "391" in content,
                "content": content[:200], "reasoning_chars": len(self._reasoning(m)), "s": dt, **self._usage(r)}

    def tool_call(self):
        msgs = [{"role": "user", "content": "Use the run_python tool to compute the sum of squares of 1..10, "
                                            "then tell me the result."}]
        r1, dt1 = self.chat(msgs, tools=[RUN_PYTHON_TOOL], tool_choice="auto")
        m = r1.choices[0].message
        out = {"finish_reason": r1.choices[0].finish_reason, "s": dt1, **self._usage(r1)}
        if not m.tool_calls:
            return out | {"ok": False, "error": "brak tool_calls", "content": (m.content or "")[:300]}
        call = m.tool_calls[0]
        out["tool_name"] = call.function.name
        try:
            args = json.loads(call.function.arguments)
        except json.JSONDecodeError as e:
            return out | {"ok": False, "error": f"argumenty to nie JSON: {e}", "raw": call.function.arguments[:300]}
        if call.function.name != "run_python" or not isinstance(args.get("code"), str):
            return out | {"ok": False, "error": "zła nazwa narzędzia albo brak 'code'", "args": args}
        # Drugi krok: oddajemy wynik narzędzia i sprawdzamy, czy model go użyje.
        msgs += [m.model_dump(exclude_none=True),
                 {"role": "tool", "tool_call_id": call.id, "content": "385"}]
        r2, dt2 = self.chat(msgs, tools=[RUN_PYTHON_TOOL])
        final = r2.choices[0].message.content or ""
        return out | {"ok": "385" in final, "final": final[:200], "s": dt1 + dt2,
                      "completion_tokens": out["completion_tokens"] + r2.usage.completion_tokens}

    def json_plain(self):
        prompt = VERDICT_TASK + ('Respond ONLY with a JSON object: {"verdict": "pass" | "fail", "feedback": "..."}')
        r, dt = self.chat([{"role": "user", "content": prompt}], max_tokens=512)
        content = (r.choices[0].message.content or "").strip()
        strict = _parse_verdict(content)
        fenced = strict or _parse_verdict(re.sub(r"^```(?:json)?\s*|\s*```$", "", content))
        return {"ok": strict is not None, "ok_after_stripping_fences": fenced is not None,
                "verdict": (fenced or {}).get("verdict"), "content": content[:300], "s": dt, **self._usage(r)}

    def json_schema(self):
        r, dt = self.chat(
            [{"role": "user", "content": VERDICT_TASK + "Give your verdict."}], max_tokens=512,
            response_format={"type": "json_schema", "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA}},
        )
        content = r.choices[0].message.content or ""
        v = _parse_verdict(content)
        return {"ok": v is not None, "verdict": (v or {}).get("verdict"), "content": content[:300], "s": dt, **self._usage(r)}

    def _filler(self, target_tokens: int) -> str:
        unit = "The quick brown fox jumps over the lazy dog near the quiet river bank. "
        r, _ = self.chat([{"role": "user", "content": unit * 200}], max_tokens=1)
        per_unit = r.usage.prompt_tokens / 200
        return unit * int(target_tokens / per_unit)

    def long_context(self):
        target = int(0.9 * self.cfg.vllm.max_model_len) - 64
        prompt = ("Remember: the secret code is 4817.\n" + self._filler(target) +
                  "\nWhat is the secret code? Answer with the number only.")
        r, dt = self.chat([{"role": "user", "content": prompt}], max_tokens=16)
        content = r.choices[0].message.content or ""
        return {"ok": "4817" in content, "content": content[:100], "s": dt, **self._usage(r)}

    def overflow(self):
        prompt = self._filler(self.cfg.vllm.max_model_len + 2000)
        try:
            r, _dt = self.chat([{"role": "user", "content": prompt}], max_tokens=16)
        except openai.APIStatusError as e:
            return {"ok": e.status_code == 400, "status": e.status_code, "message": str(e)[:500]}
        return {"ok": False, "error": "vLLM przyjął zapytanie ponad max_model_len", **self._usage(r)}


def _parse_verdict(text: str):
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) and d.get("verdict") in ("pass", "fail") else None


def gpu_memory() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.strip()
        name, used, total = (x.strip() for x in out.splitlines()[0].split(","))
        return {"gpu": name, "used_mib": int(used), "total_mib": int(total)}
    except (OSError, subprocess.CalledProcessError, ValueError) as e:
        return {"error": str(e)}


def vllm_log_memory(path: str | None) -> list[str]:
    if not path:
        return []
    pat = re.compile(r"KV cache|Maximum concurrency|Model loading took|CUDA graph|memory|OutOfMemory", re.IGNORECASE)
    return [ln.strip()[:300] for ln in Path(path).read_text(errors="ignore").splitlines() if pat.search(ln)][-30:]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--vllm-log")
    p.add_argument("--skip-long", action="store_true", help="pomiń long_context i overflow")
    a = p.parse_args()

    cfg = load_config(a.config)
    ck = Checker(cfg)
    root = cfg.endpoints.vllm_base_url.removesuffix("/v1")
    report: dict = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "vllm_version": httpx.get(f"{root}/version", timeout=10).json().get("version"),
        "models": [m.id for m in ck.client.models.list().data],
        "config": cfg.model_dump(),
        "tests": {},
    }

    tests = [("thinking_off", a.reps), ("tool_call", a.reps), ("json_plain", a.reps), ("json_schema", a.reps)]
    if not a.skip_long:
        tests += [("long_context", 1), ("overflow", 1)]
    for name, reps in tests:
        runs = []
        for _ in range(reps):
            try:
                runs.append(getattr(ck, name)())
            except Exception as e:  # noqa: BLE001 - raport ma zawierać każdy błąd, nie przerywać testów
                runs.append({"ok": False, "exception": f"{type(e).__name__}: {e}"[:500]})
        report["tests"][name] = {"passed": sum(r["ok"] for r in runs), "reps": reps, "runs": runs}
        print(f"{name:14s} {sum(r['ok'] for r in runs)}/{reps}")

    report["gpu_memory"] = gpu_memory()
    report["vllm_log_memory"] = vllm_log_memory(a.vllm_log)
    print("VRAM:", report["gpu_memory"])
    for ln in report["vllm_log_memory"]:
        print("  log:", ln)

    out = Path(cfg.output.results_dir) / "checks" / f"vllm_check_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("Raport:", out)


if __name__ == "__main__":
    main()
