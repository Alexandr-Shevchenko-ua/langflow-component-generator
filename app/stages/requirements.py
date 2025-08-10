# app/stages/requirements.py
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

# We reuse your existing LLM wrapper (OpenAI client behind it).
from app.llm import call_llm

# app/stages/requirements.py (top)
REQUIRED_KEYS = [
    "raw_spec","purpose","context","inputs","outputs",
    "constraints","non_functional","acceptance_criteria",
    "risks","assumptions","open_questions"
]
POLICY_REL_PATH = "config/requirements_policy.json"

_POLICY_CACHE = None

def _load_policy() -> dict:
    global _POLICY_CACHE
    if _POLICY_CACHE is not None:
        return _POLICY_CACHE
    policy_path = os.getenv("AI_AGENCY_REQ_POLICY", "config/requirements_policy.json")
    with open(policy_path, "r", encoding="utf-8") as f:
        _POLICY_CACHE = json.load(f)
    return _POLICY_CACHE

# app/stages/requirements.py
def _policy_prompt(spec: str) -> str:
    return f"""
You are a requirements policy designer. Produce a JSON policy tailored to the user's brief.

USER BRIEF:
<<<
{spec}
>>>

Return ONE JSON object with keys:
- "min_counts": object with optional integer minimums for:
  "inputs","outputs","constraints","non_functional","acceptance_criteria","risks","assumptions","open_questions"
  (Set realistic minimums for THIS task; omit keys if no minimum should be enforced.)
- "banned_tokens": array of lowercase substrings to disallow in outputs (e.g., ["tbd","placeholder"])
- "prompt": object:
  - "system": one sentence describing the assistant role (e.g., "You are a senior requirements engineer...")
  - "format_rules": array of natural-language MUST rules customized to this task (avoid numbers hardcoded; reference min_counts symbolically where helpful)
  - "example_shape": a single-line example JSON shape (no values from the brief)
- "fallback_defaults": object with tailored defaults:
  - "context": string (1–2 sentences)
  - "inputs_extra": array of IO objects to add if too few inputs
  - "outputs_extra": array of IO objects to add if too few outputs
  - "constraints": array of strings
  - "non_functional": array of strings
  - "acceptance_criteria": array of strings
  - "risks": array of strings
  - "assumptions": array of strings
  - "open_questions": array of strings

Rules:
- Valid JSON only. No markdown, no backticks, no comments.
- Tailor the content to the brief.
- Do not restate the brief verbatim in fallback_defaults except where necessary.
""".strip()

def generate_policy(spec: str, workspace: Path) -> dict:
    ws = Path(workspace)
    policy_path = ws / POLICY_REL_PATH
    if policy_path.exists():
        return json.loads(policy_path.read_text(encoding="utf-8"))
    # Ask LLM to tailor the policy
    raw = call_llm(_policy_prompt(spec), temperature=1).strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise ValueError("Policy LLM did not return JSON")
    policy = json.loads(m.group(0))

    # Inject truly universal constants if missing
    policy.setdefault("prompt", {})
    policy["prompt"].setdefault("example_shape",
        "{\"raw_spec\":\"...\",\"purpose\":\"...\",\"context\":\"...\",\"inputs\":[{\"name\":\"...\",\"type\":\"...\",\"required\":true,\"description\":\"...\"}],\"outputs\":[{\"name\":\"...\",\"type\":\"...\",\"description\":\"...\"}],\"constraints\":[\"...\"],\"non_functional\":[\"...\"],\"acceptance_criteria\":[\"...\"],\"risks\":[\"...\"],\"assumptions\":[\"...\"],\"open_questions\":[\"...\"]}"
    )
    # Ensure keys exist even if LLM omits (no thresholds enforced if missing)
    policy.setdefault("min_counts", {})
    policy.setdefault("banned_tokens", [])
    policy.setdefault("fallback_defaults", {})
    # Persist
    (policy_path.parent).mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    return policy

def _build_prompt_from_policy(policy: dict, spec: str) -> str:
    system = policy.get("prompt", {}).get("system", "You are a senior requirements engineer.")
    rules = "\n".join(policy.get("prompt", {}).get("format_rules", []))
    example = policy.get("prompt", {}).get("example_shape", "")
    return (
        f"{system}\n\nUSER BRIEF:\n<<<\n{spec}\n>>>\n\n"
        "Output ONE SINGLE-LINE JSON object with EXACT keys:\n"
        + ",".join(f"\"{k}\"" for k in REQUIRED_KEYS)
        + "\n\nQuality rules (MUST):\n"
        + rules
        + ("\n\nExample shape (do NOT copy values):\n" + example if example else "")
    )

def _check_against_policy(doc: dict, spec: str, policy: dict) -> tuple[bool, str]:
    # schema keys
    missing = set(REQUIRED_KEYS) - set(doc.keys())
    if missing:
        return False, f"missing keys: {sorted(missing)}"
    # min counts (optional, only if present)
    mc = policy.get("min_counts", {})
    for k, n in mc.items():
        v = doc.get(k)
        if not isinstance(v, list) or len(v) < int(n):
            return False, f"{k} < {n}"
    # banned tokens (optional)
    banned = [b.lower() for b in policy.get("banned_tokens", [])]
    if banned:
        blob = json.dumps(doc, ensure_ascii=False).lower()
        if any(tok in blob for tok in banned):
            return False, "banned token detected"
    # IO field shapes
    for io in doc.get("inputs", []):
        if not all(x in io for x in ("name","type","required","description")):
            return False, "inputs missing fields"
    for oo in doc.get("outputs", []):
        if not all(x in oo for x in ("name","type","description")):
            return False, "outputs missing fields"
    # sanity: purpose shouldn’t equal raw_spec (universal)
    if str(doc.get("purpose","")).strip() == str(spec).strip():
        return False, "purpose equals raw_spec"
    return True, ""

def _fallback_from_policy(spec: str, policy: dict) -> RequirementsDoc:
    fb = policy.get("fallback_defaults", {})
    # minimal first sentence purpose
    def first_sentence(t: str) -> str:
        t = (t or "").strip()
        for sep in [". ", "!\n", "?\n", "\n"]:
            if sep in t: return t.split(sep,1)[0].strip()
        return t[:200]
    purpose = first_sentence(spec) or "Deliver the described capability."

    # try to guess basic IO from spec, then top up from policy extras
    inputs, outputs = _guess_io(spec)
    for e in fb.get("inputs_extra", []):
        if all(f.name != e.get("name") for f in inputs):
            inputs.append(IOField(**e))
    for e in fb.get("outputs_extra", []):
        if all(f.name != e.get("name") for f in outputs):
            outputs.append(IOField(**e))

    return RequirementsDoc(
        raw_spec=spec,
        purpose=purpose,
        context=fb.get("context", "Context not specified."),
        inputs=inputs or [IOField(name="input", type="string", required=True, description="Primary input")],
        outputs=outputs or [IOField(name="result", type="string", required=True, description="Primary output")],
        constraints=fb.get("constraints", []),
        non_functional=fb.get("non_functional", []),
        acceptance_criteria=fb.get("acceptance_criteria", []),
        risks=fb.get("risks", []),
        assumptions=fb.get("assumptions", []),
        open_questions=fb.get("open_questions", []),
    )

def _repair_prompt(policy: dict, spec: str, current: dict) -> str:
    return (
        "Your JSON violates the policy. FIX it.\n"
        f"POLICY:\n<<<{json.dumps(policy, ensure_ascii=False)}>>>\n"
        f"BRIEF:\n<<<{spec}>>>\n"
        f"CURRENT_JSON:\n<<<{json.dumps(current, ensure_ascii=False)}>>>\n"
        "Return ONE SINGLE-LINE VALID JSON ONLY with the required keys."
    )

def _analysis_prompt(spec: str, doc: dict, policy: dict) -> str:
    return f"""
You are writing a Stage 1 analysis report for a software moderator.
Produce Markdown with these sections:

# Executive Summary (2–4 sentences)
# Detailed Understanding of Request
# Risks & Gaps
# Suggested Improvements
# Clarifying Questions for Moderator Approval

Inputs you can use:
- USER BRIEF:
<<<
{spec}
>>>
- STRUCTURED REQUIREMENTS JSON:
<<<
{json.dumps(doc, ensure_ascii=False, indent=2)}
>>>
- POLICY (optional):
<<<
{json.dumps(policy, ensure_ascii=False)}
>>>

Write concise, specific, testable content.
Return Markdown only (no fences).
""".strip()

def generate_analysis_report(spec: str, doc: dict, policy: dict, workspace: Path) -> str:
    ws = Path(workspace); ws.mkdir(parents=True, exist_ok=True)
    md = call_llm(_analysis_prompt(spec, doc, policy), temperature=1).strip()
    (ws / "analysis_report.md").write_text(md, encoding="utf-8")
    return md


# ---------- Pydantic Schemas (strict output) ----------

class IOField(BaseModel):
    name: str
    type: str = Field(default="string")
    required: bool = Field(default=True)
    description: str = Field(default="")

    @field_validator("name", "type", "description")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()


class RequirementsDoc(BaseModel):
    raw_spec: str
    purpose: str
    context: str
    inputs: List[IOField] = Field(default_factory=list)
    outputs: List[IOField] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    non_functional: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)

    @field_validator(
        "constraints",
        "non_functional",
        "acceptance_criteria",
        "risks",
        "assumptions",
        "open_questions",
        mode="before",
    )
    @classmethod
    def ensure_list_of_str(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return [str(v).strip()]


# ---------- Core Stage Function ----------

def run_requirements(spec: str, workspace: Optional[Path | str] = None) -> dict:
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("Empty spec provided to requirements stage")
    ws = Path(workspace or ".artifacts/stage1")
    ws.mkdir(parents=True, exist_ok=True)

    # 0) policy
    policy = generate_policy(spec, ws)

    use_llm = os.getenv("AI_AGENCY_USE_LLM", "true").lower() not in {"0", "false", "no"}
    error_info = None

    def write_doc(docdict: dict):
        (ws / "requirements.json").write_text(json.dumps(docdict, ensure_ascii=False, indent=2), encoding="utf-8")

    if use_llm:
        try:
            raw = call_llm(_build_prompt_from_policy(policy, spec), temperature=1).strip()
            m = re.search(r"\{.*\}", raw, flags=re.S)
            if not m:
                raise ValueError("Model did not return JSON")
            data = json.loads(m.group(0))
            ok, why = _check_against_policy(data, spec, policy)
            if not ok:
                # repair once
                repaired_raw = call_llm(_repair_prompt(policy, spec, data), temperature=1).strip()
                m2 = re.search(r"\{.*\}", repaired_raw, flags=re.S)
                if not m2:
                    raise ValueError(f"Repair failed: {why}")
                data2 = json.loads(m2.group(0))
                ok2, why2 = _check_against_policy(data2, spec, policy)
                if not ok2:
                    raise ValueError(f"Repaired JSON still weak: {why2}")
                data = data2
            # validate with Pydantic for shape (strict types)
            doc = RequirementsDoc(**data)
            write_doc(json.loads(doc.model_dump_json()))
        except Exception as e:
            error_info = f"LLM path failed: {e}"
            doc = _fallback_from_policy(spec, policy)
            write_doc(json.loads(doc.model_dump_json()))
    else:
        doc = _fallback_from_policy(spec, policy)
        write_doc(json.loads(doc.model_dump_json()))

    # Part B: analysis report
    generate_analysis_report(spec, json.loads(doc.model_dump_json()), policy, ws)

    # Save any logs
    if error_info:
        (ws / "requirements.llm_log.txt").write_text(error_info, encoding="utf-8")

    return json.loads(doc.model_dump_json())



# ---------- LLM Prompting (JSON-only, robust) ----------

_REQUIREMENTS_PROMPT = """
You are a senior requirements engineer. Extract a COMPLETE, SPECIFIC requirements JSON from the brief.

USER BRIEF:
<<<
{SPEC}
>>>

Output: ONE SINGLE-LINE JSON object with EXACT keys:
"raw_spec","purpose","context","inputs","outputs","constraints","non_functional","acceptance_criteria","risks","assumptions","open_questions"

Quality rules (MUST):
- Valid JSON only (no markdown, no backticks).
- "purpose" MUST NOT equal "raw_spec".
- "context" MUST be 1–3 sentences and mention repo/runtime if implied.
- "inputs" MUST have ≥2 items; each has name,type,required,description.
- "outputs" MUST have ≥2 items; each has name,type,description.
- "constraints" ≥5; "non_functional" ≥4; "acceptance_criteria" ≥6; "risks" ≥3; "assumptions" ≥3; "open_questions" ≥3.
- No placeholders like "TBD", "Derived automatically", or "Confirm inputs".
- Keep it concise, but explicit and testable.

Example shape (do NOT copy values):
{"raw_spec":"...","purpose":"...","context":"...","inputs":[{"name":"...","type":"...","required":true,"description":"..."}],"outputs":[{"name":"...","type":"...","description":"..."}],"constraints":["..."],"non_functional":["..."],"acceptance_criteria":["..."],"risks":["..."],"assumptions":["..."],"open_questions":["..."]}
""".strip()

_MIN_COUNTS = {
    "inputs": 2,
    "outputs": 2,
    "constraints": 5,
    "non_functional": 4,
    "acceptance_criteria": 6,
    "risks": 3,
    "assumptions": 3,
    "open_questions": 3,
}
_BAD_TOKENS = {"tbd", "derived automatically", "confirm inputs", "insufficient detail"}

def _is_semantically_strong(payload: dict, spec: str) -> tuple[bool, str]:
    # purpose not equal raw_spec
    if payload.get("purpose","").strip() == spec.strip():
        return False, "purpose equals raw_spec"
    # min counts
    for k, n in _MIN_COUNTS.items():
        if not isinstance(payload.get(k), list) or len(payload[k]) < n:
            return False, f"{k} has less than {n} items"
    # no placeholders
    blob = json.dumps(payload).lower()
    if any(tok in blob for tok in _BAD_TOKENS):
        return False, "placeholder tokens detected"
    # inputs/outputs fields check
    for io in payload.get("inputs", []):
        if not all(x in io for x in ("name","type","required","description")):
            return False, "inputs missing fields"
    for oo in payload.get("outputs", []):
        if not all(x in oo for x in ("name","type","description")):
            return False, "outputs missing fields"
    return True, ""

_REPAIR_PROMPT = """
You produced this JSON, but it violates the quality rules. FIX it.
BRIEF:
<<<{SPEC}>>>
CURRENT_JSON:
<<<{CURRENT}>>>
Apply the same schema and MUST rules as before (min counts, no placeholders, purpose≠raw_spec). Return ONE SINGLE-LINE VALID JSON ONLY.
""".strip()


def _call_llm_requirements(spec: str) -> dict:
    raw = call_llm(_REQUIREMENTS_PROMPT.format(SPEC=spec), temperature=1).strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise ValueError("Model did not return JSON")

    data = json.loads(m.group(0))

    # shape check
    expected_keys = {"raw_spec","purpose","context","inputs","outputs","constraints","non_functional","acceptance_criteria","risks","assumptions","open_questions"}
    missing = expected_keys - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys in JSON: {sorted(missing)}")

    ok, why = _is_semantically_strong(data, spec)
    if not ok:
        repaired_raw = call_llm(_REPAIR_PROMPT.format(SPEC=spec, CURRENT=json.dumps(data, ensure_ascii=False)), temperature=1).strip()
        m2 = re.search(r"\{.*\}", repaired_raw, flags=re.S)
        if not m2:
            raise ValueError(f"Repair failed: {why}")
        data2 = json.loads(m2.group(0))
        ok2, why2 = _is_semantically_strong(data2, spec)
        if not ok2:
            raise ValueError(f"Repaired JSON still weak: {why2}")
        return data2

    return data


# ---------- Deterministic Fallback (no LLM) ----------

def _heuristic_requirements(spec: str) -> RequirementsDoc:
    """
    Very small, deterministic parser to ensure progress when LLM fails or is disabled.
    It guesses a sane default structure based on simple patterns.
    """
    purpose = _first_sentence(spec)
    inputs, outputs = _guess_io(spec)
    constraints = _grep_lines(spec, ["must", "should", "require", "limit", "only"])
    nonfunc = _grep_lines(spec, ["performance", "latency", "security", "privacy", "compliance", "maintain"])
    risks = []
    open_q = ["Confirm inputs/outputs and any environment constraints."]

    return RequirementsDoc(
        raw_spec=spec,
        purpose=purpose or "Deliver the described capability.",
        context="Derived automatically due to insufficient detail.",
        inputs=inputs or [IOField(name="input", type="string", required=True, description="Primary input")],
        outputs=outputs or [IOField(name="result", type="string", required=True, description="Primary output")],
        constraints=constraints,
        non_functional=nonfunc,
        acceptance_criteria=["A runnable demo validates the main flow."],
        risks=risks,
        assumptions=[],
        open_questions=open_q,
    )


# ----- tiny helpers (deterministic, robust) -----

def _first_sentence(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    # naive sentence split
    for sep in [". ", "!\n", "?\n", "\n"]:
        if sep in text:
            return text.split(sep, 1)[0].strip()
    return text[:200].strip()


def _guess_io(text: str):
    """
    Heuristics: look for tokens like 'input', 'output', 'timeout', 'name', etc.
    This is intentionally conservative; just to keep the pipeline flowing.
    """
    text_l = (text or "").lower()
    inputs = []
    outputs = []

    def add_in(name, typ="string", required=True, desc=""):
        inputs.append(IOField(name=name, type=typ, required=required, description=desc))

    def add_out(name, typ="string", desc=""):
        outputs.append(IOField(name=name, type=typ, required=True, description=desc))

    # common hints
    if "timeout" in text_l:
        add_in("timeout_s", "int", False, "Execution timeout in seconds")

    if "name" in text_l:
        add_in("name", "string", True, "Name value")

    if "code" in text_l or "script" in text_l or "snippet" in text_l:
        add_in("code", "string", True, "Python code snippet")

    if "greet" in text_l or "hello" in text_l:
        add_out("greeting", "string", "Greeting message")

    if "stdout" in text_l or "stderr" in text_l:
        add_out("stdout", "string", "Captured standard output")
        add_out("stderr", "string", "Captured standard error")

    # default if empty
    if not inputs:
        add_in("input", "string", True, "Primary input")
    if not outputs:
        add_out("result", "string", "Primary output")

    return inputs, outputs


def _grep_lines(text: str, keywords: list[str]) -> list[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    found = []
    for ln in lines:
        lower = ln.lower()
        if any(k in lower for k in keywords):
            found.append(ln)
    return found