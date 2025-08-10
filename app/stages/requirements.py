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
    """
    Transform a natural-language brief into a structured requirements document.
    - Uses LLM (if enabled) to extract structure.
    - Validates with Pydantic.
    - Falls back to deterministic heuristic if LLM output is unusable.
    - Writes artifacts to workspace if provided.
    """
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("Empty spec provided to requirements stage")

    use_llm = os.getenv("AI_AGENCY_USE_LLM", "true").lower() not in {"0", "false", "no"}

    doc: RequirementsDoc
    error_info: Optional[str] = None

    if use_llm:
        try:
            payload = _call_llm_requirements(spec)
            doc = RequirementsDoc(**payload)
        except Exception as e:
            error_info = f"LLM requirements extraction failed: {e}"
            doc = _heuristic_requirements(spec)
    else:
        doc = _heuristic_requirements(spec)

    # Persist artifacts for auditability
    if workspace:
        ws = Path(workspace)
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "requirements.json").write_text(
            json.dumps(json.loads(doc.model_dump_json()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if error_info:
            (ws / "requirements.llm_log.txt").write_text(error_info, encoding="utf-8")

    return json.loads(doc.model_dump_json())


# ---------- LLM Prompting (JSON-only, robust) ----------

_REQUIREMENTS_PROMPT = """
You are a senior requirements engineer. Extract a structured requirements document from the user's brief.

USER BRIEF:
<<<
{SPEC}
>>>

Return ONLY one single-line JSON object (no markdown, no code fences, no extra text) with these EXACT keys:

- "raw_spec": string (verbatim user brief)
- "purpose": string (business outcome in one sentence)
- "context": string (operational/business context in 1-3 sentences)
- "inputs": array of {{"name": str, "type": str, "required": bool, "description": str}}
- "outputs": array of {{"name": str, "type": str, "description": str}}
- "constraints": array of strings (functional constraints)
- "non_functional": array of strings (performance, security, privacy, compliance, maintainability)
- "acceptance_criteria": array of strings (testable statements)
- "risks": array of strings (key project or technical risks)
- "assumptions": array of strings
- "open_questions": array of strings (unknowns to clarify)

Strictness:
- Valid JSON only (double quotes, commas, no comments).
- Keep it concise and specific.
- If some sections are unclear, include a short list of open_questions instead of guessing.
""".strip()


def _call_llm_requirements(spec: str) -> dict:
    """Call your LLM and parse the JSON response safely."""
    # We ask the model for JSON; extract the first {...} to be safe.
    raw = call_llm(_REQUIREMENTS_PROMPT.format(SPEC=spec), temperature=0.0).strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise ValueError("Model did not return JSON")

    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse error: {e}")

    # Quick shape check before Pydantic validation
    expected_keys = {
        "raw_spec", "purpose", "context", "inputs", "outputs",
        "constraints", "non_functional", "acceptance_criteria",
        "risks", "assumptions", "open_questions",
    }
    missing = expected_keys - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys in JSON: {sorted(missing)}")

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