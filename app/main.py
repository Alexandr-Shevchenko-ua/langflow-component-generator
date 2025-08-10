import base64, json, os, re, ast
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from jinja2 import Environment, FileSystemLoader, select_autoescape
from .llm import call_llm

app = FastAPI(title="Component Generator")

env = Environment(
    loader=FileSystemLoader("app/templates"),
    autoescape=select_autoescape()
)

class GenRequest(BaseModel):
    spec: str = Field(..., description="Natural language spec of the component")
    filename: str | None = Field(None, description="Override target filename if desired")
    char_limit: int = 60000

class SaveRequest(BaseModel):
    bundle: dict

def compile_pythonexec_from_spec(spec: dict) -> str:
    # Minimal validation of spec
    if spec.get("component") != "PythonExec":
        raise ValueError("Unsupported component for this compiler")
    display_name = spec.get("display_name") or "PythonExec"
    description = spec.get("description") or "Runs a short Python snippet with a timeout and returns stdout/stderr."
    banned = spec.get("banned_patterns") or []

    # ✅ normalize double-escaped sequences from JSON to single backslashes
    def _norm(p: str) -> str:
        # turn "\\bfrom\\s+socket\\b" into "\bfrom\s+socket\b"
        return p.replace("\\\\", "\\")

    normed = [_norm(p) for p in banned]

    lines = []
    lines.append("# Execute Python code safely with timeout")
    lines.append("class PythonExecComponent:")
    lines.append(f"    display_name = {display_name!r}")
    lines.append(f"    description = {description!r}")
    lines.append("")
    lines.append("    def build(self, code: str, timeout_s: int = 3) -> dict:")
    lines.append("        import sys, tempfile, subprocess, os, re, ast")
    lines.append("        # validate inputs")
    lines.append("        if not isinstance(code, str) or not code.strip():")
    lines.append('            raise ValueError("code must be a non-empty string")')
    lines.append("        if not isinstance(timeout_s, int):")
    lines.append('            raise ValueError("timeout_s must be an integer")')
    lines.append("        if timeout_s < 1 or timeout_s > 10:")
    lines.append('            raise ValueError("timeout_s must be between 1 and 10 seconds")')
    lines.append("")
    lines.append("        # deny common dangerous patterns (case-insensitive)")
    lines.append("        banned = [")
    for pat in normed:
        lines.append(f"            r'{pat}',")
    lines.append("        ]")
    lines.append("        for pat in banned:")
    lines.append("            if re.search(pat, code, re.IGNORECASE):")
    lines.append('                raise ValueError("code rejected by safety filter")')
    lines.append("")
    lines.append("        tmp = None")
    lines.append("        try:")
    lines.append('            tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".py", encoding="utf-8")')
    lines.append("            tmp.write(code)")
    lines.append("            tmp.close()")
    lines.append('            env = {"PYTHONIOENCODING": "utf-8"}')
    lines.append("            try:")
    lines.append('                completed = subprocess.run([sys.executable, "-I", tmp.name], capture_output=True, text=True, timeout=timeout_s, env=env)')
    lines.append("                return {")
    lines.append('                    "stdout": completed.stdout,')
    lines.append('                    "stderr": completed.stderr,')
    lines.append('                    "returncode": int(completed.returncode),')
    lines.append('                    "timed_out": False,')
    lines.append("                }")
    lines.append("            except subprocess.TimeoutExpired:")
    lines.append('                return {"stdout": "", "stderr": f"Timeout after {timeout_s}s", "returncode": -1, "timed_out": True}')
    lines.append("")
    lines.append("          # static import guard via AST")
    lines.append("        except:")
    lines.append("          try:")
    lines.append("              tree = ast.parse(code)")
    lines.append("          except SyntaxError as e:")
    lines.append('              raise ValueError(f"code has syntax error: {e}")')
    lines.append("          banned_modules = {")
    lines.append("              'socket','requests','urllib','httpx','ftplib','smtplib',")
    lines.append("              'aiohttp','importlib'")
    lines.append("          }")
    lines.append("          for node in ast.walk(tree):")
    lines.append("              if isinstance(node, (ast.Import, ast.ImportFrom)):")
    lines.append("                  names = [n.name.split('.')[0] for n in getattr(node, 'names', [])]")
    lines.append("                 mod = getattr(node, 'module', None)")
    lines.append("                  for n in names + ([mod] if mod else []):")
    lines.append("                        if n in banned_modules:")
    lines.append('                         raise ValueError(f"import of {n} is not allowed")')
    lines.append("        finally:")
    lines.append("            try:")
    lines.append("                if tmp is not None:")
    lines.append("                    os.unlink(tmp.name)")
    lines.append("            except Exception:")
    lines.append("                pass")

    return "\n".join(lines)

def bundle_from_compiled(filename: str, code_text: str) -> dict:
    # validate and emit code_lines
    ast.parse(code_text)  # should always pass; we generated it
    return {
        "filename": filename,
        "language": "python",
        "code_lines": code_text.split("\n"),
    }


def _extract_first_json(s: str) -> dict:
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        raise ValueError("Repair step did not return JSON")
    return json.loads(m.group(0))

def _decode_any_bundle(bundle: dict) -> tuple[str, str]:
    """
    Accepts either:
      - {"filename","language":"python","code_lines":[...]}
      - {"filename","encoding":"base64","language":"python","code":"..."}
    Returns (filename, code_text)
    """
    if "code_lines" in bundle:
        if (bundle.get("language") or "").lower() != "python":
            raise ValueError("language must be python")
        lines = bundle["code_lines"]
        if not isinstance(lines, list) or not all(isinstance(x, str) for x in lines):
            raise ValueError("code_lines must be array of strings")
        return bundle["filename"], "\n".join(lines)
    if "code" in bundle:
        if (bundle.get("encoding") or "").lower() != "base64":
            raise ValueError("encoding must be base64 for 'code'")
        return bundle["filename"], base64.b64decode(bundle["code"]).decode("utf-8", "ignore")
    raise ValueError("bundle has neither code_lines nor code")

def _try_self_repair(filename: str, bad_code_text: str, syntax_err: SyntaxError, spec: str, char_limit: int) -> str:
    """
    Ask the model to fix syntax ONLY, return a valid file using our contract.
    Returns repaired code_text (string) on success, or raises on failure.
    """
    repair_instructions = f"""
You returned a Python file with a syntax error. Repair it with minimal changes.

Constraints:
- Keep the class + method shape intact.
- Use ONLY stdlib. No triple-quoted docstrings; one-line # header only.
- File MUST parse with ast.parse().
- Return ONLY one single-line JSON object (no markdown):
  EITHER: {{"filename":"{filename}","language":"python","code_lines":[...]}}
  OR    : {{"filename":"{filename}","encoding":"base64","language":"python","code":"<base64>"}}
- No extra prose.

Context for repair:
- Target filename: {filename}
- User spec: {spec}
- Original SyntaxError: {syntax_err}

Original file (verbatim):
<<<PYTHON
{bad_code_text}
PYTHON
    """.strip()

    raw = call_llm(repair_instructions, temperature=1).strip()
    bundle = _extract_first_json(raw)
    fn, repaired_text = _decode_any_bundle(bundle)

    # Optional: re-apply light sanitize
    repaired_text = _light_sanitize(repaired_text)

    # Validate
    ast.parse(repaired_text)
    return repaired_text


def render_prompt(spec: str, char_limit: int) -> str:
    tmpl = env.get_template("component_prompt.txt")
    # Include a char limit hint in the prompt (models sometimes need it)
    return tmpl.render(SPEC=spec, SAFE_SLUG="component", CHAR_LIMIT=char_limit)

def write_file(path: str, data: bytes) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return str(p.resolve())

def _light_sanitize(code: str) -> str:
    # Fix common LLM glitches seen in practice
    code = code.replace("=>", ":")
    code = code.replace("issincess", "isinstance")
    code = code.replace("TypeType", "TypeError")
    code = code.replace("return f \"", "return f\"")
    code = code.replace("{;punctuation}", "{punctuation}")
    code = code.replace(" else .", " else \".\"")
    code = code.replace("f \"", "f\"")
    # Ensure class naming convention if obvious
    code = code.replace("class Greeter:", "class GreeterComponent:")
    code = code.replace("def run(", "def build(")
    return code

def decode_bundle_lines(bundle: dict) -> tuple[str, bytes, str]:
    try:
        filename = bundle["filename"]
        language = (bundle["language"] or "").lower()
        code_lines = bundle["code_lines"]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bundle shape")

    if language != "python":
        raise HTTPException(status_code=400, detail="language must be 'python'")

    if not isinstance(code_lines, list) or not all(isinstance(x, str) for x in code_lines):
        raise HTTPException(status_code=400, detail="code_lines must be an array of strings")

    code_text = "\n".join(code_lines)
    code_text = _light_sanitize(code_text)
    return filename, code_text.encode("utf-8"), code_text

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/generate_component")
def generate_component(req: GenRequest):
    prompt = render_prompt(req.spec, req.char_limit)
    try:
        raw = call_llm(prompt, temperature=0.0).strip()
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))

    # Get spec JSON (not code)
    try:
        spec = _extract_first_json(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="Model did not return JSON spec")

    # If this is PythonExec, we compile it ourselves (enterprise path)
    if spec.get("component") == "PythonExec":
        try:
            code_text = compile_pythonexec_from_spec(spec)
            filename = req.filename or "components/custom/python_exec_component.py"
            return bundle_from_compiled(filename, code_text)
        except Exception as e:
            # deterministic fallback: golden implementation
            code_text = compile_pythonexec_from_spec({
                "component": "PythonExec",
                "display_name": "PythonExec",
                "description": "Runs a short Python snippet with a timeout and returns stdout/stderr.",
                "banned_patterns": spec.get("banned_patterns") or []
            })
            filename = req.filename or "components/custom/python_exec_component.py"
            return bundle_from_compiled(filename, code_text)

    # For other components you can keep your previous code_lines path + self-heal
    # (decode LLM bundle, sanitize, ast.parse, repair once...)
    raise HTTPException(status_code=400, detail="Only PythonExec is supported by spec-compiler in this endpoint right now.")

@app.post("/save_bundle")
def save_bundle(req: SaveRequest | dict):
    payload = req if isinstance(req, dict) else req.model_dump()
    bundle = payload.get("bundle", payload)  # accept raw or wrapped
    filename, code_bytes, code_text = decode_bundle_lines(bundle)
    saved = write_file(filename, code_bytes)
    return {"ok": True, "saved_path": saved, "bytes": len(code_bytes)}
