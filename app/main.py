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
        raw = call_llm(prompt, temperature=1).strip()
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))

    # Extract first JSON object on a single line
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise HTTPException(status_code=400, detail="Model did not return JSON")

    try:
        bundle = json.loads(m.group(0))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON parse error: {e}")

    if req.filename:
        bundle["filename"] = req.filename

    filename, code_bytes, code_text = decode_bundle_lines(bundle)

    # Validate Python syntax
    try:
        ast.parse(code_text)
    except SyntaxError as e:
        print("---- DECODED CODE START ----")
        print(code_text)
        print("---- DECODED CODE END ----")
        raise HTTPException(status_code=400, detail=f"Python syntax error: {e}")

    # Normalize response
    bundle["code_lines"] = code_text.split("\n")
    return bundle

@app.post("/save_bundle")
def save_bundle(req: SaveRequest | dict):
    payload = req if isinstance(req, dict) else req.model_dump()
    bundle = payload.get("bundle", payload)  # accept raw or wrapped
    filename, code_bytes, _ = decode_bundle_lines(bundle)
    saved = write_file(filename, code_bytes)
    return {"ok": True, "saved_path": saved, "bytes": len(code_bytes)}