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
    char_limit: int = 18000

class SaveRequest(BaseModel):
    bundle: dict

def render_prompt(spec: str, char_limit: int) -> str:
    tmpl = env.get_template("component_prompt.txt")
    return tmpl.render(SPEC=spec, CHAR_LIMIT=char_limit)

def decode_bundle(bundle: dict) -> tuple[str, bytes]:
    try:
        filename = bundle["filename"]
        encoding = bundle["encoding"].lower()
        code_b64 = bundle["code"]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bundle shape")

    if encoding != "base64":
        raise HTTPException(status_code=400, detail="Only base64 encoding supported")

    try:
        code_bytes = base64.b64decode(code_b64, validate=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64 decode failed: {e}")

    return filename, code_bytes

def write_file(path: str, data: bytes) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return str(p.resolve())

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/generate_component")
def generate_component(req: GenRequest):
    prompt = render_prompt(req.spec, req.char_limit)
    raw = call_llm(prompt, temperature=0.0).strip()

    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise HTTPException(status_code=400, detail="Model did not return JSON")

    try:
        bundle = json.loads(m.group(0))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON parse error: {e}")

    if req.filename:
        bundle["filename"] = req.filename

    filename, code_bytes = decode_bundle(bundle)
    if filename.endswith(".py"):
        try:
            ast.parse(code_bytes.decode("utf-8"))
        except SyntaxError as e:
            raise HTTPException(status_code=400, detail=f"Python syntax error: {e}")

    return bundle

@app.post("/save_bundle")
def save_bundle(req: SaveRequest):
    filename, code_bytes = decode_bundle(req.bundle)
    saved = write_file(filename, code_bytes)
    return {"ok": True, "saved_path": saved, "bytes": len(code_bytes)}