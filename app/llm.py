import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY")

_client = OpenAI(api_key=API_KEY)

SYSTEM = (
    "You generate a Langflow custom component as a single Python file. "
    "Return ONLY one single-line JSON object (no markdown) with keys: "
    'filename, encoding="base64", language="python", code=base64(file). '
    "Any text outside JSON is a fatal error."
)

def call_llm(prompt: str, temperature: float = 0.0) -> str:
    resp = _client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()
