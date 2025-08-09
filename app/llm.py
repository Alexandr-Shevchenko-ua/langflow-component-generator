import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI, APITimeoutError

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07")
API_KEY = os.getenv("OPENAI_API_KEY")

_client = OpenAI(api_key=API_KEY, timeout=60.0, max_retries=2)

SYSTEM = (
    "You generate a Langflow custom component as a single minimal Python file. "
    "Return ONLY one single-line JSON object (no markdown) with keys: "
    'filename, language=\"python\", code_lines=[...]. '
    "Each element of code_lines is ONE line of the file (without trailing newline). "
    "Any text outside JSON is a fatal error. "
    "No triple-quoted docstrings; use a single # header comment. "
    "The file MUST parse with ast.parse(). Keep it short."
)

def call_llm(prompt: str, temperature: float = 0.0) -> str:
    try:
        resp = _client.chat.completions.create(
            model=MODEL,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except APITimeoutError as e:
        raise TimeoutError("OpenAI request timed out") from e