import os
from litellm import completion

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def call_llm(prompt: str, temperature: float = 0.1) -> str:
    resp = completion(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message["content"]