# Component for executing Python code securely
import sys
import tempfile
import subprocess
import re
class PythonExecComponent:
    display_name = "PythonExec"
    description = "Runs a short Python snippet with a timeout and returns stdout/stderr."
    def build(self, code: str, timeout_s: int = 3) -> dict:
        if not code.strip():
            raise ValueError("Code must be a non-empty string.")
        if not (1 <= timeout_s <= 10):
            raise ValueError("Timeout must be between 1 and 10 seconds.")
        banned_patterns = [
            r'\bfrom\s+socket\b',
            r'\bimport\s+socket\b',
            r'\bfrom\s+requests\b',
            r'\bimport\s+requests\b',
            r'\bfrom\s+urllib\b',
            r'\bimport\s+urllib\b',
            r'\bfrom\s+httpx\b',
            r'\bimport\s+httpx\b',
            r'\bfrom\s+ftplib\b',
            r'\bimport\s+ftplib\b',
            r'\bfrom\s+smtplib\b',
            r'\bimport\s+smtplib\b',
            r'\baiohttp\b',
            r'\bos\.system\s*\(',
            r'\bsubprocess\.(Popen|call|check_output)\s*\(',
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'\b__import__\s*\(',
            r'\bimportlib\.',
            r'\bbuiltins\.__import__',
            r'\bopen\s*\(',
        ]
        for pattern in banned_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                raise ValueError("Code contains banned patterns.")
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        try:
            result = subprocess.run([sys.executable, "-I", tmp_path],
                                   capture_output=True, text=True, timeout=timeout_s,
                                   env={"PYTHONIOENCODING": "utf-8"})
            return {"stdout": result.stdout, "stderr": result.stderr,
                   "returncode": result.returncode, "timed_out": False}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": f"Timeout after {timeout_s}s",
                    "returncode": -1, "timed_out": True}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass