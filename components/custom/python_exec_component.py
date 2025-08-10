# Execute Python code safely with timeout
class PythonExecComponent:
    display_name = "PythonExec"
    description = "Runs a short Python snippet with a timeout and returns stdout/stderr."

    def build(self, code: str, timeout_s: int = 8) -> dict:
        import sys, tempfile, subprocess, os, re, ast

        # 1) validate inputs
        if not isinstance(code, str) or not code.strip():
            raise ValueError("code must be a non-empty string")
        if not isinstance(timeout_s, int):
            raise ValueError("timeout_s must be an integer")
        if timeout_s < 1 or timeout_s > 10:
            raise ValueError("timeout_s must be between 1 and 10 seconds")

        # 2) deny-list (regex) — quick, case-insensitive scan
        banned = [
            r'\bfrom\s+socket\b', r'\bimport\s+socket\b',
            r'\bfrom\s+requests\b', r'\bimport\s+requests\b',
            r'\bfrom\s+urllib\b', r'\bimport\s+urllib\b',
            r'\bfrom\s+httpx\b', r'\bimport\s+httpx\b',
            r'\bfrom\s+ftplib\b', r'\bimport\s+ftplib\b',
            r'\bfrom\s+smtplib\b', r'\bimport\s+smtplib\b',
            r'\baiohttp\b',
            r'\bos\s*\.\s*system\s*\(',
            r'\bsubprocess\s*\.\s*(Popen|call|check_output)\s*\(',
            r'\beval\s*\(', r'\bexec\s*\(',
            r'\b__import__\s*\(', r'\bimportlib\s*\.',
            r'\bbuiltins\s*\.\s*__import__',
            r'\bopen\s*\(',
        ]
        for pat in banned:
            if re.search(pat, code, re.IGNORECASE):
                raise ValueError("code rejected by safety filter")

        # 3) static import guard via AST — more robust than regex
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"code has syntax error: {e}")

        banned_modules = {
            "socket", "requests", "urllib", "httpx", "ftplib", "smtplib",
            "aiohttp", "importlib",
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [n.name.split(".")[0] for n in getattr(node, "names", [])]
                mod = getattr(node, "module", None)
                for n in names + ([mod] if mod else []):
                    if n in banned_modules:
                        raise ValueError(f"import of {n} is not allowed")

        # 4) execute in isolated mode with timeout
        tmp = None
        try:
            tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".py", encoding="utf-8")
            tmp.write(code)
            tmp.close()
            env = {"PYTHONIOENCODING": "utf-8"}
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", tmp.name],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    env=env,
                    shell=False,  # explicit
                )
                return {
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "returncode": int(completed.returncode),
                    "timed_out": False,
                }
            except subprocess.TimeoutExpired:
                return {
                    "stdout": "",
                    "stderr": f"Timeout after {timeout_s}s",
                    "returncode": -1,
                    "timed_out": True,
                }
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
