from pathlib import Path
from app.stages.requirements import run_requirements
doc = run_requirements("Create a PythonExec component that runs short code with a timeout and returns stdout/stderr.", workspace=Path("/tmp/ai_agency_demo"))
print(doc.keys())
print(doc["inputs"])
print("Artifact written to:", "/tmp/ai_agency_demo/requirements.json")
