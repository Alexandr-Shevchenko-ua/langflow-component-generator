from pathlib import Path
from app.stages.requirements import run_requirements

SPEC = "Create a PythonExec component that runs short code with a timeout and returns stdout/stderr."

def test_stage1_generates_json(tmp_path: Path):
    doc = run_requirements(SPEC, workspace=tmp_path)
    # shape
    must_keys = {
        "raw_spec","purpose","context","inputs","outputs",
        "constraints","non_functional","acceptance_criteria",
        "risks","assumptions","open_questions"
    }
    assert must_keys.issubset(doc.keys())
    # critical fields not empty
    assert doc["raw_spec"] == SPEC
    assert isinstance(doc["inputs"], list) and len(doc["inputs"]) >= 1
    assert isinstance(doc["outputs"], list) and len(doc["outputs"]) >= 1
def _min(doc, key, n):
    assert isinstance(doc[key], list) and len(doc[key]) >= n, f"{key} must have >= {n}"

def test_semantic_minimums(tmp_path):
    from app.stages.requirements import run_requirements, _MIN_COUNTS
    doc = run_requirements("Create a PythonExec component that runs short code with a timeout and returns stdout/stderr.", workspace=tmp_path)
    assert doc["purpose"] != doc["raw_spec"]
    for k,n in _MIN_COUNTS.items():
        _min(doc, k, n)
