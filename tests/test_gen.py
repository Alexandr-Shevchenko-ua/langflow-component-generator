import base64, json, ast

def test_bundle_shape():
    sample = {
        "filename": "components/custom/greeter_component.py",
        "encoding": "base64",
        "language": "python",
        "code": base64.b64encode(b"print('ok')").decode("ascii")
    }
    assert sample["encoding"] == "base64"
    code = base64.b64decode(sample["code"])
    ast.parse(code.decode("utf-8"))