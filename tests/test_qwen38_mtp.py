import ast
from pathlib import Path


def _model_method(name: str) -> tuple[Path, ast.FunctionDef | None]:
    source_path = (
        Path(__file__).parents[1] / "harness" / "executors" / "qwen38_mtp.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    model_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen38MTPModel"
    )
    method = next(
        (
            node
            for node in model_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == name
        ),
        None,
    )
    return source_path, method


def test_native_mtp_preserves_raw_target_hidden_for_its_own_projection():
    source_path, projection = _model_method("project_target_hidden")

    assert projection is not None, "DFlash requires project_target_hidden"

    namespace = {}
    isolated = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            projection,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(isolated), source_path, "exec"), namespace)
    target_hidden = object()

    projected = namespace["project_target_hidden"](None, target_hidden)

    assert projected is target_hidden


def test_native_mtp_does_not_apply_target_final_norm_before_mtp_norm():
    _, predict_block = _model_method("predict_block")

    assert predict_block is not None
    called_attributes = {
        node.func.attr
        for node in ast.walk(predict_block)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "text_model" not in called_attributes
