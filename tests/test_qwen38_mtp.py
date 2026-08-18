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


def _backend_method(name: str) -> ast.FunctionDef | None:
    source_path = (
        Path(__file__).parents[1] / "harness" / "executors" / "qwen38_mtp.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    backend_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "MTPDraftBackend"
    )
    return next(
        (
            node
            for node in backend_class.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )


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


def test_ddtree_topk_comes_from_each_native_mtp_draft_step():
    method = _backend_method("draft_with_topk")

    assert method is not None
    called_attributes = {
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "predict_block_with_topk" in called_attributes
    assert "logits_from_hidden" not in called_attributes


def test_ddtree_branch_generation_preserves_full_forced_prefix():
    method = _backend_method("draft_branch_blocks_batch")

    assert method is not None
    called_attributes = {
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "predict_candidate" in called_attributes


def test_native_mtp_backend_owns_an_explicit_thread_local_draft_stream():
    source_path = (
        Path(__file__).parents[1] / "harness" / "executors" / "qwen38_mtp.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "mx.new_stream(mx.gpu)" in source
    assert "threading.get_ident()" in source
    assert "with mx.stream(self._draft_stream):" in source


def test_ddtree_batches_topk_materialization_after_all_mtp_steps():
    _, method = _model_method("_predict_candidate_with_topk")

    assert method is not None
    loop = next(node for node in ast.walk(method) if isinstance(node, ast.For))
    loop_calls = {
        node.func.id
        for node in ast.walk(loop)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "top_ids_and_values_desc" not in loop_calls
