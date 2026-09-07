from __future__ import annotations

import ast
from pathlib import Path


TRAINER = Path(__file__).resolve().parents[1] / "tools" / "audio_fix_int8_train.py"


def _argument_defaults():
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    defaults = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        name = node.args[0].value
        if not isinstance(name, str) or not name.startswith("--"):
            continue
        default = next((kw.value for kw in node.keywords if kw.arg == "default"), None)
        if default is not None:
            defaults[name] = ast.literal_eval(default)
    return defaults


def test_direct_int8_trainer_defaults_to_canonical_released_stack():
    defaults = _argument_defaults()
    assert defaults["--stage-b-strength"] == 1.0
    assert defaults["--turbo-strength"] == 1.0
    assert defaults["--global-gate-mode"] == "checkpoint"
