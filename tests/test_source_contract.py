"""Guard against reintroducing retired workflows into the public pipeline."""

import ast
import re
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SourceContractTests(unittest.TestCase):
    def test_retired_names_and_altered_logging_are_absent(self):
        retired = {
            "fmri_encoder",
            "fmri_recon",
            "fMRIs",
            "scale_metric_from_epoch",
            "ei_semantic_wandb_scale_factor",
            "legacy_auxiliary_from_full",
        }
        for folder in ["model", "data_loader", "eval"]:
            for filename in (ROOT / folder).glob("*.py"):
                names = {
                    n.id
                    for n in ast.walk(ast.parse(filename.read_text()))
                    if isinstance(n, ast.Name)
                }
                names |= {
                    n.attr
                    for n in ast.walk(ast.parse(filename.read_text()))
                    if isinstance(n, ast.Attribute)
                }
                self.assertFalse(names & retired, str(filename))

    def test_yaml_sgm_targets_exist(self):
        def exists(module, symbol):
            source = ROOT / "generative_models" / (module.replace(".", "/") + ".py")
            package = module.rpartition(".")[0]
            if not source.is_file():
                source = (
                    ROOT
                    / "generative_models"
                    / module.replace(".", "/")
                    / "__init__.py"
                )
                package = module
            if not source.is_file():
                return False
            tree = ast.parse(source.read_text())
            if symbol in {
                n.name
                for n in tree.body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef))
            }:
                return True
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if (alias.asname or alias.name) == symbol:
                            target = (
                                importlib.util.resolve_name(
                                    "." * node.level + (node.module or ""), package
                                )
                                if node.level
                                else node.module
                            )
                            return exists(target, alias.name)
            return False

        for filename in (ROOT / "generative_models/configs").rglob("*.yaml"):
            for target in re.findall(
                r"target:\s*['\"]?(sgm\.[\w.]+)", filename.read_text()
            ):
                module, symbol = target.rsplit(".", 1)
                self.assertTrue(exists(module, symbol), target)

    def test_caption_input_is_fixed_to_generated_images(self):
        source = (ROOT / "eval/caption_keyframe.py").read_text()
        self.assertIn('input_dir / "all_recons.pt"', source)
        self.assertNotIn("--images_file", source)
