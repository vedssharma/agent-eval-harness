"""Load and validate task directories."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, TypeAdapter

from .schemas import ReferenceStep, TaskSpec

_REFERENCE = TypeAdapter(list[ReferenceStep])


class Task(BaseModel):
    spec: TaskSpec
    root: Path

    @property
    def repo_path(self) -> Path:
        return self.root / self.spec.repo_dir

    @property
    def hidden_tests_path(self) -> Path:
        return self.root / self.spec.hidden_tests_dir

    @property
    def solution_path(self) -> Path | None:
        return self.root / self.spec.solution_dir if self.spec.solution_dir else None

    def reference_path(self) -> list[ReferenceStep]:
        if not self.spec.reference_path_file:
            return []
        raw = json.loads((self.root / self.spec.reference_path_file).read_text())
        return _REFERENCE.validate_python(raw)


def load_task(root: Path | str) -> Task:
    root = Path(root).resolve()
    spec = TaskSpec.model_validate(yaml.safe_load((root / "task.yaml").read_text()))
    if spec.id != root.name:
        raise ValueError(f"task id {spec.id!r} must match directory name {root.name!r}")
    task = Task(spec=spec, root=root)
    for label, path in [("repo", task.repo_path), ("hidden tests", task.hidden_tests_path)]:
        if not path.is_dir():
            raise FileNotFoundError(f"{label} directory missing: {path}")
    task.reference_path()  # validate eagerly
    return task


def load_all(tasks_dir: Path | str) -> list[Task]:
    return [load_task(p) for p in sorted(Path(tasks_dir).iterdir()) if (p / "task.yaml").exists()]
