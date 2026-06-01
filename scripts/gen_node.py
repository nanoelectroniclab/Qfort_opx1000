#!/usr/bin/env python3
"""Interactive CLI to scaffold QualibrationNode scripts and utils modules."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Literal

NodeType = Literal["calibration", "experiment"]

PREFIX_PATTERN = re.compile(r"^(\d+)([a-z]?)_", re.IGNORECASE)
MODULE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

UTILS_TEMPLATE_FILES = {
    "__init__.py": "utils/__init__.py.template",
    "parameters.py": "utils/parameters.py.template",
    "analysis.py": "utils/analysis.py.template",
    "plotting.py": "utils/plotting.py.template",
}


@dataclass(frozen=True)
class ScaffoldPlan:
    node_type: NodeType
    module_name: str
    node_name: str
    main_script: Path
    utils_dir: Path
    utils_files: dict[str, Path]
    node_template: Path
    templates_dir: Path
    variables: dict[str, str]


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root by walking up from the current directory."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "calibrations").is_dir():
            return candidate
    raise SystemExit("Could not find repository root (expected pyproject.toml and calibrations/).")


def normalize_name(raw: str) -> str:
    """Convert user input to lowercase snake_case module name."""
    name = raw.strip()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = name.lower()
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        raise SystemExit("Node name cannot be empty.")
    if not MODULE_NAME_PATTERN.match(name):
        raise SystemExit(f"Invalid module name '{name}'. Use letters, digits, and underscores only.")
    return name


def title_from_module(module_name: str) -> str:
    return module_name.replace("_", " ").upper()


def parse_calibration_prefixes(calibrations_dir: Path) -> list[tuple[int, str]]:
    prefixes: list[tuple[int, str]] = []
    for path in calibrations_dir.glob("*.py"):
        match = PREFIX_PATTERN.match(path.name)
        if not match:
            continue
        number = int(match.group(1))
        if 1 <= number <= 99:
            prefixes.append((number, match.group(2).lower()))
    return prefixes


def suggest_calibration_prefix(calibrations_dir: Path) -> str:
    """Suggest the next calibration filename prefix in the 01-99 range."""
    prefixes = parse_calibration_prefixes(calibrations_dir)
    if not prefixes:
        return "01a"

    max_number = max(number for number, _ in prefixes)
    letters_for_max = sorted(letter for number, letter in prefixes if number == max_number)

    if letters_for_max:
        last_letter = letters_for_max[-1]
        if not last_letter:
            return f"{max_number}a"
        if last_letter < "z":
            return f"{max_number}{chr(ord(last_letter) + 1)}"
        if max_number >= 99:
            raise SystemExit("No available calibration prefix in the 01-99 range.")
        return f"{max_number + 1}a"

    if max_number >= 99:
        raise SystemExit("No available calibration prefix in the 01-99 range.")
    return f"{max_number + 1}a"


def validate_prefix(prefix: str) -> str:
    cleaned = prefix.strip().lower()
    if not re.fullmatch(r"\d+[a-z]?", cleaned):
        raise SystemExit(f"Invalid prefix '{prefix}'. Expected format like '22' or '22a'.")
    number = int(re.match(r"\d+", cleaned).group())
    if not 1 <= number <= 99:
        raise SystemExit(f"Prefix number must be between 01 and 99, got '{number}'.")
    return cleaned


def prompt_node_type() -> NodeType:
    print("Node type:")
    print("  [c]alibrations")
    print("  [e]xperiments")
    choice = input("> ").strip().lower() or "c"
    if choice in {"c", "calibration", "calibrations"}:
        return "calibration"
    if choice in {"e", "experiment", "experiments"}:
        return "experiment"
    raise SystemExit(f"Unknown choice '{choice}'. Use 'c' or 'e'.")


def prompt_prefix(suggested: str, module_name: str) -> str | None:
    print(f"Suggested prefix: {suggested} -> {suggested}_{module_name}.py")
    print("Press Enter to accept, type a custom prefix (e.g. 17b), or type 'n' to skip prefix.")
    choice = input("> ").strip().lower()
    if choice in {"", "y", "yes"}:
        return suggested
    if choice in {"n", "no", "none"}:
        return None
    return validate_prefix(choice)


def render_template(template_path: Path, variables: dict[str, str]) -> str:
    content = template_path.read_text(encoding="utf-8")
    return Template(content).safe_substitute(variables)


def build_plan(
    repo_root: Path,
    node_type: NodeType,
    module_name: str,
    prefix: str | None,
) -> ScaffoldPlan:
    templates_dir = Path(__file__).resolve().parent / "templates"
    title = title_from_module(module_name)

    if node_type == "calibration":
        node_name = f"{prefix}_{module_name}" if prefix else module_name
        main_script = repo_root / "calibrations" / f"{node_name}.py"
        utils_dir = repo_root / "calibration_utils" / module_name
        node_template = templates_dir / "calibration" / "node.py.template"
    else:
        node_name = module_name
        main_script = repo_root / "experiments" / f"{module_name}.py"
        utils_dir = repo_root / "experiments" / "utils" / module_name
        node_template = templates_dir / "experiment" / "node.py.template"

    variables = {
        "module_name": module_name,
        "node_name": node_name,
        "title": title,
    }

    utils_files: dict[str, Path] = {}
    for filename, template_rel in UTILS_TEMPLATE_FILES.items():
        utils_files[filename] = utils_dir / filename

    return ScaffoldPlan(
        node_type=node_type,
        module_name=module_name,
        node_name=node_name,
        main_script=main_script,
        utils_dir=utils_dir,
        utils_files=utils_files,
        node_template=node_template,
        templates_dir=templates_dir,
        variables=variables,
    )


def collect_existing_paths(plan: ScaffoldPlan) -> list[Path]:
    paths = [plan.main_script, *plan.utils_files.values()]
    return [path for path in paths if path.exists()]


def confirm_plan(plan: ScaffoldPlan) -> None:
    print("\nWill create:")
    print(f"  {plan.main_script.relative_to(find_repo_root())}")
    for path in plan.utils_files.values():
        print(f"  {path.relative_to(find_repo_root())}")
    choice = input("\nProceed? [Y/n] > ").strip().lower()
    if choice in {"n", "no"}:
        raise SystemExit("Aborted.")


def scaffold(plan: ScaffoldPlan) -> list[Path]:
    existing = collect_existing_paths(plan)
    if existing:
        lines = "\n".join(f"  - {path}" for path in existing)
        raise SystemExit(f"Refusing to overwrite existing files:\n{lines}")

    plan.utils_dir.mkdir(parents=True, exist_ok=False)
    created = [plan.main_script]

    plan.main_script.write_text(render_template(plan.node_template, plan.variables), encoding="utf-8")

    for filename, template_rel in UTILS_TEMPLATE_FILES.items():
        target = plan.utils_files[filename]
        template_path = plan.templates_dir / template_rel
        target.write_text(render_template(template_path, plan.variables), encoding="utf-8")
        created.append(target)

    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scaffold a QualibrationNode script and utils module.")
    parser.add_argument("name", help="Node name (e.g. exp_A or my_experiment)")
    parser.add_argument(
        "--type",
        choices=["calibration", "experiment", "c", "e"],
        help="Node type. If omitted, an interactive prompt is shown.",
    )
    parser.add_argument(
        "--prefix",
        help="Calibration filename prefix (e.g. 22a). If omitted for calibrations, a suggestion is shown.",
    )
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="Create calibration script without numeric prefix.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    args = parser.parse_args(argv)

    repo_root = find_repo_root()
    module_name = normalize_name(args.name)

    if args.type in {"c", "calibration"}:
        node_type: NodeType = "calibration"
    elif args.type in {"e", "experiment"}:
        node_type = "experiment"
    elif args.type is None:
        node_type = prompt_node_type()
    else:
        node_type = args.type  # type: ignore[assignment]

    prefix: str | None = None
    if node_type == "calibration":
        if args.no_prefix:
            prefix = None
        elif args.prefix:
            prefix = validate_prefix(args.prefix)
        else:
            suggested = suggest_calibration_prefix(repo_root / "calibrations")
            prefix = prompt_prefix(suggested, module_name)

    plan = build_plan(repo_root, node_type, module_name, prefix)

    if not args.yes:
        confirm_plan(plan)

    created = scaffold(plan)

    print("\nCreated:")
    for path in created:
        print(f"  {path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
