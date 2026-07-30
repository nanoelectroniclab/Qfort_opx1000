# Repository Guidance

## Scope

- This repository controls a Quantum Machines OPX1000 superconducting-qubit environment.
- `main` is the calibration baseline. Keep calibration nodes in `calibrations/`, their reusable implementation in `calibration_utils/`, and shared machine code in `quam_config/`.
- Develop each experiment on its own `experiment/<name>` branch and sibling worktree named `Qfort_opx1000-<name>`. Do not add experiment-only code to `main`.
- In an experiment worktree, keep experiment code under `experiments/<name>/`.

## Environment

- Use Python 3.12.
- Create an independent environment in every worktree with:

  ```bash
  uv sync --frozen --group dev
  ```

- Do not share or symlink `.venv` directories between worktrees. Editable installs contain checkout-specific paths.
- Run tools through the active worktree environment, for example `.venv/bin/python` and `.venv/bin/qualibrate`.

## Hardware and State Safety

- Do not connect to the OPX1000, execute QUA programs, interrupt jobs, or start hardware-facing services unless the user explicitly requests it.
- Prefer static checks and simulation when they can answer the question.
- Treat `quam_state_qfort/config/` as the live calibrated machine state. Inspect proposed changes carefully and do not update or commit state changes unless the task explicitly requires it.
- Experiment worktrees read the shared live state configured by QUAlibrate. Experiments must not silently overwrite that state.
- Keep credentials, tokens, private network secrets, and measurement data out of Git.

## Documentation

- Put behavioral requirements, experiment plans, and acceptance criteria in `docs/spec/`.
- Put operating procedures and project-specific setup instructions in `docs/manual/`.
- Put API notes, hardware references, compatibility notes, and links to authoritative external material in `docs/reference/`.
- Track human-authored Markdown and small, redistributable supporting files in Git so every worktree receives the same context.
- Do not commit proprietary or license-restricted vendor documents without permission. Prefer an indexed link or an approved external storage location for large binary manuals.
- Record website references as Markdown entries with a descriptive title, stable or versioned URL, access date, and a short note explaining when to use the source.
- When behavior or operating procedures change, update the corresponding documentation in the same branch.

## Development and Verification

- Preserve unrelated user changes and inspect `git status` before editing.
- Follow the existing Black line length of 120 characters.
- At minimum, syntax-check changed Python modules. Run focused tests when available.
- Hardware-free verification must not be presented as proof of successful OPX1000 execution.
- Report the worktree, branch, tests run, and any uncommitted files when handing work back.

## Git Workflow

- Keep `main` clean and synchronized with `origin/main`.
- Rebase experiment branches onto `main` when they need updated calibration or shared code.
- Do not merge an experiment branch into `main` unless the user explicitly promotes part of it to shared or calibration code.
- Never delete a worktree containing untracked experiment files without first preserving them.
