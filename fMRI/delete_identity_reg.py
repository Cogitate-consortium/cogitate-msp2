#!/usr/bin/env python3
"""
Reset identity-registration artifacts for 2nd-level FEAT prep (activation, gPPI, FIR).

Reverses changes from ``glm/identity_reg_for_2nd_lvl.sh`` and pipeline scripts:
``activation/02_create_identity_reg_for_2nd_lvl.sh``, ``gPPI_code/05_...``,
``FIR/04_create_identity_reg_for_2nd_lvl.sh``.

For each selected analysis this tool can:

1. Remove ``job_files/create_identity_reg_for_2nd_lvl/**/*identity_reg.run`` markers.
2. Restore run-level 1st-level ``.feat`` registration dirs (``reg_bkp`` → ``reg``,
   ``reg_standard_bkp`` → ``reg_standard``, or remove partial identity ``reg/``).

Does **not** delete ``.feat`` / ``.gfeat`` trees — use ``delete_fslfeat_derivatives.py``
for that.

Supported ``--analysis`` values: ``activation``, ``gPPI``, ``FIR``, or ``all``
(intersection with folders under fslFeat that support identity reg).
Not supported: ``glm``, ``group``, ``EVC``.

Default is dry-run. Pass ``--execute`` to apply changes.

Examples::

    python delete_identity_reg.py --analysis activation --subject SC108
    python delete_identity_reg.py --analysis gPPI --execute
    python delete_identity_reg.py --analysis FIR --execute
    python delete_identity_reg.py --analysis all --execute

Exit code: 0 on success; 1 on argument errors or any failed operation.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from delete_fslfeat_derivatives import (
    CODE_SUBDIR_BY_FSLFEAT_ANALYSIS,
    FSLFEAT_ROOT,
    delete_deepest_first,
    discover_analysis_dirs,
    feat_matches_level,
    infer_code_root,
    iter_feat_under,
    resolve_analyses,
)

IDENTITY_REG_ANALYSES: tuple[str, ...] = (
    "activation",
    "gPPI",
    "FIR",
)

IDENTITY_REG_JOB_SUBDIR = "job_files/create_identity_reg_for_2nd_lvl"

_MARKER_GLOBS = ("*identity_reg.run", "*.identity_reg.run")

# Extra safety for gPPI: only revert 1st-level PPI feats.
_GPPI_1ST_BASENAME = re.compile(r"analysis-1stGLM_PPI_", re.IGNORECASE)


def normalize_subject_filter(raw: str | None) -> str | None:
    """Return sub-XXX or None."""
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    if s.startswith("sub-"):
        return s
    return f"sub-{s}"


def subject_matches(path: Path, sub_filter: str | None) -> bool:
    if sub_filter is None:
        return True
    sub_code = sub_filter[4:] if sub_filter.startswith("sub-") else sub_filter
    parts = path.parts
    return sub_filter in parts or sub_code in parts


def resolve_identity_analyses(
    fslfeat_root: Path,
    analysis_args: list[str],
) -> tuple[list[str], str | None]:
    """Resolve --analysis to supported identity-reg analysis names."""
    discovered = discover_analysis_dirs(fslfeat_root)
    supported = [a for a in discovered if a in IDENTITY_REG_ANALYSES]

    if not analysis_args:
        if supported:
            return supported, None
        return [], (
            "no supported identity-reg analyses under fslFeat "
            f"(expected any of {list(IDENTITY_REG_ANALYSES)!r})"
        )

    flat: list[str] = []
    for chunk in analysis_args:
        for part in chunk.split(","):
            part = part.strip()
            if part:
                flat.append(part)

    if not flat:
        return [], "no --analysis values"

    has_all = any(a.lower() == "all" for a in flat)
    if has_all:
        if len(flat) > 1:
            return [], "do not combine 'all' with other --analysis values"
        return supported, None

    unknown_support = [a for a in flat if a not in IDENTITY_REG_ANALYSES]
    if unknown_support:
        return (
            [],
            f"analysis(s) {unknown_support!r} do not use identity reg in this repo. "
            f"Supported: {list(IDENTITY_REG_ANALYSES)!r}",
        )

    unknown_discovered = [a for a in flat if a not in discovered]
    if unknown_discovered:
        return (
            [],
            f"unknown fslFeat folder(s): {unknown_discovered!r}. "
            f"Known under {fslfeat_root}: {discovered!r}",
        )

    return flat, None


def marker_roots_for_analysis(code_root: Path, analysis: str) -> list[Path]:
    """Job marker search roots for one fslFeat analysis name."""
    subdir = CODE_SUBDIR_BY_FSLFEAT_ANALYSIS.get(analysis)
    if not subdir:
        return []
    base = (code_root / subdir / IDENTITY_REG_JOB_SUBDIR).resolve()
    if not base.is_dir():
        return []

    if analysis == "FIR":
        variant = base / "FIR"
        if variant.is_dir():
            return [variant]
        return [base]

    return [base]


def collect_marker_files(
    code_root: Path,
    analyses: list[str],
    subject_filter: str | None,
) -> list[Path]:
    """Marker files under create_identity_reg_for_2nd_lvl."""
    out: list[Path] = []
    seen: set[Path] = set()

    for analysis in analyses:
        for root in marker_roots_for_analysis(code_root, analysis):
            if not root.is_dir():
                continue
            for pattern in _MARKER_GLOBS:
                for path in root.rglob(pattern):
                    if not path.is_file():
                        continue
                    if not subject_matches(path, subject_filter):
                        continue
                    rp = path.resolve()
                    if rp not in seen:
                        seen.add(rp)
                        out.append(path)

    out.sort(key=lambda p: str(p))
    return out


def feat_matches_identity_analysis(feat_dir: Path, analysis: str) -> bool:
    """Extra basename checks beyond feat_matches_level(..., '1st')."""
    name = feat_dir.name
    namel = name.lower()
    if analysis == "gPPI":
        return bool(_GPPI_1ST_BASENAME.search(name))
    if analysis == "FIR":
        return (
            "analysis-1stglm_fir_" in namel
            and "fir_face" not in namel
            and "fir_object" not in namel
        )
    return True  # activation


def collect_feat_dirs(
    fslfeat_root: Path,
    analyses: list[str],
    subject_filter: str | None,
) -> list[Path]:
    """Run-level 1st-level .feat dirs that may have identity reg applied."""
    out: list[Path] = []
    seen: set[Path] = set()

    for analysis in analyses:
        sub = fslfeat_root / analysis
        if not sub.is_dir():
            continue
        for feat_dir in iter_feat_under(sub):
            if not feat_matches_level(feat_dir, "1st"):
                continue
            if not feat_matches_identity_analysis(feat_dir, analysis):
                continue
            if not subject_matches(feat_dir, subject_filter):
                continue
            rp = feat_dir.resolve()
            if rp not in seen:
                seen.add(rp)
                out.append(feat_dir)

    out.sort(key=lambda p: str(p))
    return out


def revert_identity_reg_in_feat(
    feat_dir: Path,
    *,
    execute: bool,
    remove_mean_func: bool,
    remove_reg_bkp: bool,
) -> int:
    """
    Undo identity reg in one 1st-level .feat directory.
    Returns number of errors.
    """
    if not feat_dir.is_dir():
        return 0

    errors = 0
    reg = feat_dir / "reg"
    reg_bkp = feat_dir / "reg_bkp"
    reg_standard = feat_dir / "reg_standard"
    reg_standard_bkp = feat_dir / "reg_standard_bkp"
    mean_func = feat_dir / "mean_func.nii.gz"
    ident_mat = reg / "example_func2standard.mat"

    def do_restore_reg() -> None:
        nonlocal errors
        if reg_bkp.is_dir():
            msg = f"restore reg from reg_bkp: {feat_dir}"
            if execute:
                try:
                    if reg.exists():
                        shutil.rmtree(reg)
                    reg_bkp.rename(reg)
                    print(f"restored: {msg}")
                except OSError as exc:
                    print(f"ERROR {msg}: {exc}", file=sys.stderr)
                    errors += 1
            else:
                print(f"would restore: {msg}")
            if remove_reg_bkp and reg_bkp.is_dir():
                if execute:
                    try:
                        shutil.rmtree(reg_bkp)
                        print(f"removed: {reg_bkp}")
                    except OSError as exc:
                        print(f"ERROR removing {reg_bkp}: {exc}", file=sys.stderr)
                        errors += 1
                else:
                    print(f"would remove: {reg_bkp}")
        elif reg.is_dir() and ident_mat.is_file():
            msg = f"remove identity reg dir: {reg}"
            if execute:
                try:
                    shutil.rmtree(reg)
                    print(f"removed: {msg}")
                except OSError as exc:
                    print(f"ERROR {msg}: {exc}", file=sys.stderr)
                    errors += 1
            else:
                print(f"would remove: {msg}")

    def do_restore_reg_standard() -> None:
        nonlocal errors
        if not reg_standard_bkp.is_dir():
            return
        msg = f"restore reg_standard from reg_standard_bkp: {feat_dir}"
        if execute:
            try:
                if reg_standard.exists():
                    shutil.rmtree(reg_standard)
                reg_standard_bkp.rename(reg_standard)
                print(f"restored: {msg}")
            except OSError as exc:
                print(f"ERROR {msg}: {exc}", file=sys.stderr)
                errors += 1
        else:
            print(f"would restore: {msg}")
        if remove_reg_bkp and reg_standard_bkp.is_dir():
            if execute:
                try:
                    shutil.rmtree(reg_standard_bkp)
                    print(f"removed: {reg_standard_bkp}")
                except OSError as exc:
                    print(f"ERROR removing {reg_standard_bkp}: {exc}", file=sys.stderr)
                    errors += 1
            else:
                print(f"would remove: {reg_standard_bkp}")

    do_restore_reg()
    do_restore_reg_standard()

    if remove_mean_func and mean_func.is_file():
        if execute:
            try:
                mean_func.unlink()
                print(f"removed: {mean_func}")
            except OSError as exc:
                print(f"ERROR removing {mean_func}: {exc}", file=sys.stderr)
                errors += 1
        else:
            print(f"would remove: {mean_func}")

    return errors


def list_supported_analyses(fslfeat_root: Path) -> int:
    discovered = discover_analysis_dirs(fslfeat_root)
    print(f"fslFeat root: {fslfeat_root}")
    print("Identity-reg supported (repo):")
    for name in IDENTITY_REG_ANALYSES:
        present = name in discovered
        code = CODE_SUBDIR_BY_FSLFEAT_ANALYSIS.get(name, "?")
        print(f"  {name}: fslFeat={'yes' if present else 'no'}, code={code}")
    print()
    print("Other fslFeat top-level folders (no identity-reg cleanup here):")
    for name in discovered:
        if name not in IDENTITY_REG_ANALYSES:
            print(f"  {name}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Reset identity reg in run-level 1st-level .feat dirs and clear "
            "create_identity_reg job markers (activation, gPPI, FIR)."
        )
    )
    p.add_argument(
        "--fslfeat-root",
        type=Path,
        default=FSLFEAT_ROOT,
        help=f"fslFeat root (default: {FSLFEAT_ROOT})",
    )
    p.add_argument(
        "--analysis",
        action="append",
        metavar="NAME",
        default=[],
        help=(
            "fslFeat folder: activation, gPPI, FIR. "
            "Repeat or comma-separate. Omit with no 'all' to process all supported "
            "folders present. Use 'all' for every supported analysis under fslFeat."
        ),
    )
    p.add_argument(
        "--code-root",
        type=Path,
        default=None,
        help="fMRI_exp2 repo root (default: sibling of fslFeat)",
    )
    p.add_argument(
        "--subject",
        metavar="ID",
        default=None,
        help="Limit to one subject (SC108 or sub-SC108)",
    )
    p.add_argument(
        "--list-analyses",
        action="store_true",
        help="List identity-reg-supported analyses and exit",
    )
    p.add_argument(
        "--no-markers",
        action="store_true",
        help="Do not remove job_files identity_reg markers",
    )
    p.add_argument(
        "--no-feat",
        action="store_true",
        help="Do not revert reg/ inside .feat directories",
    )
    p.add_argument(
        "--remove-mean-func",
        action="store_true",
        help="Also delete mean_func.nii.gz in targeted .feat dirs",
    )
    p.add_argument(
        "--remove-reg-bkp",
        action="store_true",
        help="After restore, remove reg_bkp/ and reg_standard_bkp/",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Apply changes (default is dry-run)",
    )
    args = p.parse_args()

    root = args.fslfeat_root.resolve()
    subject_filter = normalize_subject_filter(args.subject)

    if args.list_analyses:
        if not root.is_dir():
            print(f"ERROR: fslFeat root missing: {root}", file=sys.stderr)
            return 1
        return list_supported_analyses(root)

    analyses, err = resolve_identity_analyses(root, args.analysis)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    if not analyses:
        print("Nothing to do: no supported identity-reg analyses found under fslFeat.")
        return 0

    if not root.is_dir():
        print(f"ERROR: fslFeat root is not a directory: {root}", file=sys.stderr)
        return 1

    code_root = (args.code_root or infer_code_root(root)).resolve()
    mode = "DELETE" if args.execute else "DRY-RUN (pass --execute to apply)"

    print(f"fslFeat root: {root}")
    print(f"Code root: {code_root}")
    print(f"Analyses: {', '.join(analyses)}")
    if subject_filter:
        print(f"Subject filter: {subject_filter}")
    print(f"Mode: {mode}")
    print()

    errors = 0

    if not args.no_markers:
        if not code_root.is_dir():
            print(
                f"WARN: code root missing; skipping markers: {code_root}",
                file=sys.stderr,
            )
        else:
            markers = collect_marker_files(code_root, analyses, subject_filter)
            print(f"Identity-reg marker files: {len(markers)}")
            print()
            errors += delete_deepest_first(markers, execute=args.execute)
    else:
        print("Skipping markers (--no-markers).")
        print()

    if not args.no_feat:
        feats = collect_feat_dirs(root, analyses, subject_filter)
        print(f"1st-level .feat dirs to revert: {len(feats)}")
        print()
        for feat_dir in feats:
            errors += revert_identity_reg_in_feat(
                feat_dir,
                execute=args.execute,
                remove_mean_func=args.remove_mean_func,
                remove_reg_bkp=args.remove_reg_bkp,
            )
    else:
        print("Skipping .feat reg revert (--no-feat).")
        print()

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
