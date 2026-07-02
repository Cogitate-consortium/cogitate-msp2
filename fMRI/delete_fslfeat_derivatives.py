#!/usr/bin/env python3
"""
Remove FSL FEAT derivative directories under ``fslFeat/<analysis>/``.

Select top-level analysis folder(s) with ``--analysis`` (repeat or comma-separated;
use ``all`` for every non-subject directory under the fslFeat root). Restrict
which ``.feat`` / ``.gfeat`` trees are targeted with ``--level`` (same meaning as
``glm/check_FSLFeat_outputs.py`` for 1st/2nd; ``3rd`` matches basenames containing
``analysis-3rdGLM`` **or** group-level outputs under a ``group/`` directory, e.g.
``activation/group/seen_vs_unseen_F.gfeat``, ``FIR/group/N*_..._3rdGLM_*.gfeat``,
and legacy ``group/ses-V2/activation/*.gfeat`` when clearing activation 3rd level).

By default, also removes the matching ``job_files`` trees under the repository
pipeline folders (e.g. ``activation/job_files/run_activation_2nd_level`` when
``--analysis activation --level 2nd``) so Slurm markers and logs are cleared and
pipelines can be resubmitted. Use ``--no-job-files`` to only delete FEAT dirs.

For some analyses, non-FEAT derivative trees under ``.../`` are also
removed (e.g. ``gppi_timecourse`` for gPPI 1st level, ``evc_rois`` for EVC).
Legacy ``*_timecourse_*.txt`` files under ``gppi_seeds`` are removed when clearing
gPPI 1st-level outputs.

For ``--analysis EVC`` (or ``all``), also removes EVCLoc pipeline artifacts:
``logfilechecks/*EVCLoc*``, ``regressoreventfiles/.../*EVCLoc*``, BIDS
``*EVCLoc*events.tsv``, and ``EVC/inclusion report/``.

Use ``--no-extra-derivatives`` to skip non-FEAT paths (FEAT + job_files still run).

``--code-root`` defaults to ``<fslFeat-parent>/fMRI_exp2`` (sibling of ``fslFeat``
under ``.../``).

Default is dry-run. Pass ``--execute`` to delete. Deepest FEAT paths are removed
first so nested ``cope*.feat`` inside ``.gfeat`` are cleaned before their parent.

To reset only identity ``reg/`` and ``create_identity_reg`` job markers (without
deleting ``.feat`` trees), use ``delete_identity_reg.py`` in this directory.

Exit code: 0 on success; 1 if any delete fails or invalid arguments.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

BIDS_ROOT = Path("/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids")
FSLFEAT_ROOT = BIDS_ROOT / "derivatives/fslFeat"

_RUN_IN_NAME = re.compile(r"_run-\d+_")
# Activation 3rd-level short output labels (also match trailing + from FEAT reruns)
_ACTIVATION_GROUP_GFEAT = re.compile(
    r"^seen_vs_unseen_[FO](\+)?\.gfeat$", re.IGNORECASE
)

# fslFeat top-level folder -> directory under code root containing job_files/
CODE_SUBDIR_BY_FSLFEAT_ANALYSIS: dict[str, str] = {
    "activation": "activation",
    "gPPI": "gPPI_code",
    "FIR": "FIR",
    "EVC": "EVC",
    "glm": "glm",
    # Group-level FEAT outputs use activation’s 3rd-level Slurm job layout.
    "group": "activation",
}

# Per fslFeat analysis: which job_files/* roots to remove for each --level.
# Paths are relative to CODE_SUBDIR (e.g. "job_files/run_activation_2nd_level").
JOB_SUBDIRS_BY_ANALYSIS_LEVEL: dict[str, dict[str, tuple[str, ...]]] = {
    "activation": {
        "1st": (),
        "2nd": (
            "job_files/run_activation_2nd_level",
            "job_files/create_identity_reg_for_2nd_lvl",
        ),
        "3rd": (
            "job_files/run_activation_3rd_level",
            "job_files/create_identity_reg_for_3rd_lvl",
        ),
    },
    "gPPI": {
        "1st": (
            "job_files/run_PPI_1st_level",
            "job_files/make_timecourse_files",
        ),
        "2nd": (
            "job_files/run_PPI_2nd_level",
            "job_files/create_identity_reg_for_2nd_lvl",
        ),
        "3rd": (
            "job_files/run_PPI_3rd_level",
            "job_files/create_identity_reg_for_3rd_lvl",
        ),
    },
    "FIR": {
        "1st": ("job_files/run_FIR_1st_level",),
        "2nd": (
            "job_files/run_FIR_2nd_level",
            "job_files/create_identity_reg_for_2nd_lvl",
        ),
        "3rd": (
            "job_files/run_FIR_3rd_level",
            "job_files/create_identity_reg_for_3rd_lvl",
        ),
    },
    "EVC": {
        "1st": (
            "job_files/run_EVC_1st_level",
            "job_files/run_make_anat_masks",
            "job_files/run_make_evc_roi",
        ),
        "2nd": (),
        "3rd": (),
    },
    "glm": {
        "1st": ("job_files/check_fslfeat_1st",),
        "2nd": ("job_files/check_fslfeat_2nd",),
        "3rd": (),
    },
    "group": {
        "1st": (),
        "2nd": (),
        "3rd": (
            "job_files/run_activation_3rd_level",
            "job_files/create_identity_reg_for_3rd_lvl",
        ),
    },
}

# Non-FEAT derivative directory names under .../ (sibling of fslFeat).
EXTRA_DERIV_DIRS_BY_ANALYSIS_LEVEL: dict[str, dict[str, tuple[str, ...]]] = {
    "gPPI": {"1st": ("gppi_timecourse",)},
    "EVC": {"1st": ("evc_rois",)},
}


def _has_group_output_ancestor(feat_dir: Path) -> bool:
    """True if ``feat_dir`` lives under a ``group/`` folder (pipeline group-level FEAT)."""
    for parent in feat_dir.parents:
        if parent.name == "group":
            return True
        if parent.name.startswith("sub-"):
            return False
    return False


def feat_matches_level(feat_dir: Path, level: str) -> bool:
    """
    Same rules as glm/check_FSLFeat_outputs.feat_matches_level, plus 3rd level.

    - all: no filter
    - 1st: run-level first-level (_run-N_ + 1stGLM / 1stROI in basename)
    - 2nd: basename includes analysis-2ndGLM
    - 3rd: basename includes analysis-3rdGLM, or group-level ``.gfeat`` under
      ``<pipeline>/group/`` (activation short names ``seen_vs_unseen_*.gfeat``, FIR/gPPI
      ``N*_..._3rdGLM_*.gfeat``), excluding subject-level ``sub-*`` trees
    """
    lev = (level or "all").strip().lower()
    if lev in ("", "all", "both"):
        return True
    name = feat_dir.name
    namel = name.lower()
    if lev == "1st":
        if "analysis-2ndglm" in namel or "analysis-3rdglm" in namel:
            return False
        if not _RUN_IN_NAME.search(name):
            return False
        return (
            "analysis-1st" in namel
            or "1stglm" in namel
            or "1stroi" in namel
        )
    if lev == "2nd":
        return "analysis-2ndglm" in namel
    if lev == "3rd":
        if "analysis-3rdglm" in namel:
            return True
        if name.endswith(".gfeat") and _has_group_output_ancestor(feat_dir):
            return True
        if _ACTIVATION_GROUP_GFEAT.match(name):
            return True
        return False
    return True


def collect_legacy_activation_group_feats(
    fslfeat_root: Path,
    level: str,
) -> list[Path]:
    """
    Old layout: ``fslFeat/group/ses-V2/activation/*.gfeat`` (not under ``fslFeat/activation/``).
    """
    lev = (level or "all").strip().lower()
    if lev not in ("3rd", "all", "both", ""):
        return []
    legacy = fslfeat_root / "group"
    if not legacy.is_dir():
        return []
    out: list[Path] = []
    for fd in iter_feat_under(legacy):
        if "activation" not in fd.parts:
            continue
        if lev in ("all", "both", "") or feat_matches_level(fd, "3rd"):
            out.append(fd)
    return out


def infer_code_root(fslfeat_root: Path) -> Path:
    """Default repo root: sibling ``fMRI_exp2`` next to ``fslFeat`` under ``.../``."""
    return fslfeat_root.resolve().parent / "fMRI_exp2"


def extra_deriv_dirs_for_analysis(feat_analysis: str, level: str) -> tuple[str, ...]:
    """Directory names under  (e.g. gppi_timecourse)."""
    entry = EXTRA_DERIV_DIRS_BY_ANALYSIS_LEVEL.get(feat_analysis)
    if not entry:
        return ()
    lev = (level or "all").strip().lower()
    if lev in ("all", "both", ""):
        order: list[str] = []
        seen: set[str] = set()
        for key in ("1st", "2nd", "3rd"):
            for name in entry.get(key, ()):
                if name not in seen:
                    seen.add(name)
                    order.append(name)
        return tuple(order)
    return tuple(entry.get(lev, ()))


def job_subdirs_for_fslfeat_analysis(feat_analysis: str, level: str) -> tuple[str, ...]:
    """
    ``job_files/...`` paths relative to the code module folder for this fslFeat
    top-level name (see CODE_SUBDIR_BY_FSLFEAT_ANALYSIS).
    """
    entry = JOB_SUBDIRS_BY_ANALYSIS_LEVEL.get(feat_analysis)
    if not entry:
        return ()
    lev = (level or "all").strip().lower()
    if lev in ("all", "both", ""):
        order: list[str] = []
        seen: set[str] = set()
        if feat_analysis == "glm":
            for rel in ("job_files/check_fslfeat",):
                if rel not in seen:
                    seen.add(rel)
                    order.append(rel)
        for key in ("1st", "2nd", "3rd"):
            for rel in entry.get(key, ()):
                if rel not in seen:
                    seen.add(rel)
                    order.append(rel)
        return tuple(order)
    return tuple(entry.get(lev, ()))


def collect_job_file_targets(
    code_root: Path,
    fslfeat_analyses: list[str],
    level: str,
) -> list[Path]:
    """Absolute paths to job_files subtree roots to remove (deduped, deepest first)."""
    targets: list[Path] = []
    seen_resolved: set[Path] = set()
    for analysis in fslfeat_analyses:
        subdir = CODE_SUBDIR_BY_FSLFEAT_ANALYSIS.get(analysis)
        if not subdir:
            continue
        rels = job_subdirs_for_fslfeat_analysis(analysis, level)
        if not rels:
            continue
        base = (code_root / subdir).resolve()
        for rel in rels:
            p = (base / rel).resolve()
            if p in seen_resolved:
                continue
            seen_resolved.add(p)
            if p.is_dir():
                targets.append(p)
    targets.sort(key=lambda x: len(x.parts), reverse=True)
    return targets


def level_matches_gppi_legacy_timecourses(level: str) -> bool:
    lev = (level or "all").strip().lower()
    return lev in ("all", "both", "", "1st")


def level_matches_evc_pipeline_artifacts(level: str) -> bool:
    """EVC has only 1st-level FEAT; artifact cleanup applies for 1st or all."""
    lev = (level or "all").strip().lower()
    return lev in ("all", "both", "", "1st")


def collect_evc_artifact_targets(
    bids_root: Path,
    code_root: Path,
    level: str,
) -> list[Path]:
    """
    EVCLoc inputs/outputs outside fslFeat and evc_rois (steps 01–02, 07 report).
    """
    if not level_matches_evc_pipeline_artifacts(level):
        return []

    targets: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        rp = path.resolve()
        if rp in seen or not path.exists():
            return
        seen.add(rp)
        targets.append(rp)

    checks = bids_root / "derivatives/logfilechecks"
    if checks.is_dir():
        for pattern in (
            "sub-*_ses-V2-EVCLoc_errorFlags.csv",
            "sub-*_ses-V2-EVCLoc_regressor_status.csv",
        ):
            for f in checks.glob(pattern):
                if f.is_file():
                    add(f)

    reg_root = bids_root / "derivatives/regressoreventfiles"
    if reg_root.is_dir():
        for sub_dir in reg_root.glob("sub-*"):
            ses_v2 = sub_dir / "ses-V2"
            if not ses_v2.is_dir():
                continue
            for f in ses_v2.rglob("*EVCLoc*"):
                if f.is_file():
                    add(f)
            conf_dir = ses_v2 / "confound_event_files"
            if conf_dir.is_dir():
                for f in conf_dir.glob("*EVCLoc*"):
                    if f.is_file():
                        add(f)

    for events_tsv in bids_root.glob("sub-*/ses-V2/func/*EVCLoc*events.tsv"):
        if events_tsv.is_file():
            add(events_tsv)
    for events_json in bids_root.glob("sub-*/ses-V2/func/*EVCLoc*events.json"):
        if events_json.is_file():
            add(events_json)

    inc_report = code_root / "EVC" / "inclusion report"
    if inc_report.is_dir():
        add(inc_report)

    targets.sort(key=lambda p: len(p.parts), reverse=True)
    return targets


def collect_extra_derivative_targets(
    deriv_root: Path,
    fslfeat_analyses: list[str],
    level: str,
) -> list[Path]:
    """Absolute paths to exclude/new derivative trees (and legacy files) to remove."""
    targets: list[Path] = []
    seen: set[Path] = set()
    for analysis in fslfeat_analyses:
        for name in extra_deriv_dirs_for_analysis(analysis, level):
            p = (deriv_root / name).resolve()
            if p not in seen and p.is_dir():
                seen.add(p)
                targets.append(p)
        if analysis == "gPPI" and level_matches_gppi_legacy_timecourses(level):
            seeds = deriv_root / "gppi_seeds"
            if seeds.is_dir():
                for f in sorted(seeds.rglob("*_timecourse_*.txt")):
                    if f.is_file():
                        rp = f.resolve()
                        if rp not in seen:
                            seen.add(rp)
                            targets.append(rp)
    targets.sort(key=lambda x: len(x.parts), reverse=True)
    return targets


def discover_analysis_dirs(fslfeat_root: Path) -> list[str]:
    """Top-level directory names under fslFeat (excludes BIDS subject folders sub-*)."""
    if not fslfeat_root.is_dir():
        return []
    names: list[str] = []
    for p in fslfeat_root.iterdir():
        if p.is_dir() and not p.name.startswith("sub-"):
            names.append(p.name)
    return sorted(names)


def parse_analyses_arg(raw: str) -> list[str]:
    parts: list[str] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def iter_feat_under(root: Path) -> list[Path]:
    """All directories under root whose names end with .feat or .gfeat."""
    if not root.is_dir():
        return []
    out: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root, followlinks=False):
        for name in dirnames:
            if name.endswith(".feat") or name.endswith(".gfeat"):
                out.append(Path(dirpath) / name)
        dirnames[:] = [
            d
            for d in dirnames
            if d.endswith(".gfeat") or not d.endswith(".feat")
        ]
    out.sort(key=lambda p: str(p))
    return out


def delete_deepest_first(paths: list[Path], *, execute: bool) -> int:
    """Remove files and directories; deepest paths first."""
    unique = sorted({p.resolve() for p in paths}, key=lambda p: len(p.parts), reverse=True)
    errors = 0
    for path in unique:
        if not path.exists():
            continue
        if execute:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                print(f"removed: {path}")
            except OSError as exc:
                print(f"ERROR removing {path}: {exc}", file=sys.stderr)
                errors += 1
        else:
            print(f"would remove: {path}")
    return errors


def resolve_analyses(
    fslfeat_root: Path,
    analysis_args: list[str],
) -> tuple[list[str], str | None]:
    """
    Return (analysis_dir_names, error_message).
    ``analysis_args`` entries are lowercased only for the special token ``all``.
    """
    discovered = discover_analysis_dirs(fslfeat_root)
    if not discovered and fslfeat_root.is_dir():
        pass  # empty fslFeat tree
    flat: list[str] = []
    for a in analysis_args:
        flat.extend(parse_analyses_arg(a))

    if not flat:
        return [], "no --analysis values (use e.g. --analysis activation or --analysis all)"

    has_all = any(a.lower() == "all" for a in flat)
    if has_all:
        if len(flat) > 1:
            return (
                [],
                "do not combine 'all' with other --analysis values",
            )
        return list(discovered), None

    unknown = [a for a in flat if a not in discovered]
    if unknown:
        return (
            [],
            f"unknown --analysis name(s): {unknown!r}. "
            f"Known under {fslfeat_root}: {discovered!r}",
        )
    return flat, None


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Delete FEAT (.feat/.gfeat) outputs under fslFeat, by analysis folder and level."
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
            "Top-level folder under fslFeat (e.g. activation, gPPI, FIR, EVC, glm, group). "
            "Repeat or use commas. Use 'all' for every non-sub-* folder. "
            "See also --list-analyses."
        ),
    )
    p.add_argument(
        "--level",
        choices=("all", "1st", "2nd", "3rd"),
        default=os.environ.get("FEAT_CHECK_LEVEL", "all"),
        help=(
            "Restrict deletions to FEAT dirs: "
            "1st = run-level first-level; 2nd = analysis-2ndGLM in basename; "
            "3rd = analysis-3rdGLM and/or group-level outputs under */group/ "
            "(e.g. activation/group/seen_vs_unseen_*.gfeat); "
            "all = no level filter (default). "
            "Default may follow env FEAT_CHECK_LEVEL (all|1st|2nd)."
        ),
    )
    p.add_argument(
        "--list-analyses",
        action="store_true",
        help="Print analysis folder names under --fslfeat-root and exit",
    )
    p.add_argument(
        "--code-root",
        type=Path,
        default=None,
        help=(
            "fMRI_exp2 repository root (parent of activation/, gPPI_code/, …). "
            "Default: parent of --fslfeat-root plus /fMRI_exp2"
        ),
    )
    p.add_argument(
        "--no-job-files",
        action="store_true",
        help="Do not remove job_files/ submission trees (only delete FEAT outputs)",
    )
    p.add_argument(
        "--no-extra-derivatives",
        action="store_true",
        help=(
            "Do not remove non-FEAT derivative trees (gppi_timecourse, evc_rois), "
            "EVCLoc regressors/logfilechecks/events, or legacy gppi_seeds timecourses"
        ),
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete (default is dry-run)",
    )
    args = p.parse_args()

    root = args.fslfeat_root.resolve()

    if args.list_analyses:
        names = discover_analysis_dirs(root)
        print(f"fslFeat root: {root}")
        if not root.is_dir():
            print("(root missing or not a directory)", file=sys.stderr)
            return 1
        if not names:
            print("No analysis directories found (only sub-* or empty).")
        else:
            print("Analysis directories:")
            for n in names:
                print(f"  {n}")
        return 0

    analyses, err = resolve_analyses(root, args.analysis)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    if not root.is_dir():
        print(f"ERROR: fslFeat root is not a directory: {root}", file=sys.stderr)
        return 1

    targets: list[Path] = []
    for an in analyses:
        sub = root / an
        if not sub.is_dir():
            print(f"WARN: skip missing analysis path: {sub}", file=sys.stderr)
            continue
        for fd in iter_feat_under(sub):
            if feat_matches_level(fd, args.level):
                targets.append(fd)

    lev = (args.level or "all").strip().lower()
    if (
        lev in ("3rd", "all", "both", "")
        and "activation" in analyses
        and "group" not in analyses
    ):
        legacy = collect_legacy_activation_group_feats(root, args.level)
        if legacy:
            print(
                f"Also matching legacy activation group outputs under {root / 'group'}: "
                f"{len(legacy)}",
                file=sys.stderr,
            )
            targets.extend(legacy)

    print(f"fslFeat root: {root}")
    print(f"Analyses: {', '.join(analyses)}")
    print(f"Level filter: {args.level}")
    print(f"Mode: {'DELETE' if args.execute else 'DRY-RUN (pass --execute to delete)'}")
    print(f"Matching .feat/.gfeat directories: {len(targets)}")
    print()

    errors = delete_deepest_first(targets, execute=args.execute)

    if not args.no_job_files:
        code_root = (args.code_root or infer_code_root(root)).resolve()
        print(f"Code root (job_files): {code_root}")
        unmapped = [a for a in analyses if a not in CODE_SUBDIR_BY_FSLFEAT_ANALYSIS]
        for a in unmapped:
            print(
                f"WARN: no job_files mapping for fslFeat analysis {a!r} (skipping job_files)",
                file=sys.stderr,
            )
        if not code_root.is_dir():
            print(
                f"WARN: code root is not a directory; skipping job_files cleanup: {code_root}",
                file=sys.stderr,
            )
        else:
            job_targets = collect_job_file_targets(code_root, analyses, args.level)
            print(f"Matching job_files roots to remove: {len(job_targets)}")
            print()
            errors += delete_deepest_first(job_targets, execute=args.execute)
    else:
        print("Skipping job_files (--no-job-files).")
        print()

    if not args.no_extra_derivatives:
        deriv_root = root.parent
        # fslFeat -> .../fslFeat; bids root is three levels above deriv_root
        bids_root = deriv_root.parent.parent.parent
        code_root = (args.code_root or infer_code_root(root)).resolve()
        extra_targets = collect_extra_derivative_targets(deriv_root, analyses, args.level)
        if "EVC" in analyses:
            evc_extra = collect_evc_artifact_targets(bids_root, code_root, args.level)
            seen_extra = {p.resolve() for p in extra_targets}
            for p in evc_extra:
                rp = p.resolve()
                if rp not in seen_extra:
                    seen_extra.add(rp)
                    extra_targets.append(p)
            extra_targets.sort(key=lambda p: len(p.parts), reverse=True)
        print(f"Derivatives root (extra): {deriv_root}")
        if "EVC" in analyses:
            print(f"BIDS root (EVC artifacts): {bids_root}")
        print(f"Matching extra derivative paths to remove: {len(extra_targets)}")
        print()
        errors += delete_deepest_first(extra_targets, execute=args.execute)
    else:
        print("Skipping extra derivatives (--no-extra-derivatives).")
        print()

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
