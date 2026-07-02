"""
Patch a higher-level FEAT .fsf after changing how many lower-level .feat inputs are used.

FEAT expects consistency between:
  - set fmri(npts)
  - set fmri(multiple)
  - set feat_files(1..N)
  - set fmri(evgI.J) for inputs I in 1..N
  - set fmri(groupmem.I) for inputs I in 1..N

Templates are generated for MAX_RUNS (e.g. 8); when fewer runs are included, strip
extra evg/groupmem lines and set multiple/npts to N.
"""

from __future__ import annotations

import re
from pathlib import Path


def apply_run_count_patch(
    fsf_text: str,
    n_inputs: int,
    feat_paths_in_order: list[str],
    *,
    insert_marker: str = "# Add confound EVs",
) -> str:
    if n_inputs < 1:
        raise ValueError("n_inputs must be >= 1")
    if len(feat_paths_in_order) != n_inputs:
        raise ValueError(
            f"feat_paths_in_order length ({len(feat_paths_in_order)}) "
            f"must equal n_inputs ({n_inputs})"
        )

    text = fsf_text
    text = re.sub(
        r"^set fmri\(npts\) \d+.*$",
        f"set fmri(npts) {n_inputs}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^set fmri\(multiple\) \d+.*$",
        f"set fmri(multiple) {n_inputs}",
        text,
        count=1,
        flags=re.MULTILINE,
    )

    kept: list[str] = []
    for line in text.splitlines(keepends=True):
        if re.match(r"^set feat_files\(\d+\)", line):
            continue
        m = re.match(r"^set fmri\(evg(\d+)\.", line)
        if m and int(m.group(1)) > n_inputs:
            continue
        m = re.match(r"^set fmri\(groupmem\.(\d+)\)", line)
        if m and int(m.group(1)) > n_inputs:
            continue
        kept.append(line)
    text = "".join(kept)

    feat_block = ""
    for idx, path in enumerate(feat_paths_in_order, start=1):
        feat_block += f'set feat_files({idx}) "{path}"\n\n'

    if insert_marker not in text:
        raise ValueError(f"cannot find insertion marker {insert_marker!r} in FSF text")
    text = text.replace(insert_marker, feat_block + insert_marker, 1)
    return text


def main() -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(
        description="Patch higher-level FEAT FSF for N lower-level inputs (CLI test / offline use)."
    )
    p.add_argument("base_fsf", type=Path)
    p.add_argument("out_fsf", type=Path)
    p.add_argument("n_inputs", type=int)
    p.add_argument("feat_paths", nargs="+", help="Exactly n_inputs paths")
    args = p.parse_args()
    if len(args.feat_paths) != args.n_inputs:
        print("ERROR: number of feat_paths must equal n_inputs", file=sys.stderr)
        return 1
    text = args.base_fsf.read_text(encoding="utf-8")
    out = apply_run_count_patch(text, args.n_inputs, list(args.feat_paths))
    args.out_fsf.parent.mkdir(parents=True, exist_ok=True)
    args.out_fsf.write_text(out, encoding="utf-8")
    print(f"wrote {args.out_fsf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
