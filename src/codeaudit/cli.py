"""
codeaudit.py — lightweight, dependency-free code review coverage tracker.

Purpose
-------
Track "has a human actually read this line?" across a whole codebase, so you
can honestly claim you reviewed 100% of an AI-generated project rather than
skimming a few files. Inspired by (but much simpler than) auditview.

State lives in a single JSON file, .codeaudit.json, at the repo root. Nothing
is sent anywhere. Safe to .gitignore or commit, your choice.

Core workflow
-------------
  python3 codeaudit.py init                 # scan repo, seed manifest
  python3 codeaudit.py status               # overall + per-file coverage
  python3 codeaudit.py next                 # suggest the biggest unreviewed file
  python3 codeaudit.py show path/to/file.py # print file with review markers
  python3 codeaudit.py mark path/to/file.py 10 40   # mark lines 10-40 reviewed
  python3 codeaudit.py unmark path/to/file.py 10 40
  python3 codeaudit.py note path/to/file.py 22 "double-check this SQL"

Excluding human-written code from the audit
--------------------------------------------
  python3 codeaudit.py exclude-file path/to/hand_written.py
  python3 codeaudit.py include-file path/to/hand_written.py   # undo

  python3 codeaudit.py exclude path/to/file.py 40 60          # manual line range
  python3 codeaudit.py include path/to/file.py 40 60          # undo

  python3 codeaudit.py exclude-block path/to/file.py 47       # Python-only heuristic:
                                                                # finds the enclosing
                                                                # def/class around line 47
                                                                # and excludes the whole
                                                                # block. For other
                                                                # languages, use `exclude`
                                                                # with a manual range.

Excluded lines never count toward the audit denominator, so coverage reflects
"percent of code that actually needs review, that has been reviewed."

Stepwise interactive review
----------------------------
  python3 codeaudit.py review                     # walk every file, chunk by chunk
  python3 codeaudit.py review --file app.py        # just one file
  python3 codeaudit.py review --chunk-size 15      # lines per chunk (default 20)

  At each chunk:
    <Enter> or r        mark the whole chunk reviewed, move on
    s                   skip, leave as-is, move on
    x                   mark the whole chunk excluded (human-written), move on
    r <start> <end>     mark a sub-range within the chunk reviewed
    x <start> <end>     mark a sub-range within the chunk excluded
    n <line> <text>     attach a note to a line, stay on this chunk
    q                   save and quit

README badge
------------
  python3 codeaudit.py badge                          # print a markdown badge line
  python3 codeaudit.py badge --update-readme README.md  # insert/update it in place

Safety model
------------
Every command re-scans the repo first and reconciles marks (reviewed AND
excluded) against current file content, content-hash based, not just
line-number based. If a marked line's content changed and no identical
line is found nearby, the mark is DROPPED — coverage can only go down on
an edit, never falsely stay up, and exclusions never silently survive a
change to code they were meant to exempt.
"""

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from urllib.parse import quote

MANIFEST_NAME = ".codeaudit.json"

DEFAULT_IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".idea", ".vscode",
    "target", ".tox", "site-packages",
}

DEFAULT_IGNORE_GLOBS = [
    "*.min.js", "*.min.css", "*.lock", "*.png", "*.jpg", "*.jpeg", "*.gif",
    "*.svg", "*.ico", "*.pdf", "*.zip", "*.tar", "*.gz", "*.whl", "*.so",
    "*.pyc", "*.woff*", "*.ttf", "*.eot",
]

DEFAULT_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".sh", ".bash", ".rb", ".go",
    ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".rs", ".php", ".pl",
    ".sql", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".html", ".css",
    ".md", ".r", ".R",
}

LINE_COMMENT_PREFIXES = {
    ".py": "#", ".sh": "#", ".bash": "#", ".rb": "#", ".pl": "#",
    ".yaml": "#", ".yml": "#", ".toml": "#", ".ini": "#", ".cfg": "#",
    ".r": "#", ".R": "#",
    ".js": "//", ".jsx": "//", ".ts": "//", ".tsx": "//", ".go": "//",
    ".java": "//", ".c": "//", ".h": "//", ".cpp": "//", ".hpp": "//",
    ".cs": "//", ".rs": "//", ".php": "//",
}

PY_HEADER_RE = re.compile(r"^(\s*)(def |class |async def )")

BADGE_START = "<!-- codeaudit:badge:start -->"
BADGE_END = "<!-- codeaudit:badge:end -->"


# ---------------------------------------------------------------- hashing --

def line_hash(text):
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def is_countable(line, ext):
    stripped = line.strip()
    if not stripped:
        return False
    prefix = LINE_COMMENT_PREFIXES.get(ext)
    if prefix and stripped.startswith(prefix):
        return False
    return True


# ------------------------------------------------------------- filesystem --

def should_ignore_dir(dirname):
    return dirname in DEFAULT_IGNORE_DIRS or dirname.startswith(".")


def should_ignore_file(relpath):
    base = os.path.basename(relpath)
    return any(fnmatch.fnmatch(base, pat) for pat in DEFAULT_IGNORE_GLOBS)


def find_root(explicit):
    if explicit:
        return os.path.abspath(explicit)
    cur = os.getcwd()
    while True:
        if os.path.isfile(os.path.join(cur, MANIFEST_NAME)):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.getcwd()
        cur = parent


def manifest_path(root):
    return os.path.join(root, MANIFEST_NAME)


def load_manifest(root):
    path = manifest_path(root)
    if not os.path.isfile(path):
        return {"root": root, "files": {}}
    with open(path) as f:
        return json.load(f)


def prepare_for_save(manifest):
    """Return a save-ready copy with in-memory-only cache fields stripped,
    without mutating the live manifest (important: `review` keeps working
    on other files' caches after an intermediate autosave)."""
    out = {"root": manifest.get("root"), "files": {}}
    for rel, entry in manifest.get("files", {}).items():
        out["files"][rel] = {k: v for k, v in entry.items() if not k.startswith("_")}
    out["summary"] = _overall_stats(manifest)
    return out


def save_manifest(root, manifest):
    with open(manifest_path(root), "w") as f:
        json.dump(prepare_for_save(manifest), f, indent=2, sort_keys=True)


def read_lines(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().split("\n")
    except (OSError, UnicodeDecodeError):
        return None


def walk_source_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not should_ignore_dir(d)]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            if should_ignore_file(rel):
                continue
            ext = os.path.splitext(fn)[1]
            if ext not in DEFAULT_EXTENSIONS:
                continue
            yield rel, full, ext


# --------------------------------------------------------- reconciliation --

def _reconcile_mark_dict(old_marks, hashes):
    """Shared logic for migrating/dropping a {line_no_str: hash} dict
    (used for both 'reviewed' and 'excluded') against current file hashes.
    Returns (new_marks, dropped_count)."""
    new_marks = {}
    dropped = 0
    for lno_str, old_h in old_marks.items():
        lno = int(lno_str)
        idx = lno - 1
        if 0 <= idx < len(hashes) and hashes[idx] == old_h:
            new_marks[lno_str] = old_h
            continue
        found = None
        for delta in range(1, 21):
            for cand in (idx - delta, idx + delta):
                if 0 <= cand < len(hashes) and hashes[cand] == old_h:
                    found = cand + 1
                    break
            if found:
                break
        if found:
            new_marks[str(found)] = old_h
        else:
            dropped += 1
    return new_marks, dropped


def whole_file_hash(hashes):
    return line_hash("\n".join(hashes))


def reconcile(root, manifest, quiet=False):
    """Re-scan the repo; migrate or drop review/exclude marks based on content."""
    files = manifest.setdefault("files", {})
    seen = set()
    dropped_reviewed = 0
    dropped_excluded = 0
    reverted_file_excl = 0

    for rel, full, ext in walk_source_files(root):
        seen.add(rel)
        lines = read_lines(full)
        if lines is None:
            continue
        hashes = [line_hash(l) for l in lines]
        countable = [is_countable(l, ext) for l in lines]

        entry = files.setdefault(rel, {"reviewed": {}, "excluded": {}, "notes": {}})
        entry.setdefault("reviewed", {})
        entry.setdefault("excluded", {})
        entry.setdefault("notes", {})

        new_reviewed, d1 = _reconcile_mark_dict(entry["reviewed"], hashes)
        new_excluded, d2 = _reconcile_mark_dict(entry["excluded"], hashes)
        dropped_reviewed += d1
        dropped_excluded += d2

        # whole-file exclusion only survives if the file is byte-for-byte
        # (line-hash-for-line-hash) identical to when it was excluded
        wfh = whole_file_hash(hashes)
        if entry.get("excluded_file"):
            if entry.get("excluded_file_hash") != wfh:
                entry["excluded_file"] = False
                entry.pop("excluded_file_hash", None)
                reverted_file_excl += 1

        new_notes = {}
        for lno_str, note in entry.get("notes", {}).items():
            idx = int(lno_str) - 1
            if 0 <= idx < len(hashes):
                new_notes[lno_str] = note

        entry["reviewed"] = new_reviewed
        entry["excluded"] = new_excluded
        entry["notes"] = new_notes
        entry["countable_total"] = sum(countable)
        entry["line_count"] = len(lines)
        entry["_hashes"] = hashes
        entry["_countable"] = countable
        entry["_excluded_lines"] = [
            True if entry.get("excluded_file") else (str(i + 1) in new_excluded)
            for i in range(len(lines))
        ]

    for rel in list(files.keys()):
        if rel not in seen:
            del files[rel]

    if not quiet:
        if dropped_reviewed:
            print(f"[reconcile] {dropped_reviewed} stale review mark(s) dropped "
                  f"(content changed).", file=sys.stderr)
        if dropped_excluded:
            print(f"[reconcile] {dropped_excluded} stale exclusion mark(s) dropped "
                  f"(content changed, needs audit again).", file=sys.stderr)
        if reverted_file_excl:
            print(f"[reconcile] {reverted_file_excl} whole-file exclusion(s) reverted "
                  f"(file content changed since it was marked human-written).",
                  file=sys.stderr)

    return manifest


# ----------------------------------------------------------------- stats --

def _audit_stats(entry):
    """(audit_total, audit_reviewed, excluded_count) — countable lines that
    are NOT excluded (denominator), how many of those are reviewed
    (numerator), and how many countable lines were excluded (for display)."""
    countable = entry.get("_countable")
    excluded_lines = entry.get("_excluded_lines")
    reviewed = entry.get("reviewed", {})
    if countable is None or excluded_lines is None:
        total = entry.get("countable_total", 0)
        return total, len(reviewed), 0

    audit_total = 0
    excluded_count = 0
    for i, c in enumerate(countable):
        if not c:
            continue
        if excluded_lines[i]:
            excluded_count += 1
        else:
            audit_total += 1

    audit_reviewed = 0
    for lno_str in reviewed:
        idx = int(lno_str) - 1
        if 0 <= idx < len(countable) and countable[idx] and not excluded_lines[idx]:
            audit_reviewed += 1

    return audit_total, audit_reviewed, excluded_count


def _coverage(entry):
    total, reviewed, _ = _audit_stats(entry)
    return 1.0 if total == 0 else reviewed / total


def _overall_stats(manifest):
    """Aggregate _audit_stats() across every tracked file. Used to stamp a
    top-level "summary" object into the saved manifest, so external tools
    (e.g. a shields.io dynamic badge) can read overall coverage straight out
    of .codeaudit.json without re-deriving it."""
    total = reviewed = excluded = 0
    for entry in manifest.get("files", {}).values():
        a, r, e = _audit_stats(entry)
        total += a
        reviewed += r
        excluded += e
    pct = 100.0 if total == 0 else (reviewed / total) * 100
    return {
        "coverage_pct": round(pct, 2),
        "reviewed": reviewed,
        "total": total,
        "excluded": excluded,
    }


# ------------------------------------------------------------- commands --

def cmd_init(args):
    root = os.path.abspath(args.path or ".")
    manifest = load_manifest(root)
    manifest["root"] = root
    reconcile(root, manifest)
    save_manifest(root, manifest)
    print(f"Initialized/updated {MANIFEST_NAME} at {root} ({len(manifest['files'])} files tracked).")


def cmd_status(args):
    root = find_root(args.db)
    manifest = load_manifest(root)
    reconcile(root, manifest, quiet=not args.verbose)

    files = manifest["files"]
    total_audit = total_reviewed = total_excluded = 0
    for entry in files.values():
        a, r, e = _audit_stats(entry)
        total_audit += a
        total_reviewed += r
        total_excluded += e
    overall = (total_reviewed / total_audit) if total_audit else 1.0

    print(f"Overall coverage: {overall*100:.1f}%  "
          f"({total_reviewed}/{total_audit} auditable lines, "
          f"{total_excluded} excluded as human-written, {len(files)} files)")

    if args.files:
        print()
        rows = []
        for rel, entry in files.items():
            a, r, e = _audit_stats(entry)
            cov = 1.0 if a == 0 else r / a
            rows.append((cov, rel, r, a, e))
        rows.sort()
        for cov, rel, r, a, e in rows:
            marker = "DONE" if a == 0 or cov >= 0.999 else f"{cov*100:5.1f}%"
            excl_note = f"  (excl {e})" if e else ""
            print(f"  {marker:>6}  {r:>5}/{a:<5}  {rel}{excl_note}")

    save_manifest(root, manifest)


def cmd_next(args):
    root = find_root(args.db)
    manifest = load_manifest(root)
    reconcile(root, manifest, quiet=True)

    candidates = []
    for rel, entry in manifest["files"].items():
        a, r, _ = _audit_stats(entry)
        if a > 0 and r < a:
            candidates.append((a - r, rel, r / a))

    save_manifest(root, manifest)

    if not candidates:
        print("Everything auditable has been reviewed. Nice.")
        return
    candidates.sort(reverse=True)
    remaining, rel, cov = candidates[0]
    print(f"{rel}  ({cov*100:.1f}% reviewed, {remaining} lines left)")


def cmd_show(args):
    root = find_root(args.db)
    manifest = load_manifest(root)
    reconcile(root, manifest, quiet=True)

    rel = os.path.relpath(os.path.abspath(args.file), root)
    entry = manifest["files"].get(rel)
    if entry is None:
        print(f"{rel} is not tracked (not a recognized source file, or ignored).")
        return

    lines = read_lines(os.path.join(root, rel)) or []
    reviewed = entry.get("reviewed", {})
    notes = entry.get("notes", {})
    excluded_lines = entry.get("_excluded_lines", [])
    countable = entry.get("_countable", [])
    for i, text in enumerate(lines):
        lno = i + 1
        if i < len(excluded_lines) and excluded_lines[i]:
            mark = "E"
        elif str(lno) in reviewed:
            mark = "R"
        elif i < len(countable) and countable[i]:
            mark = " "
        else:
            mark = "."
        note = f"   # {notes[str(lno)]}" if str(lno) in notes else ""
        print(f"{lno:5} [{mark}] {text}{note}")

    save_manifest(root, manifest)


def _resolve_range(start, end):
    if end is None:
        end = start
    return (start, end) if start <= end else (end, start)


def _apply_mark(entry, hashes, start, end, key):
    changed = 0
    for lno in range(start, end + 1):
        idx = lno - 1
        if 0 <= idx < len(hashes):
            entry.setdefault(key, {})[str(lno)] = hashes[idx]
            changed += 1
    return changed


def _apply_unmark(entry, start, end, key):
    changed = 0
    for lno in range(start, end + 1):
        if entry.setdefault(key, {}).pop(str(lno), None) is not None:
            changed += 1
    return changed


def _get_tracked_entry(root, manifest, filearg):
    rel = os.path.relpath(os.path.abspath(filearg), root)
    entry = manifest["files"].get(rel)
    if entry is None:
        print(f"{rel} is not tracked.", file=sys.stderr)
        sys.exit(1)
    return rel, entry


def cmd_mark(args, mark=True):
    root = find_root(args.db)
    manifest = load_manifest(root)
    reconcile(root, manifest, quiet=True)
    rel, entry = _get_tracked_entry(root, manifest, args.file)
    start, end = _resolve_range(args.start, args.end)
    hashes = entry["_hashes"]
    n = _apply_mark(entry, hashes, start, end, "reviewed") if mark \
        else _apply_unmark(entry, start, end, "reviewed")
    save_manifest(root, manifest)
    print(f"{'Marked' if mark else 'Unmarked'} {rel}:{start}-{end} ({n} lines).")


def cmd_exclude_range(args, exclude=True):
    root = find_root(args.db)
    manifest = load_manifest(root)
    reconcile(root, manifest, quiet=True)
    rel, entry = _get_tracked_entry(root, manifest, args.file)
    start, end = _resolve_range(args.start, args.end)
    hashes = entry["_hashes"]
    n = _apply_mark(entry, hashes, start, end, "excluded") if exclude \
        else _apply_unmark(entry, start, end, "excluded")
    save_manifest(root, manifest)
    verb = "Excluded" if exclude else "Un-excluded"
    print(f"{verb} {rel}:{start}-{end} ({n} lines) — {'now exempt from audit' if exclude else 'now needs audit again'}.")


def cmd_exclude_file(args, exclude=True):
    root = find_root(args.db)
    manifest = load_manifest(root)
    reconcile(root, manifest, quiet=True)
    rel, entry = _get_tracked_entry(root, manifest, args.file)
    if exclude:
        entry["excluded_file"] = True
        entry["excluded_file_hash"] = whole_file_hash(entry["_hashes"])
        print(f"Excluded whole file: {rel} (marked human-written, exempt from audit).")
    else:
        entry["excluded_file"] = False
        entry.pop("excluded_file_hash", None)
        print(f"Un-excluded: {rel} (needs audit again).")
    save_manifest(root, manifest)


def _find_python_block(lines, target_line):
    """Best-effort: find the (start, end) 1-indexed line range of the
    innermost enclosing def/class around target_line, using indentation.
    Returns None if target_line is at module level (no enclosing block)."""
    idx = target_line - 1
    if idx < 0 or idx >= len(lines):
        return None

    def indent_of(s):
        return len(s) - len(s.lstrip(" \t"))

    # if the target line itself is a header, treat it as its own block
    start_idx = None
    cur_min = None
    for i in range(idx, -1, -1):
        s = lines[i]
        if not s.strip():
            continue
        ind = indent_of(s)
        if cur_min is None:
            cur_min = ind if i != idx else ind + 1  # ensure header search looks upward properly
            if i == idx and PY_HEADER_RE.match(s):
                start_idx = i
                break
            cur_min = ind
            continue
        if ind < cur_min:
            cur_min = ind
            if PY_HEADER_RE.match(s):
                start_idx = i
                break

    if start_idx is None:
        return None

    header_indent = indent_of(lines[start_idx])
    end_idx = start_idx
    for i in range(start_idx + 1, len(lines)):
        s = lines[i]
        if not s.strip():
            end_idx = i
            continue
        if indent_of(s) > header_indent:
            end_idx = i
        else:
            break
    return start_idx + 1, end_idx + 1


def cmd_exclude_block(args):
    root = find_root(args.db)
    manifest = load_manifest(root)
    reconcile(root, manifest, quiet=True)
    rel, entry = _get_tracked_entry(root, manifest, args.file)

    if not rel.endswith(".py"):
        print(f"exclude-block only supports Python heuristically right now; "
              f"use `exclude {args.file} <start> <end>` for other languages.",
              file=sys.stderr)
        sys.exit(1)

    lines = read_lines(os.path.join(root, rel)) or []
    block = _find_python_block(lines, args.line)
    if block is None:
        print(f"Couldn't find an enclosing def/class around line {args.line} "
              f"(it may be at module level). Use `exclude` with a manual range instead.",
              file=sys.stderr)
        sys.exit(1)

    start, end = block
    hashes = entry["_hashes"]
    n = _apply_mark(entry, hashes, start, end, "excluded")
    save_manifest(root, manifest)
    print(f"Excluded {rel}:{start}-{end} ({n} lines) — detected enclosing block around line {args.line}.")


def cmd_note(args):
    root = find_root(args.db)
    manifest = load_manifest(root)
    reconcile(root, manifest, quiet=True)
    rel, entry = _get_tracked_entry(root, manifest, args.file)
    entry.setdefault("notes", {})[str(args.line)] = args.text
    save_manifest(root, manifest)
    print(f"Noted {rel}:{args.line} — {args.text}")


# ------------------------------------------------------------------ badge --

def _badge_color(pct):
    if pct >= 99.999:
        return "brightgreen"
    if pct >= 90:
        return "green"
    if pct >= 75:
        return "yellowgreen"
    if pct >= 50:
        return "yellow"
    if pct >= 25:
        return "orange"
    return "red"


_GITHUB_REMOTE_RE = re.compile(
    r"^(?:git@github\.com:|https://github\.com/)(?P<owner>[^/]+)/(?P<repo>.+?)(?:\.git)?$"
)


def _git_raw_base(root, path_in_repo):
    """Best-effort: derive a raw.githubusercontent.com URL for a file in this
    repo, from the 'origin' remote and current branch. Returns None if it
    can't be determined (no git, no remote, not a github.com remote)."""
    try:
        remote = subprocess.run(
            ["git", "-C", root, "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=5,
        )
        branch = subprocess.run(
            ["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if remote.returncode != 0 or branch.returncode != 0:
        return None

    m = _GITHUB_REMOTE_RE.match(remote.stdout.strip())
    ref = branch.stdout.strip()
    if not m or not ref:
        return None
    return (f"https://raw.githubusercontent.com/{m.group('owner')}/"
            f"{m.group('repo')}/{ref}/{path_in_repo}")


def cmd_badge(args):
    root = find_root(args.db)
    manifest = load_manifest(root)
    reconcile(root, manifest, quiet=True)

    total_audit = total_reviewed = 0
    for entry in manifest["files"].values():
        a, r, _ = _audit_stats(entry)
        total_audit += a
        total_reviewed += r
    pct = 100.0 if total_audit == 0 else (total_reviewed / total_audit) * 100
    save_manifest(root, manifest)

    label = args.label.replace(" ", "%20")
    color = _badge_color(pct)

    if args.dynamic:
        raw_url = _git_raw_base(root, MANIFEST_NAME)
        if raw_url is None:
            print("Couldn't determine a GitHub raw URL for this repo (no 'origin' "
                  "remote configured, or it's not a github.com remote). Push this "
                  "repo to GitHub and add an 'origin' remote first, or omit "
                  "--dynamic for a static badge.", file=sys.stderr)
            sys.exit(1)
        badge_line = (
            f"![{args.label}](https://img.shields.io/badge/dynamic/json"
            f"?url={quote(raw_url, safe='')}"
            f"&query={quote('$.summary.coverage_pct', safe='')}"
            f"&suffix=%25&label={label}&color={color})"
        )
    else:
        badge_line = (f"![{args.label}](https://img.shields.io/badge/"
                      f"{label}-{round(pct)}%25-{color})")

    if args.update_readme:
        _update_readme_badge(args.update_readme, badge_line)
        mode = "dynamic (reads .codeaudit.json live from GitHub)" if args.dynamic else "static"
        print(f"Updated {mode} badge in {args.update_readme}: {round(pct)}% ({color}).")
    else:
        print(badge_line)


def _update_readme_badge(path, badge_line):
    block = f"{BADGE_START}\n{badge_line}\n{BADGE_END}"
    if os.path.isfile(path):
        with open(path) as f:
            content = f.read()
    else:
        content = ""

    if BADGE_START in content and BADGE_END in content:
        pattern = re.compile(re.escape(BADGE_START) + r".*?" + re.escape(BADGE_END), re.DOTALL)
        new_content = pattern.sub(block, content, count=1)
    elif content:
        new_content = block + "\n\n" + content
    else:
        new_content = block + "\n"

    with open(path, "w") as f:
        f.write(new_content)


# ----------------------------------------------------------------- review --

_REVIEW_HELP = """\
  <Enter> or r          mark this whole chunk reviewed, move on
  s                     skip, leave as-is, move on
  x                     mark this whole chunk excluded (human-written), move on
  r <start> <end>       mark a sub-range (absolute line numbers) reviewed
  x <start> <end>       mark a sub-range (absolute line numbers) excluded
  n <line> <text>       attach a note to a line, stay on this chunk
  q                     save and quit
  ? or h                show this help
"""


def cmd_review(args):
    root = find_root(args.db)
    manifest = load_manifest(root)
    reconcile(root, manifest, quiet=True)

    if args.file:
        targets = [os.path.relpath(os.path.abspath(args.file), root)]
        if targets[0] not in manifest["files"]:
            print(f"{targets[0]} is not tracked.", file=sys.stderr)
            sys.exit(1)
    else:
        targets = sorted(manifest["files"].keys())

    chunk_size = max(1, args.chunk_size)
    print(f"Stepwise review. {len(targets)} file(s), chunk size {chunk_size}. "
          f"Type ? for help, q to quit.\n")

    quit_all = False
    for rel in targets:
        if quit_all:
            break
        entry = manifest["files"][rel]
        full_path = os.path.join(root, rel)
        hashes = entry["_hashes"]
        countable = entry["_countable"]
        text_lines = read_lines(full_path) or []
        n = len(hashes)
        i = 0
        while i < n:
            end = min(i + chunk_size, n)
            excluded_lines = entry["_excluded_lines"]
            reviewed = entry.get("reviewed", {})
            pending = any(
                countable[j] and not excluded_lines[j] and str(j + 1) not in reviewed
                for j in range(i, end)
            )
            if not pending:
                i = end
                continue

            a, r, _ = _audit_stats(entry)
            file_cov = 100.0 if a == 0 else (r / a) * 100
            print(f"=== {rel}  lines {i+1}-{end}  (file: {file_cov:.1f}% reviewed) ===")
            notes = entry.get("notes", {})
            for j in range(i, end):
                lno = j + 1
                if excluded_lines[j]:
                    mark = "E"
                elif str(lno) in reviewed:
                    mark = "R"
                elif countable[j]:
                    mark = " "
                else:
                    mark = "."
                suffix = f"   # {notes[str(lno)]}" if str(lno) in notes else ""
                print(f"{lno:5} [{mark}] {text_lines[j]}{suffix}")

            while True:
                try:
                    raw = input("> ").strip()
                except EOFError:
                    raw = "q"
                low = raw.lower()

                if raw == "" or low == "r":
                    _apply_mark(entry, hashes, i + 1, end, "reviewed")
                    break
                if low == "s":
                    break
                if low == "x":
                    _apply_mark(entry, hashes, i + 1, end, "excluded")
                    entry["_excluded_lines"] = [
                        True if entry.get("excluded_file") else (str(k + 1) in entry["excluded"])
                        for k in range(n)
                    ]
                    break
                if low.startswith("r ") or low.startswith("x "):
                    parts = raw.split()
                    key = "reviewed" if low.startswith("r ") else "excluded"
                    try:
                        a0 = int(parts[1])
                        b0 = int(parts[2]) if len(parts) > 2 else a0
                    except (IndexError, ValueError):
                        print("usage: r|x <start> [end]")
                        continue
                    s0, e0 = _resolve_range(a0, b0)
                    _apply_mark(entry, hashes, s0, e0, key)
                    if key == "excluded":
                        entry["_excluded_lines"] = [
                            True if entry.get("excluded_file") else (str(k + 1) in entry["excluded"])
                            for k in range(n)
                        ]
                    break
                if low.startswith("n "):
                    parts = raw.split(maxsplit=2)
                    if len(parts) < 3:
                        print("usage: n <line> <text>")
                        continue
                    try:
                        lno = int(parts[1])
                    except ValueError:
                        print("usage: n <line> <text>")
                        continue
                    entry.setdefault("notes", {})[str(lno)] = parts[2]
                    print(f"noted line {lno}")
                    continue
                if low in ("q", "quit"):
                    quit_all = True
                    break
                if low in ("?", "h", "help"):
                    print(_REVIEW_HELP)
                    continue
                print("unrecognized command, ? for help")

            save_manifest(root, manifest)  # autosave after every chunk decision
            if quit_all:
                break
            i = end

    print("\nSession saved.")


# ------------------------------------------------------------------- cli --

def build_parser():
    p = argparse.ArgumentParser(description="Lightweight code review coverage tracker.")
    p.add_argument("--db", metavar="ROOT", help="repo root (default: search upward for .codeaudit.json)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="scan repo and create/update the manifest")
    s.add_argument("path", nargs="?", help="repo path (default: current directory)")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("status", help="show coverage")
    s.add_argument("--files", action="store_true", help="show per-file breakdown")
    s.add_argument("--verbose", action="store_true", help="report dropped/reverted marks during reconcile")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("next", help="suggest the file with the most unreviewed lines")
    s.set_defaults(func=cmd_next)

    s = sub.add_parser("show", help="print a file annotated with review status")
    s.add_argument("file")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("mark", help="mark a line range as reviewed")
    s.add_argument("file"); s.add_argument("start", type=int); s.add_argument("end", type=int, nargs="?")
    s.set_defaults(func=lambda a: cmd_mark(a, mark=True))

    s = sub.add_parser("unmark", help="unmark a line range")
    s.add_argument("file"); s.add_argument("start", type=int); s.add_argument("end", type=int, nargs="?")
    s.set_defaults(func=lambda a: cmd_mark(a, mark=False))

    s = sub.add_parser("exclude", help="mark a line range as human-written (exempt from audit)")
    s.add_argument("file"); s.add_argument("start", type=int); s.add_argument("end", type=int, nargs="?")
    s.set_defaults(func=lambda a: cmd_exclude_range(a, exclude=True))

    s = sub.add_parser("include", help="undo `exclude` for a line range (needs audit again)")
    s.add_argument("file"); s.add_argument("start", type=int); s.add_argument("end", type=int, nargs="?")
    s.set_defaults(func=lambda a: cmd_exclude_range(a, exclude=False))

    s = sub.add_parser("exclude-file", help="mark an entire file as human-written (exempt from audit)")
    s.add_argument("file")
    s.set_defaults(func=lambda a: cmd_exclude_file(a, exclude=True))

    s = sub.add_parser("include-file", help="undo `exclude-file` (file needs audit again)")
    s.add_argument("file")
    s.set_defaults(func=lambda a: cmd_exclude_file(a, exclude=False))

    s = sub.add_parser("exclude-block",
                        help="Python-only heuristic: exclude the enclosing def/class around a line")
    s.add_argument("file"); s.add_argument("line", type=int)
    s.set_defaults(func=cmd_exclude_block)

    s = sub.add_parser("note", help="attach a note to a specific line")
    s.add_argument("file"); s.add_argument("line", type=int); s.add_argument("text")
    s.set_defaults(func=cmd_note)

    s = sub.add_parser("badge", help="print (or write into a README) a coverage badge")
    s.add_argument("--label", default="audit coverage")
    s.add_argument("--update-readme", metavar="README_PATH")
    s.add_argument("--dynamic", action="store_true",
                    help="reference .codeaudit.json live via a shields.io dynamic JSON "
                         "badge instead of baking the percentage into the URL (requires "
                         "a GitHub 'origin' remote; the percentage stays live on every "
                         "page load, the color reflects the value at generation time)")
    s.set_defaults(func=cmd_badge)

    s = sub.add_parser("review", help="stepwise, chunk-by-chunk interactive review")
    s.add_argument("--file", help="restrict to one file (default: whole repo, path order)")
    s.add_argument("--chunk-size", type=int, default=20)
    s.set_defaults(func=cmd_review)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
