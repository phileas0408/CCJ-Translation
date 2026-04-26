"""
XUnity.AutoTranslator hash renamer.

Walks a folder, computes the two hashes XUAT uses for texture filenames
(TextureHashGenerationStrategy=FromImageName) and renames each file to the
"name [NAMEHASH-DATAHASH].ext" format expected by the plugin.

  - NAMEHASH = first 5 bytes of SHA1(UTF-8 name without BOM, no extension)
  - DATAHASH = first 5 bytes of SHA1(file bytes)

Usage:
    python xuat_hash_rename.py <folder>
    python xuat_hash_rename.py <folder> --recursive
    python xuat_hash_rename.py <folder> --dry-run
    python xuat_hash_rename.py <folder> --extensions .png .jpg
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

DEFAULT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".webp"}

# Matches a trailing " [XXXXXXXXXX-XXXXXXXXXX]" so we can strip it before re-hashing.
EXISTING_HASH_RE = re.compile(r"\s*\[[0-9A-Fa-f]{10}-[0-9A-Fa-f]{10}\]$")


def sha1_first5(data: bytes) -> str:
    """Return the first 5 bytes (10 hex chars, uppercase) of SHA1(data)."""
    return hashlib.sha1(data).hexdigest()[:10].upper()


def clean_stem(stem: str) -> str:
    """Strip any existing [XXXXXXXXXX-XXXXXXXXXX] suffix from a filename stem."""
    return EXISTING_HASH_RE.sub("", stem).rstrip()


def build_new_name(path: Path) -> str:
    """Compute the XUAT-style filename for a given file path."""
    base = clean_stem(path.stem)
    name_hash = sha1_first5(base.encode("utf-8"))
    data_hash = sha1_first5(path.read_bytes())
    return f"{base} [{name_hash}-{data_hash}]{path.suffix}"


def iter_files(folder: Path, recursive: bool, extensions: set[str]):
    """Yield files in folder, optionally recursively, filtered by extension."""
    pattern = "**/*" if recursive else "*"
    for p in folder.glob(pattern):
        if p.is_file() and p.suffix.lower() in extensions:
            yield p


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rename files to XUnity.AutoTranslator hash format."
    )
    parser.add_argument("folder", type=Path, help="Folder containing files to rename.")
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Recurse into subfolders."
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without changing anything.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=sorted(DEFAULT_EXTENSIONS),
        help=f"Extensions to process (default: {' '.join(sorted(DEFAULT_EXTENSIONS))}).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every file regardless of extension.",
    )
    args = parser.parse_args()

    folder: Path = args.folder
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory.", file=sys.stderr)
        return 1

    if args.all:
        # Sentinel: accept any suffix.
        extensions = None
    else:
        extensions = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.extensions}

    renamed = skipped = errored = 0

    files = (
        (p for p in folder.rglob("*") if p.is_file())
        if args.recursive and extensions is None
        else (p for p in folder.glob("*") if p.is_file())
        if extensions is None
        else iter_files(folder, args.recursive, extensions)
    )

    for path in files:
        try:
            new_name = build_new_name(path)
        except OSError as e:
            print(f"  ! {path.name}: read failed ({e})", file=sys.stderr)
            errored += 1
            continue

        if new_name == path.name:
            print(f"  = {path.name} (already correct)")
            skipped += 1
            continue

        target = path.with_name(new_name)
        if target.exists() and target != path:
            print(f"  ! {path.name} -> {new_name} (target exists, skipping)", file=sys.stderr)
            errored += 1
            continue

        if args.dry_run:
            print(f"  ~ {path.name} -> {new_name}")
        else:
            path.rename(target)
            print(f"  + {path.name} -> {new_name}")
        renamed += 1

    print()
    print(f"Done. renamed={renamed}  skipped={skipped}  errors={errored}"
          + ("  (dry run)" if args.dry_run else ""))
    return 0 if errored == 0 else 2


if __name__ == "__main__":
    sys.exit(main())