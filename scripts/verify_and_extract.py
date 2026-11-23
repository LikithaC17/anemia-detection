"""
Phase 1: Task 1 & 2 - ZIP Verification and Safe Extraction
Author: Senior Python Developer & Data Science Engineer
Project: AI-Based Anemia Detection
"""

import os
import zipfile
import shutil
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(r"c:\Users\LIKITHA\OneDrive\Desktop\Anemia Detection")
DATASETS_DIR = WORKSPACE_ROOT / "datasets"

ARCHIVES = [
    {
        "name": "dataset_cleaned.zip",
        "path": DATASETS_DIR / "dataset_cleaned.zip",
        "extract_to": DATASETS_DIR / "dataset_cleaned",
    },
    {
        "name": "Palm_preprocessed.zip",
        "path": DATASETS_DIR / "Palm_preprocessed.zip",
        "extract_to": DATASETS_DIR / "Palm_preprocessed",
    },
]


def is_safe_path(target_dir: Path, path: str) -> bool:
    """Prevent zip-slip path traversal vulnerability."""
    target_dir_resolved = target_dir.resolve()
    dest_path = (target_dir / path).resolve()
    return target_dir_resolved in dest_path.parents or dest_path == target_dir_resolved


def is_system_file(filename: str) -> bool:
    """Filter out OS artifacts like __MACOSX, .DS_Store, Thumbs.db."""
    parts = Path(filename).parts
    if "__MACOSX" in parts:
        return True
    if any(p.startswith("._") for p in parts):
        return True
    base = Path(filename).name
    if base in [".DS_Store", "Thumbs.db", "desktop.ini"]:
        return True
    return False


def verify_archive(archive_path: Path) -> dict:
    """Verifies archive existence, size, header, and CRC integrity."""
    result = {
        "exists": archive_path.exists(),
        "size_bytes": 0,
        "size_mb": 0.0,
        "is_zip": False,
        "crc_ok": False,
        "bad_file": None,
        "total_entries": 0,
        "clean_entries": 0,
        "error": None,
    }

    if not archive_path.exists():
        result["error"] = f"File not found: {archive_path}"
        return result

    size = archive_path.stat().st_size
    result["size_bytes"] = size
    result["size_mb"] = round(size / (1024 * 1024), 2)

    if not zipfile.is_zipfile(archive_path):
        result["error"] = f"Not a valid zip file: {archive_path}"
        return result
    result["is_zip"] = True

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            bad_file = zf.testzip()
            if bad_file is not None:
                result["bad_file"] = bad_file
                result["crc_ok"] = False
                result["error"] = f"Corrupted file in archive: {bad_file}"
            else:
                result["crc_ok"] = True

            namelist = zf.namelist()
            result["total_entries"] = len(namelist)
            result["clean_entries"] = len([f for f in namelist if not is_system_file(f)])
    except Exception as e:
        result["error"] = str(e)

    return result


def extract_archive(archive_path: Path, dest_dir: Path, force: bool = False) -> dict:
    """Safely extracts zip archive contents, skipping system noise."""
    res = {
        "archive": archive_path.name,
        "dest": str(dest_dir),
        "status": "pending",
        "extracted_files": 0,
        "skipped_files": 0,
        "error": None,
    }

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    existing_items = [p for p in dest_dir.iterdir() if not is_system_file(p.name)]
    if existing_items and not force:
        res["status"] = "already_extracted"
        res["extracted_files"] = len(list(dest_dir.rglob("*")))
        print(f"[{archive_path.name}] Destination already populated ({res['extracted_files']} files/folders found). Skipping re-extraction.")
        return res

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            infolist = zf.infolist()
            print(f"[{archive_path.name}] Extracting {len(infolist)} entries to {dest_dir}...")
            for member in infolist:
                # Security check
                if not is_safe_path(dest_dir, member.filename):
                    print(f"  [SECURITY WARNING] Traversal attempt blocked: {member.filename}")
                    res["skipped_files"] += 1
                    continue

                # Filter system junk
                if is_system_file(member.filename):
                    res["skipped_files"] += 1
                    continue

                # Extract
                zf.extract(member, dest_dir)
                res["extracted_files"] += 1

        res["status"] = "success"
    except Exception as e:
        res["status"] = "failed"
        res["error"] = str(e)
        print(f"[{archive_path.name}] Extraction failed: {e}")

    return res


def main():
    print("=" * 70)
    print("PHASE 1: TASK 1 & TASK 2 - ZIP VERIFICATION & SAFE EXTRACTION")
    print("=" * 70)

    for item in ARCHIVES:
        archive_path = item["path"]
        dest_dir = item["extract_to"]
        print(f"\nVerifying: {item['name']}")
        print(f"  Path: {archive_path}")

        v = verify_archive(archive_path)
        print(f"  Exists: {v['exists']}")
        print(f"  File Size: {v['size_bytes']:,} bytes ({v['size_mb']} MB)")
        print(f"  Valid ZIP Header: {v['is_zip']}")
        print(f"  Integrity Check (CRC testzip): {'PASSED' if v['crc_ok'] else 'FAILED'}")
        if v["bad_file"]:
            print(f"  First Bad CRC File: {v['bad_file']}")
        print(f"  Total Archive Entries: {v['total_entries']}")
        print(f"  Clean Entries (excl. system files): {v['clean_entries']}")

        if not v["crc_ok"] or not v["is_zip"]:
            print(f"  [ERROR] Cannot proceed to extraction: {v['error']}")
            sys.exit(1)

        print(f"\nExtracting: {item['name']} -> {dest_dir.name}")
        e = extract_archive(archive_path, dest_dir, force=False)
        print(f"  Status: {e['status']}")
        print(f"  Extracted items: {e['extracted_files']}")
        print(f"  Skipped system/junk items: {e['skipped_files']}")

    print("\nVerification & Extraction completed successfully.")


if __name__ == "__main__":
    main()
