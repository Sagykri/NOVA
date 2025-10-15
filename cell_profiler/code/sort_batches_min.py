from pathlib import Path
import shutil
from datetime import datetime
import os
import sys
import re

# --- SETTINGS YOU CAN EDIT ---


BASE_DIR = os.path.join('/home', 'projects', 'hornsteinlab', 'Collaboration', 'NOVA')S
sys.path.insert(1, BASE_DIR)


BASE = os.path.join(BASE_DIR,"input","images","raw","Sorbitol_experiment_PBs_TDP43")
BASE = Path(BASE)
PANELS = ["PanelA", "PanelB", "PanelC", "PanelD"]   # or ["PanelD"] for only D
DRY_RUN = False                                      # True = preview only
UNDO = False                                         # True = undo the LAST run
# EXTENSIONS = {".tif", ".tiff"}                     # optional filter; uncomment to use
EXTENSIONS = None
# --- END SETTINGS ---

BATCH_MAP = {
    "batch1": ("c01", "c02", "c03", "c04"),
    "batch2": ("c05", "c06", "c07", "c08"),
    "batch3": ("c09", "c10", "c11", "c12"),
}

LOG_DIR = BASE / "_sort_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def which_batch(name: str):
    n = name.lower()
    m = re.search(r'c(\d{2})', n)
    if not m:
        return None
    col = m.group(1)  # '01'..'12'
    if col in {"01", "02", "03", "04"}:
        return "batch1"
    if col in {"05", "06", "07", "08"}:
        return "batch2"
    if col in {"09", "10", "11", "12"}:
        return "batch3"
    return None

def _print_move(src: Path, dst: Path, dry: bool):
    print(f"{'[DRY] ' if dry else ''}{src} -> {dst}")

def safe_move(src: Path, dst_dir: Path, dry: bool):
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / src.name
    if target.exists():
        stem, suffix = target.stem, target.suffix
        i = 1
        while True:
            alt = dst_dir / f"{stem} ({i}){suffix}"
            if not alt.exists():
                target = alt
                break
            i += 1
    _print_move(src, target, dry)
    if not dry:
        shutil.move(str(src), str(target))
    return target

def safe_restore(current: Path, desired: Path, dry: bool):
    desired.parent.mkdir(parents=True, exist_ok=True)
    target = desired
    if target.exists():
        stem, suffix = desired.stem, desired.suffix
        i = 1
        while True:
            alt = desired.parent / f"{stem} (restored {i}){suffix}"
            if not alt.exists():
                target = alt
                break
            i += 1
    _print_move(current, target, dry)
    if not dry:
        shutil.move(str(current), str(target))
    return target

def find_latest_log():
    logs = sorted(LOG_DIR.glob("undo_*.log"))
    return logs[-1] if logs else None

def do_sort():
    # open a new undo log
    log_path = LOG_DIR / f"undo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"# undo log created {datetime.now().isoformat()}\n")
        for panel in PANELS:
            panel_dir = BASE / panel
            if not panel_dir.is_dir():
                print(f"Skipping missing panel: {panel_dir}")
                continue
            print(f"\n=== Sorting {panel_dir} ===")
            for f in panel_dir.glob("*"):
                if not f.is_file():
                    continue
                if EXTENSIONS and f.suffix.lower() not in EXTENSIONS:
                    continue
                batch = which_batch(f.name)
                if batch:
                    dst_dir = panel_dir / batch
                    before = f.resolve()
                    after = safe_move(f, dst_dir, DRY_RUN)
                    if not DRY_RUN:
                        # Log as: MOVE <src> <dst>
                        log.write(f"MOVE\t{before}\t{after}\n")
                else:
                    # Not matched; leave as-is
                    pass
    print(f"\nUndo log saved to: {log_path}")

def do_undo():
    log_path = find_latest_log()
    if not log_path:
        print("No undo logs found.")
        return
    print(f"Using undo log: {log_path}")
    # Read all moves, then reverse apply (dst -> src)
    moves = []
    with open(log_path, "r", encoding="utf-8") as log:
        for line in log:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3 and parts[0] == "MOVE":
                src = Path(parts[1])  # original location (before sort)
                dst = Path(parts[2])  # actual location (after sort)
                moves.append((src, dst))
    if not moves:
        print("No MOVE entries found in log.")
        return

    # Apply in reverse order to avoid collisions
    for src, dst in reversed(moves):
        if not dst.exists():
            print(f"Missing file to restore (skipping): {dst}")
            continue
        safe_restore(dst, src, DRY_RUN)

def main():
    if UNDO:
        do_undo()
    else:
        do_sort()
    print("\nDone.")

if __name__ == "__main__":
    main()
