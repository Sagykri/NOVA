"""Spec-driven config generation for FuNOVA_Screen UMAPs.

Workflow:
    runs = [
        {"data": {"name": "Plate1_AllCond", "conditions": plate1_conditions},
         "plot": {"color_by": "condition"}},
        {"data": {"name": "Plate1_woDAPI", "conditions": plate1_conditions,
                  "markers_to_exclude": ["HDGFL2", "FK-2", "DAPI"]},
         "plot": {"color_by": "marker"}},
    ]
    cmds = submit_runs(runs, memory=10000, submit=False)

Each call to submit_runs() APPENDS new class blocks to the two _generated.py
files. If a class with the same name already exists in the file, it is left
alone — the new spec is *not* written. This protects bsub jobs that are still
queued: their class definition won't change underneath them. To change a spec
that's already written, either pick a new `name`, or pass `reset=True` to wipe
the files first.
"""

import re
import subprocess
from pathlib import Path

# -- file locations -----------------------------------------------------------
_HERE = Path(__file__).resolve().parent
GENERATED_DATA_FILE = _HERE / "manuscript_figures_data_config_FuNOVA_Screen_generated.py"
GENERATED_PLOT_FILE = _HERE / "manuscript_plot_config_FuNOVA_Screen_generated.py"

# bsub-arg paths. The runnable's get_class() turns "./manuscript/X" into
# `import manuscript.X`. For that to find the user's manuscript dir
# (giliwo/NOVA/manuscript) — even when the bsub job runs against a different
# NOVA_HOME (e.g. Collaboration/NOVA) — the bsub command must put the user's
# NOVA root on PYTHONPATH. _bsub_command() does that, derived from this file's
# location.
DATA_MODULE_PATH = "./NOVA/manuscript/manuscript_figures_data_config_FuNOVA_Screen_generated"
PLOT_MODULE_PATH = "./NOVA/manuscript/manuscript_plot_config_FuNOVA_Screen_generated"

# Default user NOVA root: the dir above this file (i.e., giliwo/NOVA).
USER_NOVA_HOME = "./NOVA"

# -- color_by → (default UMAP type, MapLabelsFunction member, color attr) -----
COLOR_BY_MAP = {
    "rep":                 (0, "REPS",                  "COLOR_MAPPINGS_FUNOVA_SCREEN_REPS"),
    "batch":               (0, "BATCHES",               "COLOR_MAPPINGS_FUNOVA_SCREEN_BATCHES"),
    "condition":           (0, "CONDITIONS",            "COLOR_MAPPINGS_FUNOVA_SCREEN_CONDITIONS"),
    "cell_line":           (0, "CELL_LINES",            "COLOR_MAPPINGS_FUNOVA_SCREEN_CELL_LINES"),
    "cell_line_condition": (0, "CELL_LINES_CONDITIONS", "COLOR_MAPPINGS_FUNOVA_SCREEN_CELL_LINE_CONDITIONS"),
    "marker":              (1, "MARKERS",               "COLOR_MAPPINGS_FUNOVA_SCREEN_MARKERS"),
}

# spec-key → BaseFigureConfig attribute name
DATA_KEY_TO_ATTR = {
    "experiment_type":    "EXPERIMENT_TYPE",
    "input_folders":      "INPUT_FOLDERS",
    "cell_lines":         "CELL_LINES",
    "conditions":         "CONDITIONS",
    "markers":            "MARKERS",
    "markers_to_exclude": "MARKERS_TO_EXCLUDE",
    "add_rep_to_label":   "ADD_REP_TO_LABEL",
    "add_batch_to_label": "ADD_BATCH_TO_LABEL",
    "show_ari":           "SHOW_ARI",
    "saveroot_infix":     "SAVEROOT_INFIX",
}


def _safe(name):
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_") or "unnamed"


_CLASS_RE = re.compile(r'^class\s+(\w+)\s*\(', re.MULTILINE)


def _existing_class_blocks(path):
    """Parse a generated file. Return ordered {classname: full_block_str}.
    Returns {} if the file doesn't exist or has no top-level class defs."""
    p = Path(path)
    if not p.exists():
        return {}
    text = p.read_text()
    matches = list(_CLASS_RE.finditer(text))
    blocks = {}
    for i, m in enumerate(matches):
        cls_name = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks[cls_name] = text[start:end].rstrip("\n") + "\n"
    return blocks


def _gen_data_class(spec):
    name = "FuNOVA_Screen_Data_" + _safe(spec.get("name", "unnamed"))
    lines = [f"class {name}(FuNOVA_Screen_BaseFigureConfig):",
             "    def __init__(self):",
             "        super().__init__()"]
    body_lines = []
    for key, attr in DATA_KEY_TO_ATTR.items():
        if key in spec:
            body_lines.append(f"        self.{attr} = {repr(spec[key])}")
    lines.extend(body_lines or ["        pass"])
    return name, "\n".join(lines)


def _gen_plot_class(spec, data=None):
    color_by = spec["color_by"]
    if color_by not in COLOR_BY_MAP:
        raise ValueError(
            f"Unknown color_by={color_by!r}. Valid: {sorted(COLOR_BY_MAP)}"
        )
    default_umap, map_func, color_attr = COLOR_BY_MAP[color_by]
    umap_type = spec.get("umap_type", default_umap)
    size = spec.get("size", 1)
    alpha = spec.get("alpha", 1)
    figsize = spec.get("figsize", None)
    overrides = spec.get("color_overrides", None)  # {item: '#hex'} or {item: {alias, color}}
    full_replace = spec.get("color_mappings", None)  # full {item: {alias, color}} dict

    name = "FuNOVA_Screen_Plot_" + _safe(spec.get("name", "unnamed")) + f"_{color_by}"

    lines = [f"class {name}(FuNOVA_Screen_BasePlotConfig):",
             "    def __init__(self):",
             "        super().__init__()",
             f"        self.MAP_LABELS_FUNCTION = MapLabelsFunction.{map_func}.name",
             f"        self.UMAP_TYPE = {umap_type}",
             f"        self.SIZE = {size}",
             f"        self.ALPHA = {alpha}"]

    # Decide the base palette expression. For condition / cell_line_condition,
    # build a palette sized exactly to THIS run's items so colors stay distinct
    # when only a few conditions are shown.
    data = data or {}
    run_conditions = data.get("conditions")
    run_cell_lines = data.get("cell_lines")
    if full_replace is not None:
        palette_expr = repr(full_replace)
    elif color_by == "condition" and run_conditions:
        palette_expr = f"self.make_condition_palette({list(run_conditions)!r})"
    elif color_by == "cell_line_condition" and run_conditions:
        cls = list(run_cell_lines) if run_cell_lines else ["C9"]
        palette_expr = f"self.make_cell_line_condition_palette({cls!r}, {list(run_conditions)!r})"
    else:
        palette_expr = f"self.{color_attr}"

    if overrides:
        lines.append(f"        self.COLOR_MAPPINGS = dict({palette_expr})")
        for item, val in overrides.items():
            if isinstance(val, str):
                lines.append(
                    f"        self.COLOR_MAPPINGS[{item!r}] = {{"
                    f"self.MAPPINGS_ALIAS_KEY: {item!r}, "
                    f"self.MAPPINGS_COLOR_KEY: {val!r}}}"
                )
            else:
                lines.append(f"        self.COLOR_MAPPINGS[{item!r}] = {repr(val)}")
    else:
        lines.append(f"        self.COLOR_MAPPINGS = {palette_expr}")

    lines.append("        self.COLOR_MAPPINGS_MARKERS = self.COLOR_MAPPINGS_FUNOVA_SCREEN_MARKERS")
    if figsize is not None:
        lines.append(f"        self.FIGSIZE = {tuple(figsize)!r}")

    return name, "\n".join(lines)


_DATA_HEADER = '''"""AUTO-GENERATED by funova_screen_run_helpers.submit_runs(). Do not edit by hand."""
import os, sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from NOVA.manuscript.manuscript_figures_data_config_FuNOVA_Screen import (
    FuNOVA_Screen_BaseFigureConfig,
    FUNOVA_SCREEN_MARKERS,
    FUNOVA_SCREEN_BATCHES,
    FUNOVA_SCREEN_CELL_LINES,
    FUNOVA_SCREEN_ALL_CONDITIONS,
    FUNOVA_SCREEN_CONDITIONS_PLATES,
    FUNOVA_SCREEN_CONTROL_CONDITIONS_PLATES,
    FUNOVA_SCREEN_KDS_CONDITIONS_PLATES
)


'''

_PLOT_HEADER = '''"""AUTO-GENERATED by funova_screen_run_helpers.submit_runs(). Do not edit by hand."""
import os, sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from NOVA.manuscript.manuscript_plot_config_FuNOVA_Screen import FuNOVA_Screen_BasePlotConfig
from src.datasets.label_utils import MapLabelsFunction


'''


def _bsub_command(idx, data_class, plot_class, dir_label, memory,
                  nova_home=None, model_path=None, user_nova_home=None):
    """Build a bsub command string.

    nova_home       : NOVA root the runnable uses (gets src/, runnables/) — e.g. Collaboration/NOVA.
    user_nova_home  : NOVA root that holds the (possibly user-edited) manuscript/
                      configs — e.g. giliwo/NOVA. Put on PYTHONPATH so
                      `manuscript.X` imports resolve to the user's files.
    model_path      : usually `$MODEL_PATH` (shell-expanded at submit time).
    """
    user_nova_home = user_nova_home or USER_NOVA_HOME
    return (
        f'"bsub -q short -R rusage[mem={memory}] '
        f'-o {dir_label}/{data_class}_{idx}.out '
        f'-J umap_{idx} '
        f'python {nova_home}/runnables/generate_umaps_and_plot.py {model_path} '
        f'{DATA_MODULE_PATH}/{data_class} '
        f'{PLOT_MODULE_PATH}/{plot_class}"'
    )


def submit_runs(runs, memory=10000, dir="$run_type", submit=False, verbose=True,
                nova_home=None, model_path="$MODEL_PATH", user_nova_home=None,
                reset=False):
    """Generate config classes from `runs` specs, append them to the two .py
    files (skipping any class names that already exist), and return / optionally
    execute the bsub commands.

    Each run is a dict: {"data": {...spec...}, "plot": {...spec..., "color_by": ...}}

    reset=True wipes both _generated.py files before writing — use this only
    when no previously-submitted bsub jobs still need their old classes.
    """
    import os
    os.makedirs(dir, exist_ok=True)

    if reset:
        data_blocks = {}
        plot_blocks = {}
    else:
        data_blocks = _existing_class_blocks(GENERATED_DATA_FILE)
        plot_blocks = _existing_class_blocks(GENERATED_PLOT_FILE)

    pairs = []
    new_data, kept_data = [], []
    new_plot, kept_plot = [], []
    for run in runs:
        d_name, d_src = _gen_data_class(run["data"])
        p_name, p_src = _gen_plot_class(
            {**run["plot"], "name": run["data"].get("name", "unnamed")},
            data=run["data"],
        )
        pairs.append((d_name, p_name))

        if d_name in data_blocks:
            kept_data.append(d_name)
        else:
            data_blocks[d_name] = d_src + "\n"
            new_data.append(d_name)

        if p_name in plot_blocks:
            kept_plot.append(p_name)
        else:
            plot_blocks[p_name] = p_src + "\n"
            new_plot.append(p_name)

    GENERATED_DATA_FILE.write_text(_DATA_HEADER + "\n\n".join(data_blocks.values()))
    GENERATED_PLOT_FILE.write_text(_PLOT_HEADER + "\n\n".join(plot_blocks.values()))

    if verbose:
        print(f"Data: +{len(new_data)} new, {len(kept_data)} already present (file now has {len(data_blocks)} classes)")
        print(f"Plot: +{len(new_plot)} new, {len(kept_plot)} already present (file now has {len(plot_blocks)} classes)")
        if kept_data:
            print(f"  data classes kept (not regenerated): {kept_data}")
        if kept_plot:
            print(f"  plot classes kept (not regenerated): {kept_plot}")

    cmds = [
        _bsub_command(i, d, p, dir, memory,
                      nova_home=nova_home, model_path=model_path,
                      user_nova_home=user_nova_home)
        for i, (d, p) in enumerate(pairs)
    ]
    print("\n".join(cmds))

    if submit:
        for cmd in cmds:
            inner = cmd.strip('"')
            if verbose:
                print(f"\n>>> submitting: {inner}")
            subprocess.run(inner, shell=True, check=False)

    return cmds
