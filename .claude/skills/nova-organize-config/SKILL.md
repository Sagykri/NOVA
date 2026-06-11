---
name: nova-organize-config
description: Interactively generates a Python config.py for the NOVA images_organizer tool. Use when setting up a new microscopy experiment that needs raw images sorted into the NOVA folder hierarchy.
model: claude-sonnet-4-6
tools:
  - Read
  - Write
  - Bash
---

# nova-organize-config

Guides the user step-by-step through every required field for a new `config.py`. Never assume or invent values — ask explicitly for each piece of information before writing anything.

## Config format reference

Two established patterns exist in `tools/images_organizer/`:

**Pattern A — numeric index** (`batch_3_6_version/config.py`, `deltaNLS_version/config_dnls.py`): each cell-line/condition maps to `(min_idx, max_idx)` tuples; one tuple per replicate.

**Pattern B — well position** (`RANBP17_exp/`, `neurons_spd18days_version/`): each cell-line/condition maps to `(row, col)` tuples; one tuple per replicate. Uses a class-based `Config` with `self.*` attributes.

Both patterns share the same key names: `KEY_CELL_LINES`, `KEY_MARKERS_ALIAS_ORDERED`, `KEY_MARKERS`, `KEY_REPS`. Both support `CUT_FILES`, `DRY_RUN`, `SKIP_EXISTING_FILES`, `RAISE_ON_MISSING_INDEX`.

## Instructions

Work through Steps 1–8 in order. After each step, wait for the user's answer before proceeding. Do not bundle multiple steps into one question.

---

### Step 1 — Paths

Ask the following questions one at a time:

1. "What is the **source path** (`SRC_ROOT_PATH`) — where are the raw image files located?"
2. "What is the **output path** (`DST_ROOT_PATH`) — where should organized images be written?"
3. "What is the **logging path** (`LOGGING_PATH`)? Suggested default: `<DST_ROOT_PATH>/logs`"

---

### Step 2 — Experiment metadata

Ask:

1. "What is the **experiment name**? (Used to name the output config folder, e.g. `RANBP17`, `FuNOVA_screen`)"
2. "What **batch label** should be used? (e.g. `batch1`)"
3. "What **file extension** do the images use — `.tif` or `.tiff`?"

---

### Step 3 — Mapping type

Ask:

"How are images indexed in their filenames?

**A) Numeric index** — filenames contain a sequential number, e.g. `conf_Cy5_1234.tif`. Mapping uses `(min_idx, max_idx)` ranges.

**B) Well position (row/col)** — filenames encode plate position, e.g. `R02C04-ch1.tiff`. Mapping uses `(row, col)` tuples.

Which pattern matches your data (A or B)?"

---

### Step 4 — Markers / panels

Ask:

1. "List the **channel aliases in order** as they appear in filenames (e.g. `DAPI, mCherry, GFP, Cy5`). These become `KEY_MARKERS_ALIAS_ORDERED`."
2. "List the **panel name(s)** for this experiment (e.g. `panelA`, `panelB`)."
3. For each panel, ask: "For **<panel_name>**, what is the biological marker for each channel in order? Use `None` for unused channels. Provide them in the same order as the channel aliases."

---

### Step 5 — Cell lines, conditions, and replicates (mapping)

Offer the user two input modes:

"How would you like to provide the cell-line / condition / replicate mapping?

**M) Manual entry** — I'll ask for the global replicate count and any exceptions, then collect positions one by one.
**P) Paste mapping** — Paste the full mapping dict directly as Python (e.g. `{'iW11': {'control': [(2,4),(3,4)], ...}}`).

Which mode (M or P)?"

#### Mode P — Paste mapping

Ask: "Paste the complete mapping dict now. Use the format:
```
{
    'cell_line': {
        'condition': [(val_rep1), (val_rep2), ...],
    },
}
```
Values are `(min_idx, max_idx)` for type A or `(row, col)` for type B."

Parse and echo back the mapping for user confirmation before proceeding.

#### Mode M — Manual entry

Ask these questions in order:

1. "How many **replicates** do most conditions have? (default replicate count)"
2. "Are there any **exceptions** — conditions that have a different number of replicates? If yes, list them as `cell_line / condition: N` (one per line). If none, press Enter to skip."
3. "List all **cell line names** (comma-separated)."
4. For each cell line: "List the **condition names** for **<cell_line>** (comma-separated)."
5. For each cell line / condition pair, ask for each replicate position using the default count (or the exception count if applicable):
   - If mapping type A: "What is the **(min_index, max_index)** for **<cell_line> / <condition> / rep<N>**?"
   - If mapping type B: "What is the **(row, col)** well position for **<cell_line> / <condition> / rep<N>**?"

---

### Step 6 — Replicate labels

Ask:

"Confirm the **replicate labels** to use (e.g. `rep1, rep2`). These become `KEY_REPS`. The count must match the replicate count given above (and exceptions where applicable)."

---

### Step 7 — Processing options

Ask each option separately with its default shown:

1. "**CUT_FILES**: Move files instead of copying? (`False` = copy, `True` = move) [default: False]"
2. "**DRY_RUN**: Run without actually moving/copying files (for testing)? [default: False]"
3. "**SKIP_EXISTING_FILES**: Skip files that already exist at the destination? [default: True]"
4. "**RAISE_ON_MISSING_INDEX**: Raise an error if an index is not found in the config? [default: False]"

---

### Step 8 — Output config path

Ask:

"Where should the config file be written? Suggested path:
`tools/images_organizer/<ExperimentName>/config.py`

Confirm or provide a different path."

Then use `Bash` to create the directory if it does not exist:
```bash
mkdir -p <directory>
```

---

## Writing the config file

After collecting all answers, write the `config.py` using the `Write` tool.

Use the **class-based pattern** (matching the newer `neurons_spd18days_version` / `RANBP17_exp` style) with `self.*` attributes inside `__init__`. This is the preferred modern format.

Template to follow:

```python
######################################################
########## Please Don't Change This Section ##########
######################################################

import os

class Config():
    def __init__(self):
        super().__init__()

        self.LOGGING_PATH = "<LOGGING_PATH>"
        self.KEY_CELL_LINES = "cell_lines"
        self.KEY_MARKERS_ALIAS_ORDERED = "markers_alias_ordered"
        self.KEY_MARKERS = "markers"
        self.KEY_REPS = "reps"
        self.FILE_EXTENSION = "<.tif or .tiff>"

        #####################################################################

        # You may change the configuration beneath this line

        #####################################
        ############### Paths ###############
        #####################################

        self.SRC_ROOT_PATH = "<SRC_ROOT_PATH>"
        self.DST_ROOT_PATH = "<DST_ROOT_PATH>"

        self.FOLDERS = []
        self.EXCLUDE_SUB_FOLDERS = []
        self.INCLUDE_SUB_FOLDERS = []

        self.DRY_RUN = <True/False>
        self.SKIP_EXISTING_FILES = <True/False>
        self.CUT_FILES = <True/False>
        self.RAISE_ON_MISSING_INDEX = <True/False>

        ########################################
        ############### Advanced ###############
        ########################################

        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "<cell_line>": {
                    "<condition>": [<(min,max) or (row,col) per rep>],
                },
            },
            self.KEY_MARKERS_ALIAS_ORDERED: [<channel aliases in order>],
            self.KEY_MARKERS: {
                "<panelA>": [<marker per channel, None for unused>],
            },
            self.KEY_REPS: [<rep labels>],
        }
```

**Rules for filling the template:**
- Preserve the header comment block exactly.
- For mapping type A (numeric index): values in `KEY_CELL_LINES` are lists of `(min_idx, max_idx)` tuples, one per replicate.
- For mapping type B (well position): values in `KEY_CELL_LINES` are lists of `(row, col)` tuples, one per replicate.
- `None` values in `KEY_MARKERS` must be unquoted Python `None`, not the string `"None"`.
- String paths must use actual collected values, not placeholder text.
- Do not add a `BATCH` key at top level — batch is encoded in the folder structure or in `KEY_BATCHES` if needed. Only add `KEY_BATCHES` if the user explicitly has batch-to-plate mappings.
- If the user pasted the mapping dict (Mode P), write it verbatim (reformatted cleanly) into `KEY_CELL_LINES`.

---

## Validation before finishing

After writing the file, confirm:
- All user-provided paths are present verbatim (no placeholders remain).
- `KEY_REPS` list length matches the default replicate count (exceptions noted in comments if any).
- Each panel's marker list length matches `KEY_MARKERS_ALIAS_ORDERED` length.
- No API keys, tokens, passwords, or secrets are present.

Report the absolute path of the written file to the user.
