# CV-Project

Segmentation + monocular depth -> scene understanding pipeline for driving
scenes. See `prompts/01_architecture_specification.md` for the full design.

## Project layout

- `configs/` — YAML configuration groups (`unet`, `midas`, `pipeline`).
- `utils/` — config loading, seeding, logging, device selection.
- `preprocessing/` — dataset loaders and shared transforms.
- `tests/` — pytest suite with synthetic fixtures (no dataset downloads needed).
- `prompts/` — step-by-step task briefs and the architecture specification.

## Datasets

Loaders never pair images by sorted lists; pairs are matched by the *stem* of
the file name and validated, so mispairs are impossible.

### Cityscapes (`preprocessing/cityscapes.py`)

Expected layout — either the official split layout or a flat one:

```
data/cityscapes/
  leftImg8bit/{train,val,test}/{city}/{id}_leftImg8bit.png   # official
  gtFine/    {train,val,test}/{city}/{id}_gtFine_labelIds.png
  # or flat:
  images/{id}_leftImg8bit.png
  labels/{id}_gtFine_labelIds.png
```

- Label PNGs are raw `labelIds` (0..33), remapped to trainId space
  (0..18, 255 = ignore) with the official lookup table from the Cityscapes
  scripts — see the `CITYSCAPES_LABEL_ID_TO_TRAIN_ID` dict.
- Labels are resized with nearest-neighbour interpolation only.
- A sample is `{image, label, image_path, label_path}`; `label`/`label_path`
  are `None` when `require_labels=False`.
- Missing labels honour `missing_policy` (`error` or `skip`).

### KITTI depth (`preprocessing/kitti.py`)

Expected layout (flat sibling directories plus optional split files):

```
data/kitti/
  images/{stem}.png
  depth/{stem}.png              # 16-bit, values in millimetres
  splits/{train,val,test}.txt   # optional; one stem per line
```

- Depth is converted to float32 metres (`value / scale_mm`, default 1000).
- Depth `0` means invalid; the sample carries `valid_mask = depth > 0`.
  `depth_cap_m` flags pixels beyond a distance as invalid.
- Depth resizing is bilinear and the valid mask is recomputed afterwards, so
  interpolation never leaks into invalid pixels.
- A split file may list bare stems, relative paths, or `scene/frame` style
  listings; identical resolution by stem. Without a split file every stem
  present in both directories is used.

### Development mode

`max_samples` limits the number of pairs loaded (`null` = full dataset). The
same loader serves both development runs (e.g. `10`) and full training.

## Tests

```bash
.venv/bin/python -m pytest -q
```