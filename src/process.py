from pathlib import Path
import os
import shutil
import subprocess
import sys

import numpy as np
import SimpleITK as sitk


DATASET_NAME = "Dataset501_SELMA3D_BinaryBaseline"
PLANS_NAME = "nnUNetResEncUNetLPlans"
CONFIGURATION = "3d_fullres"
FOLD = "0"

INPUT_DIR = Path(os.environ.get(
    "GC_INPUT_DIR",
    "/input/images/light-sheet-3d-microscopy",
))
OUTPUT_DIR = Path(os.environ.get(
    "GC_OUTPUT_DIR",
    "/output/images/isolated-biological-structure",
))
MODEL_DIR = Path(os.environ.get(
    "MODEL_DIR",
    "/opt/ml/model",
))

WORK_DIR = Path(os.environ.get("WORK_DIR", "/tmp/selma3d_nnunet_work"))
NNUNET_RAW = WORK_DIR / "nnUNet_raw"
NNUNET_PREPROCESSED = WORK_DIR / "nnUNet_preprocessed"
NNUNET_RESULTS = WORK_DIR / "nnUNet_results"

TMP_IMAGES = WORK_DIR / "input_images"
TMP_PRED = WORK_DIR / "predictions"

CASE_ID = "selma3d_case"


def find_input_image(input_dir: Path) -> Path:
    candidates = []
    for pattern in ("*.mha", "*.tif", "*.tiff"):
        candidates.extend(sorted(input_dir.glob(pattern)))

    if not candidates:
        raise FileNotFoundError(f"No .mha/.tif/.tiff image found in {input_dir}")

    if len(candidates) > 1:
        print(f"Warning: multiple input images found, using first: {candidates[0]}", file=sys.stderr)

    return candidates[0]


def prepare_nnunet_layout():
    raw_dataset = NNUNET_RAW / DATASET_NAME
    pre_dataset = NNUNET_PREPROCESSED / DATASET_NAME
    res_base = (
        NNUNET_RESULTS
        / DATASET_NAME
        / f"nnUNetTrainer__{PLANS_NAME}__{CONFIGURATION}"
    )
    res_fold = res_base / "fold_0"

    raw_dataset.mkdir(parents=True, exist_ok=True)
    pre_dataset.mkdir(parents=True, exist_ok=True)
    res_fold.mkdir(parents=True, exist_ok=True)
    TMP_IMAGES.mkdir(parents=True, exist_ok=True)
    TMP_PRED.mkdir(parents=True, exist_ok=True)

    shutil.copy2(MODEL_DIR / "dataset.json", raw_dataset / "dataset.json")
    shutil.copy2(MODEL_DIR / "nnUNetResEncUNetLPlans.json", pre_dataset / "nnUNetResEncUNetLPlans.json")

    # nnUNetv2_predict expects dataset.json and plans.json in the trained model folder,
    # while the checkpoint itself lives under fold_0/.
    shutil.copy2(MODEL_DIR / "dataset.json", res_base / "dataset.json")
    shutil.copy2(MODEL_DIR / "nnUNetResEncUNetLPlans.json", res_base / "plans.json")
    shutil.copy2(MODEL_DIR / "checkpoint_best.pth", res_fold / "checkpoint_best.pth")

    return raw_dataset, pre_dataset, res_fold


def run_prediction(input_image: Path):
    image = sitk.ReadImage(str(input_image))

    nnunet_input = TMP_IMAGES / f"{CASE_ID}_0000.nii.gz"
    sitk.WriteImage(image, str(nnunet_input))

    env = os.environ.copy()
    env["nnUNet_raw"] = str(NNUNET_RAW)
    env["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    env["nnUNet_results"] = str(NNUNET_RESULTS)
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    env.setdefault("NUMEXPR_NUM_THREADS", "4")

    cmd = [
        "nnUNetv2_predict",
        "-i", str(TMP_IMAGES),
        "-o", str(TMP_PRED),
        "-d", "501",
        "-c", CONFIGURATION,
        "-f", FOLD,
        "-p", PLANS_NAME,
        "-tr", "nnUNetTrainer",
        "-chk", "checkpoint_best.pth",
    ]

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)

    pred_path = TMP_PRED / f"{CASE_ID}.nii.gz"
    if not pred_path.exists():
        raise FileNotFoundError(f"nnU-Net prediction not found: {pred_path}")

    pred = sitk.ReadImage(str(pred_path))
    pred_arr = sitk.GetArrayFromImage(pred)
    pred_arr = (pred_arr > 0).astype(np.uint8)

    out_img = sitk.GetImageFromArray(pred_arr)
    out_img.CopyInformation(image)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{input_image.stem}.mha"
    sitk.WriteImage(out_img, str(output_path))

    print(f"Wrote output: {output_path}", flush=True)


def main():
    print(f"Input dir: {INPUT_DIR}", flush=True)
    print(f"Output dir: {OUTPUT_DIR}", flush=True)
    print(f"Model dir: {MODEL_DIR}", flush=True)

    input_image = find_input_image(INPUT_DIR)
    print(f"Using input image: {input_image}", flush=True)

    prepare_nnunet_layout()
    run_prediction(input_image)


if __name__ == "__main__":
    main()
