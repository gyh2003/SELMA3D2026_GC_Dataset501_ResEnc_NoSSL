# SELMA3D 2026 No SSL Baseline

Original submission source for the **ResEnc-L nnU-Net baseline** used in the SELMA3D 2026 final evaluation without self-supervised learning, together with model files recovered from an archived baseline training backup.

The challenge baseline uses **one fold-0 model trained from scratch**. The original submission source is preserved under the tag **`v0.1.3-dataset501-resenc-no-ssl`**.

## 1. Files and links

| Item | Location |
| --- | --- |
| Original submission source | [Browse the v0.1.3 tag](https://github.com/gyh2003/SELMA3D2026_GC_Dataset501_ResEnc_NoSSL/tree/v0.1.3-dataset501-resenc-no-ssl) |
| Model weights, approximately 760 MB | [Download model_dataset501_fold0_recovered.tar.gz](https://github.com/gyh2003/SELMA3D2026_GC_Dataset501_ResEnc_NoSSL/releases/download/v0.1.3-dataset501-resenc-no-ssl/model_dataset501_fold0_recovered.tar.gz) |
| Model checksums | [Download MODEL_SHA256SUMS.txt](https://github.com/gyh2003/SELMA3D2026_GC_Dataset501_ResEnc_NoSSL/releases/download/v0.1.3-dataset501-resenc-no-ssl/MODEL_SHA256SUMS.txt) |
| Original container record | [Grand Challenge image](https://grand-challenge.org/algorithms/resenc-l-nnu-net-no-ssl-baseline/images/fbd2b791-a4ab-4671-b25f-cb68421f94ea/) |
| Final evaluation | [Grand Challenge result](https://selma3d2026.grand-challenge.org/evaluation/47790c8d-86aa-43b9-9ce6-208ebc4a86ac/) |

**Weights are distributed as a Release attachment**, separately from the source code. Some Grand Challenge pages require a login.

## 2. Model overview

| Setting | Challenge baseline |
| --- | --- |
| Framework | nnU-Net v2, inference package `nnunetv2==2.7.0` |
| Architecture | ResEnc-L residual encoder U-Net, 3D full resolution |
| Training data | Official annotated data, using the training partition of the original fold-0 split |
| Initialization | Random initialization; no SSL pretraining |
| Checkpoint | `checkpoint_best.pth` |
| Patch size | `160 × 224 × 192` |
| Batch size | 2 |
| Normalization | Z-score normalization |
| Inference | `nnUNetv2_predict`, fold 0; the exact command is in `src/process.py` |
| Ensemble and routing | None |
| Additional postprocessing | None; the predicted mask is saved as binary `uint8` |

The source code retains the internal dataset identifier `Dataset501_SELMA3D_BinaryBaseline` and plan name `nnUNetResEncUNetLPlans` for compatibility with the checkpoint.

## 3. Model archive and provenance

Extracting the model archive produces three files directly at its root:

```text
checkpoint_best.pth
dataset.json
nnUNetResEncUNetLPlans.json
```

Mount this directory at **`/opt/ml/model`** when running the container.

The downloadable archive was prepared on **20 November 2026** from the archived baseline training backup. The checkpoint and configuration files were copied without modification and checked against that backup using SHA-256. This is a **repackaged recovery archive**. The original Grand Challenge model archive was not downloaded, so equality between the recovered checkpoint and the checkpoint in that upload has not been directly verified.

| File | SHA-256 |
| --- | --- |
| `model_dataset501_fold0_recovered.tar.gz` | `fec21c01a747c870da99530f5710adff57528292bf5133cec7c4b2de19b12b03` |
| `checkpoint_best.pth` | `8d061a4703a90f9cc370635e9269945b2111f9a7c04c93baef84d1e5046cc401` |

Check the downloaded archive with:

```bash
shasum -a 256 model_dataset501_fold0_recovered.tar.gz
```

## 4. Build and run

**Validation status:** the commands below are usage instructions. A new Docker build and GPU inference run using this recovery package have not been performed.

Use a Docker host with NVIDIA GPU support for inference. Build from the original submission tag:

```bash
git clone --branch v0.1.3-dataset501-resenc-no-ssl --depth 1 \
  https://github.com/gyh2003/SELMA3D2026_GC_Dataset501_ResEnc_NoSSL.git \
  selma3d-no-ssl
cd selma3d-no-ssl

docker build --platform linux/amd64 -t selma3d-no-ssl:original .
```

Download the model archive using the link above. Replace `/path/to/` below with its download directory:

```bash
mkdir -p model input/images/light-sheet-3d-microscopy output
tar -xzf /path/to/model_dataset501_fold0_recovered.tar.gz -C model
```

Place **one** `.mha`, `.tif`, or `.tiff` input volume in `input/images/light-sheet-3d-microscopy/`. The container runs as a non-root user, so the model and input files must be readable and the dedicated output directory must be writable:

```bash
chmod -R a+rX model input
chmod a+rwx output

docker run --rm --gpus all \
  -v "$PWD/model:/opt/ml/model:ro" \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output" \
  selma3d-no-ssl:original
```

The original container uses the **Exec API**: its entry point is `python /opt/algorithm/src/process.py`, and it exits after writing the prediction. The `app.py` file is retained from development but is not the original submission's entry point.

A fresh build may resolve different versions of dependencies that were not pinned in the original requirements. The historical container identity is recorded below for exact version identification.

## 5. Input and output

| Item | Container path |
| --- | --- |
| Input | `/input/images/light-sheet-3d-microscopy/<case>.mha` or `<case>.tif` |
| Model directory | `/opt/ml/model/` |
| Output | `/output/images/isolated-biological-structure/<case>.mha` |

Output values are **0 for background** and **1 for the target biological structure**, with the input image geometry preserved. The challenge output socket is named `isolated-biological-structure`; this binary output interface is also used for contiguous cases.

## 6. Original Grand Challenge submission

The successful final evaluation submitted on **28 September 2026** references the container built on **15 August 2026** from the v0.1.3 tag and the original model upload.

| Record | Identifier |
| --- | --- |
| Source commit | `c27567436d3d6cc28cdb804551726a9739f9c8e1` |
| Container image version | `fbd2b791-a4ab-4671-b25f-cb68421f94ea` |
| Container image SHA-256 | `24a6b06a90b52b14bd5a3a7777f6f588f5a05b249e9910d09e765685da523550` |
| Original model version | `a80ec112-6843-4c88-a548-1fe715b2dc0d` |
| Original uploaded model filename | `dataset501_resenc_no_ssl_model.tar.gz` |
| Final evaluation ID | `47790c8d-86aa-43b9-9ce6-208ebc4a86ac` |

The original container is identified by the Grand Challenge record above. This Release provides source and recovered model weights; it does not include an export of the original built image.

### Final evaluation metrics

| Structure group | Metric | Mean ± standard deviation |
| --- | --- | --- |
| Isolated | DSC | 0.5672 ± 0.2405 |
| Isolated | PQ | 0.3414 ± 0.2378 |
| Contiguous | DSC | 0.6379 ± 0.1900 |
| Contiguous | clDice | 0.7108 ± 0.1904 |

These scores are the historical Grand Challenge final evaluation results for the image and model versions above. They have not been recomputed using the recovered archive.

## 7. Repository contents

```text
Dockerfile              Original container build definition
requirements.txt        Inference dependencies
src/process.py          Exec inference entry point
app.py                  Earlier development server wrapper
README_submission.md    Original submission notes
LICENSE                 Source-code license
```

The `main` branch includes this handover documentation. Use the v0.1.3 tag when referring to the original submission source.
