# SELMA3D2026 Task 1 No-SSL Baseline Submission

Algorithm: ResEnc-L nnU-Net No-SSL Baseline

## Model

- Framework: nnU-Net v2
- Architecture: 3D full-resolution Residual Encoder U-Net, ResEnc-L plan
- Dataset: Dataset501_SELMA3D_BinaryBaseline
- Training: from scratch
- SSL pretraining: none
- Postprocessing: none
- Checkpoint: checkpoint_best.pth

## Grand Challenge Interface

Input:
- `/input/images/light-sheet-3d-microscopy/<uuid>.mha`
- `/input/images/light-sheet-3d-microscopy/<uuid>.tif`

Output:
- `/output/images/isolated-biological-structure/<uuid>.mha`

Output mask is binary:
- 0 = background
- 1 = biological structure

## Local Validation

A Grand Challenge-style local I/O test was run on AutoDL.

Result:
- input image was read from the expected input socket path
- nnU-Net inference completed successfully
- binary `.mha` segmentation was written to the expected output socket path
- output shape and image metadata should match the input image
