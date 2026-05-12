3D FURNITURE GENERATION ON IPAD — PROJECT GOALS
=================================================

PRIMARY GOAL
------------
Learn 3D deep learning end-to-end while producing a demo that looks good
enough to show people. Two tracks running in parallel, feeding each other.


TRACK 1: LEARNING (BUILD IT FROM SCRATCH)
-----------------------------------------
Goal: Understand WHY the field works the way it does by hitting the
failure modes yourself.

Stages:
  1. Non-learned baseline
     - Fixed template mesh -> RealityKit on iPad
     - Builds the full pipeline before adding ML

  2. PointNet on point clouds (no images yet)
     - Learn Chamfer distance, set-based losses, permutation invariance
     - Isolates 3D learning from 2D understanding

  3. AtlasNet with image conditioning
     - Add ResNet image encoder
     - Train on chairs only (PIX3D)
     - Run ablations: Chamfer alone, +Laplacian, different templates

  4. Modern loss swap
     - Try normal consistency or differentiable rendering loss
     - This is often where output quality jumps significantly

  5. iPad port
     - MLX-Swift + LowLevelMesh + zero-copy vertex updates
     - Done AFTER the model works, not during


TRACK 2: RESULTS (STAND ON GIANTS' SHOULDERS)
---------------------------------------------
Goal: Have something that looks good within ~1 week.

Stages:
  1. Get InstantMesh or TripoSR running locally on Mac
  2. Wire it into the same RealityKit pipeline as Track 1
  3. Optional: distill it into a smaller on-device model


SCOPE DECISIONS
---------------
- Chairs only. One category done well > all furniture done mediocrely.
- Curated demo inputs (clean backgrounds, canonical angles).
- Texture projection from input image (cheap trick, big visual win).
- Aggressive post-processing: smoothing, remeshing, removing
  disconnected components.


TECH STACK
----------
- Training:   Python + MLX (or PyTorch) on Mac
- Datasets:   Pix3D chairs to start; Objaverse later if needed
- Export:     .safetensors
- Inference:  MLX-Swift on iPad
- Rendering:  RealityKit LowLevelMesh (zero-copy from model output)
- UI:         SwiftUI + RealityView


KEY MINDSET
-----------
- The from-scratch model is for LEARNING. It will not match TripoSR.
- The impressive demo uses the foundation-model track.
- Both are real accomplishments. Don't conflate them.
- Failure modes in Track 1 are the curriculum, not the problem.


OPEN QUESTIONS TO REVISIT
-------------------------
- Switch from Chamfer to EMD or normal consistency — when?
- Differentiable rendering (NVDiffRast / DIB-R) — worth the complexity?
- Distillation target: full InstantMesh, or just its geometry head?
- Editability: can a user drag a chair leg and keep it plausible?