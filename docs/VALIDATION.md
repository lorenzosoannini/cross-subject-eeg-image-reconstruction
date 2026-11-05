# Validation — 5 November 2026

Completed locally on CPU with Python 3.12.14, PyTorch 2.5.0 and NumPy 2.4.6:

- **17 tests passed**, with no failures or errors. See `test_results.json` and `test_results.txt`.
- **Nine CLI help checks passed**: trainer, reconstruction, metrics, captioning, refinement, protocol preparation, high-pass filtering, recording indexing and image-token caching. See `cli_checks.json`. Help checks do not load the model weights or exercise a GPU backend.
- All **65 Python source files** compiled successfully. Shell syntax validation passed for `run.sh`.
- Image partitions were generated from the local THINGS-EEG2 visual catalogue: **13,232 / 1,654 / 1,654** disjoint training/validation/evaluation photographs, sharing all 1,654 categories.

The tests cover deterministic per-category image partitioning; real-sample population moments; exclusion of source validation, evaluation and heldout EEG from normalization; rejection of changed normalization provenance; frozen source normalization and zero padding on target input; absence of target CLIP tokens from the evaluation loader; training/inference F_s conditioning; gradients through the intended projection; independent learned latent branches; LoRA gating; subject gradient reversal; nonlinear-then-average repetition aggregation; prior forward/backward and exactly 20 corrected-DDIM transitions; image center cropping; blurry decoder output geometry; pairwise metric pairing/ties; required SGM YAML targets; fixed generated-image caption inputs; and removal of retired variable/logging names.

The full EEG dataset was not reprocessed to fit the new statistics in this task. The estimator and preparation commands are provided; their behavior was tested on controlled fixtures. REVE and image-generator weights were not downloaded. Full CUDA training, complete model checkpoint loading, image generation/refinement and pretrained feature-extractor evaluation were not run. The local test environment lacks the full generation dependencies, including OmegaConf and PyTorch Lightning; these are listed in `requirements.txt` for the deployment environment.

The numerical metric functions retain their original definitions. CLIP is imported lazily inside its metric so unrelated CPU tests do not require loading the CLIP package. The dependency list includes the legacy setuptools interface required by that external CLIP implementation. No local environment packages were modified.

This verification establishes the tested component contracts and source integrity. It does not certify reproduction of historical thesis scores. The new partition, normalization and corrected semantic prior route require fresh compatible checkpoints.
