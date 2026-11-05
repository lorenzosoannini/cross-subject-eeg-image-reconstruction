# Third-party components

- [ZEBRA](https://github.com/xmed-lab/ZEBRA): visual projectors, supervision heads, feature prior, image-generation interface and evaluation definitions, adapted for EEG and the selected protocol.
- [REVE](https://huggingface.co/brain-bzh/reve-large): pretrained EEG representation. Its weights and remote model implementation are obtained separately.
- [Stability AI generative-models](https://github.com/Stability-AI/generative-models): selected SGM dependencies for image decoding and refinement. Their original code license is preserved in `generative_models/LICENSE-CODE`; available SDXL model-license texts are retained under `generative_models/model_licenses/`. Image-only adaptations remove temporal/video paths.
- [THINGS-EEG2](https://osf.io/anp5v/): EEG recordings and visual-stimulus metadata are obtained separately under the dataset's terms. No EEG recordings or stimulus images are bundled.
- OpenCLIP, CLIP, BLIP-2, diffusers and dalle2-pytorch supply external model components. Their weights are not redistributed here and retain their respective upstream terms.

`docs/source_provenance.json` identifies local source modules and extracted symbols. `docs/vendor_manifest.json` records the upstream-local SGM source hashes used during extraction. This repository does not assign a new blanket license to third-party-derived code.

Reference: Wang et al., *ZEBRA: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding*, NeurIPS 2025.
