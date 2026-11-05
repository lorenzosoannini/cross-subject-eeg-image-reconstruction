# Cross-Subject EEG-to-Image Reconstruction

Can an image reconstruction model work for a person it was not trained on? For my master's thesis, I investigated this question using THINGS-EEG2 recordings, a pretrained REVE encoder and the visual-semantic generation framework of [ZEBRA](https://github.com/xmed-lab/ZEBRA). The project adapts ZEBRA from fMRI to EEG and replaces its fixed neural residual with two learned representation branches.

The task is **cross-subject reconstruction of unseen photographs from familiar categories**. Subjects 1–9 are the source participants; subject 10 is the heldout participant in the reported experiment. This is not a category-zero-shot task.

<p align="center">
  <img src="assets/reconstructions.png" width="690" alt="Four selected thesis examples: target photograph, EEG-conditioned base reconstruction and enhanced reconstruction for heldout subject 10">
</p>

*Selected examples from the thesis: aquarium, lifebuoy, polar bear and sunflower. Columns show the target, base reconstruction and caption-refined reconstruction. These examples illustrate recovered visual content; their selection does not estimate a success rate.*

## From EEG to a visual condition

REVE encodes 63-channel EEG windows and is adapted with LoRA. A learned token resampler bridges its channel-indexed features to the visual token interface. Two independently learned branches represent shared and subject-related information, with joint reconstruction and covariance/variance objectives supporting information preservation and separation.

The architectural change concerns the neural space: instead of defining `E_s = E − E_i`, both `E_i` and `E_s` are learned from the complete EEG representation `E`. The visual residual `F_i = F − F_s` is retained. The shared branch supplies `F_s`, the sole neural condition for the diffusion prior and SDXL unCLIP generator. Subject heads, the private branch and latent reconstruction support training without becoming extra generator inputs.

![Thesis architecture: REVE, learned shared/private branches, visual projection and semantic diffusion conditioning](assets/architecture.png)

*Architecture from the thesis. The joint latent decoder reconstructs the EEG representation; the blurry decoder provides auxiliary image-latent supervision. The token resampler is included in the projection paths.*

Repeated responses to the same photograph are encoded separately. Their `F_s` embeddings are averaged before prior sampling. An optional refinement stage captions the generated image with BLIP-2 and uses that caption in image-to-image diffusion. Ground-truth photographs and category labels are not inference conditions.

## Thesis results

On the full vocabulary, the reported evaluation contains **1,654 target photographs** for subject 10, with one target per category. The table below reports the base and enhanced outputs from that experiment.

| Metric | Base | Enhanced |
|---|---:|---:|
| PixCorr ↑ | 0.1169 | 0.1188 |
| SSIM ↑ | 0.3363 | 0.3762 |
| AlexNet-2 identification ↑ | 79.15% | 79.31% |
| AlexNet-5 identification ↑ | 88.97% | 87.75% |
| Inception identification ↑ | 89.38% | 88.58% |
| CLIP identification ↑ | 87.60% | 86.37% |
| EfficientNet distance ↓ | 0.6313 | 0.6342 |
| SwAV distance ↓ | 0.4492 | 0.4635 |

Identification scores measure **two-way comparisons**, with a nominal chance reference of 50%. They are not top-1 accuracy over 1,654 candidates. Enhancement increases SSIM here while slightly reducing several semantic identification scores.

Two controlled comparisons use the same **100-target development set**:

- Replacing residual decomposition with learned branches raises base CLIP identification from **58.87% to 87.58%** (+28.71 percentage points).
- With the proposed-model checkpoint fixed, averaging response embeddings raises base CLIP identification from **70.73% to 87.58%** (+16.85 points) compared with the first retained response.

![Thesis controlled comparisons of neural decomposition and repeated-response aggregation](assets/controlled-comparisons.png)

*Subject 10, base reconstructions, 100 targets. Both interventions improve all eight reported metrics for base and enhanced outputs. The decomposition comparison measures the combined architectural and regularization change; it does not isolate a single loss term.*

The thesis also evaluates an adapted [ATM](https://github.com/ncclab-sustech/EEG_Image_decode) baseline. On the full-vocabulary subject-10 comparison, the proposed model has stronger feature-based correspondence, while ATM encoder-only generation has higher PixCorr and SSIM. [Full results, comparisons and limitations](docs/RESULTS.md) · [Machine-readable tables](results/)

## What the images still get wrong

Recovering recognizable content does not guarantee the original viewpoint, object count, pose or layout. The thesis includes failures such as altered animal anatomy, router-to-boat substitution and a walrus reconstructed as a rodent. Refinement can preserve an incorrect interpretation.

<details>
<summary>Selected failure cases and source-subject comparisons</summary>

![Thesis failure examples comparing target, subject-10 outputs and subject-4 source-reference outputs](assets/failure-cases.png)

*Columns: target, subject-10 base/enhanced, subject-4 base/enhanced. These are selected diagnostic examples, not aggregate evidence that source-subject reconstruction is always better.*

</details>

The reported transfer result concerns one heldout participant. Broader participant-level generalization requires additional folds. The main results use repeated responses, and the identification candidate pool does not test fine-grained discrimination among photographs of the same category.

## Running the implementation

The source is organized around preprocessing, source-only normalization, learned decomposition, representation pretraining, `F_s`-conditioned downstream training, image generation and paired evaluation. EEG variable names replace inherited fMRI names; abandoned experimental and target-calibration workflows are excluded.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Full training and generation require CUDA, THINGS-EEG2 and external pretrained weights. [Data preparation and commands](docs/RUNNING.md) · [Configuration](config.json) · [Protocol](docs/PROTOCOL.md) · [Validation](docs/VALIDATION.md)

**Results provenance.** The figures and scores above come from the completed thesis and defence materials. This repository is the accompanying consolidated implementation, with explicitly documented partition, normalization and prior-routing changes for publication. Those changes require compatible retraining; the published checkout has not been rerun to reproduce the historical scores. [Source records](docs/result_sources.json)

**Master's thesis project by Lorenzo Soannini**, Engineering in Computer Science, Sapienza University of Rome. Thesis: *EEG-to-Image Reconstruction through Multimodal Visual–Semantic Alignment*. Supervisor: Prof. Danilo Comminiello. [Third-party components and attribution](THIRD_PARTY.md)
