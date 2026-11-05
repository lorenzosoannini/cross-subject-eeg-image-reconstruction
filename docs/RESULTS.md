# Reported thesis results

These tables reproduce the experimental results retained with the thesis, rather than new evaluations of the consolidated portfolio checkout. Full-precision values are stored in the linked CSV files. Identification measures use pairwise two-way comparisons; EfficientNet and SwAV measure matched-pair feature correlation distance. Up/down arrows show the preferred direction.

## Full vocabulary: 1,654 targets per export

Thesis Table 5.2, printed page 44 (PDF page 52). Subject 10 is the heldout participant in the reported protocol; subject 4 is a source-subject reference. Both use embedding-mean aggregation. [CSV](../results/full_vocabulary.csv)

| Metric | S10 base | S10 enhanced | S4 base | S4 enhanced |
|---|---:|---:|---:|---:|
| PixCorr ↑ | 0.1169 | 0.1188 | 0.1108 | 0.1123 |
| SSIM ↑ | 0.3363 | 0.3762 | 0.3426 | 0.3774 |
| AlexNet-2 ↑ | 79.15% | 79.31% | 77.45% | 77.59% |
| AlexNet-5 ↑ | 88.97% | 87.75% | 87.56% | 86.89% |
| Inception ↑ | 89.38% | 88.58% | 90.24% | 89.64% |
| CLIP ↑ | 87.60% | 86.37% | 89.94% | 89.27% |
| EfficientNet ↓ | 0.6313 | 0.6342 | 0.6040 | 0.6027 |
| SwAV ↓ | 0.4492 | 0.4635 | 0.4446 | 0.4558 |

## Learned versus residual decomposition: 100 matched targets

The source model is replaced by two learned branches with joint latent reconstruction and separation objectives. The comparison evaluates that combined intervention, not subtraction alone. Base and enhanced learned-decomposition outputs improve all eight metrics on the reported matched set. Thesis Table 5.3. [CSV](../results/decomposition_100.csv)

| Metric | Residual base | Learned base | Residual enhanced | Learned enhanced |
|---|---:|---:|---:|---:|
| PixCorr ↑ | 0.0170 | 0.1128 | 0.0164 | 0.1149 |
| SSIM ↑ | 0.3461 | 0.3545 | 0.3594 | 0.3925 |
| AlexNet-2 ↑ | 55.60% | 77.29% | 55.47% | 78.77% |
| AlexNet-5 ↑ | 58.21% | 89.90% | 58.14% | 88.31% |
| Inception ↑ | 58.06% | 89.03% | 58.99% | 90.47% |
| CLIP ↑ | 58.87% | 87.58% | 59.03% | 87.09% |
| EfficientNet ↓ | 0.8977 | 0.6430 | 0.8958 | 0.6512 |
| SwAV ↓ | 0.6301 | 0.4497 | 0.6592 | 0.4634 |

## First response versus mean embedding: 100 matched targets

The proposed-model checkpoint and target set are fixed. `first` uses the first eligible response under the loader order; `embedding_mean` averages separately encoded responses before the prior. Thesis Table 5.7, printed page 50 (PDF page 58). [CSV](../results/aggregation_100.csv)

| Metric | First base | Mean base | First enhanced | Mean enhanced |
|---|---:|---:|---:|---:|
| PixCorr ↑ | 0.0895 | 0.1128 | 0.0923 | 0.1149 |
| SSIM ↑ | 0.3465 | 0.3545 | 0.3869 | 0.3925 |
| AlexNet-2 ↑ | 69.36% | 77.29% | 70.94% | 78.77% |
| AlexNet-5 ↑ | 75.98% | 89.90% | 75.06% | 88.31% |
| Inception ↑ | 77.22% | 89.03% | 73.02% | 90.47% |
| CLIP ↑ | 70.73% | 87.58% | 69.30% | 87.09% |
| EfficientNet ↓ | 0.7760 | 0.6430 | 0.7864 | 0.6512 |
| SwAV ↓ | 0.5118 | 0.4497 | 0.5344 | 0.4634 |

## ATM comparison: full-vocabulary subject 10

ATM was adapted to joint source-subject training and evaluated with prior-based and encoder-only generation. Resolution-dependent feature metrics were recalculated using the thesis evaluation functions. This is a system-level comparison; the residual-decomposition ablation is the controlled architectural comparison. Thesis Table 5.5 and defence PDF page 16. [Harmonized ATM CSV](../results/atm_harmonized.csv)

| Metric | ATM prior | ATM encoder-only | Proposed base |
|---|---:|---:|---:|
| PixCorr ↑ | 0.1187 | 0.1456 | 0.1169 |
| SSIM ↑ | 0.3713 | 0.3738 | 0.3363 |
| AlexNet-2 ↑ | 60.98% | 67.30% | 79.15% |
| AlexNet-5 ↑ | 63.80% | 72.39% | 88.97% |
| Inception ↑ | 59.56% | 63.71% | 89.38% |
| CLIP ↑ | 59.27% | 63.48% | 87.60% |
| EfficientNet ↓ | 0.9062 | 0.9055 | 0.6313 |
| SwAV ↓ | 0.6006 | 0.6086 | 0.4492 |

The proposed model has higher identification and lower feature distances in this comparison. ATM encoder-only generation has higher PixCorr and SSIM. Enhancement improves some structural scores but does not consistently improve semantic correspondence.

## Interpretation and implementation provenance

- The reported experiment uses one heldout participant. It does not establish population-wide transfer performance.
- Evaluation categories occur during training. The intended exclusions concern participant and photograph identity, rather than unseen categories.
- The main result aggregates repeated EEG responses and should not be described as single-trial reconstruction.
- Each candidate photograph represents a different category. Identification does not establish within-category photographic specificity.
- Scores are aggregate estimates without confidence intervals; selected examples are qualitative illustrations.
- The accompanying source was consolidated for publication. The explicitly fixed 8/1/1 split, pooled source-only statistics and semantic prior route are recorded in [PROTOCOL.md](PROTOCOL.md); historical checkpoints do not become compliant through renaming. A complete rerun has not been performed.

The thesis PDF and defence PDF were inspected to verify the displayed scores. Their hashes, source figures and CSV provenance are recorded in [result_sources.json](result_sources.json). Full thesis documents, EEG recordings and pretrained weights are not bundled in this source release. The figure panels preserve the original reported examples; they were rasterized without altering reconstructed content or numerical values.
