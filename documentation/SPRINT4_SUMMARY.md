# Sprint 4 summary — CNN-ViT multi-label

Sprint 4 extends the thesis pipeline to multi-label classification over all 14 NIH ChestX-ray14 findings. The final system ensembles two CNN-ViT variants: a four-block Transformer model and a six-block Transformer model.

## Final configuration

| Component | v1 | v2 |
|---|---:|---:|
| CNN backbone | DenseNet121 | DenseNet121 |
| Transformer blocks | 4 | 6 |
| Attention heads | 8 | 8 |
| Embedding dimension | 512 | 512 |
| MLP dimension | 1,024 | 1,024 |
| Validation macro AUC | 0.7909 | 0.7950 |

The final ensemble weights are 0.3 for v1 and 0.7 for v2.

## Evaluation

On 4,023 test images, the ensemble reached a macro AUC of 0.8045 and mean average precision of 0.1521. Its macro AUC was approximately 0.059 higher than the Wang et al. (2017) reference value of 0.7452, with 12 of 14 classes above the corresponding reference AUC.

Full metrics and calibrated thresholds are stored in `experiments_s4ml_v2/`.

## Known limitations

- Labels are weakly supervised and inherit limitations of the NIH dataset.
- Rare findings produce unstable precision and F1 estimates.
- Infiltration and Pneumothorax remain below the corresponding Wang et al. reference AUC values.
- The model has not been prospectively validated in a clinical workflow.
- Results must not be interpreted as clinical performance claims.

