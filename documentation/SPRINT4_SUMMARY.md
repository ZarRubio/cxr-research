# Sprint 4 summary — CNN-ViT multi-label

Sprint 4 extends the thesis pipeline to multi-label classification over all 14 NIH ChestX-ray14 findings. The final system ensembles two CNN-ViT variants: a four-block Transformer model and a six-block Transformer model.

## Final configuration

| Component | v1 | v2 |
|---|---|---|
| CNN backbone | DenseNet121 | DenseNet121 |
| Transformer blocks | 4 | 6 |
| Attention heads | 8 | 8 |
| Embedding dimension | 512 | 512 |
| MLP dimension | 1,024 | 1,024 |

Historical metric files are not included in the repository. The earlier DenseNet backbone used TorchXRayVision weights trained on NIH ChestX-ray14, so the NIH holdout does not provide an independent estimate of generalization. The previous comparison with Wang et al. (2017) is not retained because the split and evaluation protocols are not matched.

Historical metrics and calibrated thresholds are excluded from this repository.

## Known limitations

- Labels are weakly supervised and inherit limitations of the NIH dataset.
- The historical validation/test protocol sampled one image per patient, and rare findings had few positives.
- The NIH-pretrained initialization overlaps with the target dataset and prevents a clean generalization claim.
- The model has not been prospectively validated in a clinical workflow.
- Results must not be interpreted as clinical performance claims.

The revised configuration uses a patient-level multilabel-stratified split, keeps all images from a patient in one partition, and includes a CheXpert-initialized experiment with an NIH-pretrained control. Test metrics are computed from aggregate predictions with patient-cluster bootstrap intervals.
