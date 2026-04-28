# Literature Search Gate: Explicit Cost Matrices, Modern Vision Backbones, and Class-Imbalance Decoupling

Date: 2026-04-28

This document is the pre-implementation literature gate for the NICME binary-first extension. The goal is to avoid overstating novelty and to keep the paper focused on the distinction that motivated the research proposal:

> Cost-sensitive classification is not the same problem as class-imbalanced classification.

NICME uses a user-defined pairwise cost matrix `C[true_label][predicted_label]` and must be evaluated on both balanced and naturally imbalanced datasets so cost effects are not conflated with class-frequency effects.

## Search Protocol

Search venues/databases:

- Google Scholar
- Semantic Scholar
- arXiv
- OpenReview
- IEEE Xplore
- ACM Digital Library
- PubMed
- Papers with Code
- Publisher pages for known papers

Query families:

- `"cost matrix" "Vision Transformer" "cost-sensitive" image classification`
- `"misclassification cost matrix" "Vision Transformer" image classification`
- `"cost-sensitive" "ConvNeXt" "cost matrix" image classification`
- `"EfficientNet" "cost sensitive regularization" "cost matrix"`
- `"explicit cost matrix" "deep learning" image classification`
- `"cost-sensitive" "class imbalance" "cost matrix" "image classification"`
- `"DINO" "cost-sensitive" "cost matrix" image classification`
- `"CLIP" "cost-sensitive" "cost matrix" image classification`

Inclusion criteria:

- Image classification.
- Explicit pairwise or `N x N` misclassification costs, or a clear cost-sensitive post-hoc decision rule.
- Modern vision backbone evidence if claimed: ViT, ConvNeXt, EfficientNet, DINO/CLIP-style foundation models.

Exclusion criteria:

- Class-frequency weighting only.
- Focal loss only.
- Resampling only.
- “Cost” means compute cost, acquisition cost, or annotation cost rather than misclassification consequence.

## Classification Tags

- **True cost-sensitive:** arbitrary/domain-defined pairwise costs, not derived from class frequency.
- **Imbalance-coupled:** costs or weights come from inverse frequency, effective number, priors, long-tail corrections, focal-style rarity weighting, or sampling.
- **Mixed/unclear:** uses cost-sensitive language but does not clearly separate class imbalance from misclassification cost.

## Gate Outcome

Proceed with the implementation, but revise the novelty claim.

The literature search does not support claiming that no ViT has been paired with explicit cost matrices. Chen et al. report additional ViT experiments for CSADA on CIFAR-10 with cost-sensitive benchmarks. Therefore the project should not claim absence of all modern-backbone cost-sensitive learning.

Recommended novelty wording:

> Many “cost-sensitive” deep learning papers address class imbalance through class weights, resampling, focal-style losses, or prior/logit corrections. NICME instead evaluates explicit user-defined pairwise misclassification costs under both balanced and imbalanced class distributions, isolating cost sensitivity from class-frequency effects. The extension studies NICME-style cost-matrix logit adjustment and cost-sensitive regularization with DINOv3/LoRA backbones.

Pause condition:

If a later search finds a method that already combines user-defined pairwise cost matrices, balanced/imbalanced decoupling experiments, NICME-like logit adjustment, CS regularization, and DINOv3/LoRA, implementation should pause and a gap-analysis memo should reframe the contribution.

## Evidence Table

| Work | Category | Backbone(s) | Costs | Balanced/Imbalanced Decoupling | Relevance |
|---|---|---|---|---|---|
| Elkan, 2001, “The Foundations of Cost-Sensitive Learning” | True cost-sensitive | Classical classifiers | Explicit costs / binary threshold | Conceptual, not vision deep learning | Establishes post-hoc minimum expected cost decision rule. |
| Domingos, 1999, “MetaCost” | True cost-sensitive | Wrapper over classifiers | Arbitrary cost matrix | Conceptual, not modern vision | Supports post-hoc/wrapper baseline framing. |
| Galdran et al., 2020, “Cost-Sensitive Regularization for Diabetic Retinopathy Grading from Eye Fundus Images” | True cost-sensitive | ResNeXt-50 in released code/paper context | Ordinal distance-style matrix for DR grade mistakes | Medical ordinal task; not primarily class-imbalance decoupling | Direct baseline for `cs_regularized_ce`; paper reports a regularizer that penalizes farther grade mistakes. |
| Menon et al., 2021, “Long-tail Learning via Logit Adjustment” | Imbalance-coupled | General deep classifiers | Class-prior adjustment, not user cost matrix | Long-tail/class-prior setting | Useful imbalance baseline, not a true cost-matrix baseline. |
| Chen et al., “Rethinking Cost-Sensitive Classification in Deep Learning via Adversarial Data Augmentation,” INFORMS JDS | True cost-sensitive | CNNs and ViT appendix; ResNet-34 on OCT | Generated/expert pairwise costs | Includes balanced-style CIFAR and expert-cost OCT experiments; not NICME-style DINOv3/LoRA or logit-adjustment regularization | Important prior; directly blocks “no ViT cost-sensitive work” claims. |
| “Diabetic retinopathy classification method based on cost sensitive regularization and EfficientNet,” 2022 | Mixed/unclear | EfficientNet | Cost-sensitive regularization; exact cost-vs-imbalance separation needs full-text verification | Appears DR-grade/domain severity oriented, but do not use a broad “no EfficientNet” claim | Treat as a counterexample candidate until fully adjudicated. |
| Hugging Face / Meta DINOv3 documentation and repository | Not a cost-sensitive method | DINOv3 ViT and ConvNeXt backbones | None | Foundation-backbone support only | Confirms implementation target: DINOv3 ViT and DINOv3 ConvNeXt are available in modern Transformers/timm ecosystems. |
| Hugging Face PEFT LoRA image-classification guide | Not a cost-sensitive method | ViT example | None | PEFT implementation guidance only | Supports default LoRA target modules `query,value` and saving classifier modules for image classification. |

## Experimental Consequence

The implementation must include both:

- Balanced datasets: primary evidence that NICME handles user-defined costs apart from class imbalance.
- Natural/controlled imbalanced datasets: deployment realism and comparison to imbalance-coupled baselines.

Every results table must report class prevalence so improvements cannot be mistaken for class-frequency effects.

## Sources

- Elkan, C. “The Foundations of Cost-Sensitive Learning.” IJCAI, 2001.
- Domingos, P. “MetaCost: A General Method for Making Classifiers Cost-Sensitive.” KDD, 1999.
- Galdran, A. et al. “Cost-Sensitive Regularization for Diabetic Retinopathy Grading from Eye Fundus Images.” MICCAI/arXiv, 2020: https://arxiv.org/abs/2010.00291
- Menon, A. K. et al. “Long-tail Learning via Logit Adjustment.” ICLR, 2021.
- Chen, Z. et al. “Rethinking Cost-Sensitive Classification in Deep Learning via Adversarial Data Augmentation.” INFORMS Journal on Data Science: https://pubsonline.informs.org/doi/10.1287/ijds.2022.0033
- “Diabetic retinopathy classification method based on cost sensitive regularization and EfficientNet,” 2022: https://cjlcd.lightpublishing.cn/en/article/doi/10.37188/CJLCD.2022-0161/
- Hugging Face DINOv3 documentation: https://huggingface.co/docs/transformers/en/model_doc/dinov3
- Meta DINOv3 repository: https://github.com/facebookresearch/dinov3
- Hugging Face PEFT LoRA image classification guide: https://huggingface.co/docs/peft/main/en/task_guides/image_classification_lora
- BreaKHis official dataset page: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
