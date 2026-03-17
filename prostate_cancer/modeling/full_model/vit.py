from transformers import ViTForImageClassification


def google_vit_full_model() -> ViTForImageClassification:
    return ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
