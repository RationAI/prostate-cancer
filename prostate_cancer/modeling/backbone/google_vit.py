from transformers import ViTForImageClassification, ViTModel


def google_vit() -> ViTModel:
    return ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=1,
        id2label={0: "carcinoma"},
        label2id={"carcinoma": 0},
        ignore_mismatched_sizes=True,
    ).vit
