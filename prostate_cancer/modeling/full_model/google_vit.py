from transformers import ViTForImageClassification


def google_vit() -> ViTForImageClassification:
    return ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=1,
        id2label={0: "carcinoma"},
        label2id={"carcinoma": 0},
        ignore_mismatched_sizes=True,
    )
