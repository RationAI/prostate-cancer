from transformers import ViTModel

def google_vit() -> ViTModel:
    return ViTModel.from_pretrained("google/vit-base-patch16-224")
