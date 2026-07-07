from collections.abc import Sequence

from rationai.staining import ColorConversion, NormalizeStainingTransform


def build_normalize_staining_transform(
    stain1: Sequence[float],
    stain2: Sequence[float],
    stain3: Sequence[float],
    target_stain1: Sequence[float],
    target_stain2: Sequence[float],
    target_stain3: Sequence[float],
) -> NormalizeStainingTransform:
    conversion = ColorConversion.from_stain_vectors(
        tuple(stain1), tuple(stain2), tuple(stain3)
    )
    return NormalizeStainingTransform(
        rgb2stain=conversion.matrix,
        target_stain1=tuple(target_stain1),
        target_stain2=tuple(target_stain2),
        target_stain3=tuple(target_stain3),
    )
