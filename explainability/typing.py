from torch import Tensor
from jaxtyping import Float


EmbeddingsLocs = Float[Tensor, "K 3"]  # [K,3] -> (b, i, j)
Embeddings = Float[Tensor, "K C"]
Cams = Float[Tensor, "B H W"]
