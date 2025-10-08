from jaxtyping import Float
from torch import Tensor


EmbeddingsLocs = Float[Tensor, "K 3"]  # [K,3] -> (b, i, j)
Embeddings = Float[Tensor, "K C"]
Cams = Float[Tensor, "B H W"]
