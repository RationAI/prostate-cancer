from app.ingress import Ingress
from app.model import Model
from app.tiling_service import TilingService

app = Ingress.bind(Model.bind(), TilingService.bind())
