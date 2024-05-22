from app.ingress import Ingress
from app.model import Model
from app.upload_service import UploadService

app = Ingress.bind(Model.bind(), UploadService.bind())
