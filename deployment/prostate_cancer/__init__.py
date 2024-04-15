from prostate_cancer.ingress import Ingress
from prostate_cancer.model import Model

app = Ingress.bind(Model.bind())
