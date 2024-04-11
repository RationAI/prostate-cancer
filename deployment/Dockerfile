FROM rayproject/ray:2.10.0-py310


# COPY requirements.txt .
# RUN pip --no-cache-dir install -r requirements.txt && rm requirements.txt
RUN apt-get update && apt-get install libvips-dev --no-install-recommends

COPY model.onnx $HOME/model.onnx
COPY model.pt $HOME/model.pt
