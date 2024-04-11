FROM rayproject/ray:2.10.0-py310


# COPY requirements.txt .
# RUN pip --no-cache-dir install -r requirements.txt && rm requirements.txt
RUN sudo apt-get update -y \
    && sudo apt-get install -y libvips-dev gcc --no-install-recommends \
    && sudo rm -rf /var/lib/apt/lists/* \
    && sudo apt-get clean


COPY model.onnx $HOME/model.onnx
COPY model.pt $HOME/model.pt
