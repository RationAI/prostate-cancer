FROM rayproject/ray:2.10.0-py310-gpu


# COPY requirements.txt .
# RUN pip --no-cache-dir install -r requirements.txt && rm requirements.txt

COPY model.onnx $HOME/model.onnx

# WORKDIR /app


# WORKDIR /serve_app
# COPY prostate_cancer /serve_app/prostate_cancer


# FROM nvcr.io/nvidia/tritonserver:24.03-py3

# COPY requirements.txt .
# RUN pip --no-cache-dir install -r requirements.txt && rm requirements.txt