FROM rayproject/ray:2.11.0-py310


RUN sudo apt-get update -y \
    && sudo apt-get install -y libvips-dev gcc --no-install-recommends \
    && sudo rm -rf /var/lib/apt/lists/* \
    && sudo apt-get clean

COPY . .

RUN pip --no-cache-dir install -r requirements.txt && rm requirements.txt