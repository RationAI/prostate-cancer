FROM rayproject/ray:latest-py311

RUN sudo apt-get update -y \
    && sudo apt-get install -y --no-install-recommends openssh-client libvips-dev gcc \
    && sudo rm -rf /var/lib/apt/lists/* \
    && sudo apt-get clean

RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan gitlab.ics.muni.cz >> ~/.ssh/known_hosts


COPY . .

RUN --mount=type=ssh,uid=1000 pip --no-cache-dir install -r requirements.txt \
    && rm requirements.txt \
    && rm -rf ~/.ssh 
