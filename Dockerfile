FROM python:3.10

RUN pip install omnisafe

RUN pip install IPython

#TODO: move all requirements to requirements.txt and replace above lines such that everything from the text file is installed

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y sudo ca-certificates openssl \
    git ssh build-essential gcc g++ cmake make \
    python3-dev python3-venv python3-opengl libosmesa6-dev && \
    rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL osmesa
ENV PYOPENGL_PLATFORM osmesa

CMD ["python", "./app/main.py"]