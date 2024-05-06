FROM python:3.10

# RUN pip install omnisafe

# RUN pip install IPython

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# ADD custom_envs/tasks/__init__.py /usr/local/lib/python3.10/site-packages/safety_gymnasium/tasks/__init__.py

# ADD custom_envs/__init__.py /usr/local/lib/python3.10/site-packages/safety_gymnasium/__init__.py

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y sudo ca-certificates openssl \
    git ssh build-essential gcc g++ cmake make \
    python3-dev python3-venv python3-opengl libosmesa6-dev && \
    rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL osmesa
ENV PYOPENGL_PLATFORM osmesa

CMD ["python", "./app/main.py"]