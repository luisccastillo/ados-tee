FROM ubuntu:20.04

RUN  apt-get update \
    && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh && \
    bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p /root/miniconda

ENV CONDA_AUTO_UPDATE_CONDA="false"
ENV PATH=/root/miniconda/bin:$PATH

RUN conda install numpy
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch
#RUN conda install torch-geometric
RUN conda install pyg -c pyg
RUN conda install pandas
RUN conda install matplotlib
RUN conda install scikit-learn

RUN ls /root/miniconda/bin/
COPY ./src /app

ENTRYPOINT ["/root/miniconda/bin/python3.8", "/app/app.py"]
