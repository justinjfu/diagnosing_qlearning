FROM ubuntu:18.04

# === Essentials ===
RUN apt-get update
RUN apt-get install -y git python3 curl python3-pip

# === Setup Python ===
WORKDIR /root/code
RUN cd /root/code

RUN pip3 install --upgrade pip
RUN pip3 install six parameterized Cython path.py cloudpickle
RUN pip3 install scipy matplotlib
RUn pip3 install torch torchvision
RUN pip3 install tensorflow
RUN pip3 install gym plotly

