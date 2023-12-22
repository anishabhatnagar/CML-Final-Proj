# Start with a Linux micro-container to keep the image tiny
# FROM alpine:3.7
FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

# Document who is responsible for this image
MAINTAINER Anisha Bhatnagar "ab10945@nyu.edu"

# Install just the Python runtime (no dev)
RUN apt-get update && apt-get install -y \
    python3 python3-dev gcc gfortran musl-dev \
    python3-pip ca-certificates

# Set up a working folder and install the pre-reqs
WORKDIR /app
ADD requirements.txt /app
RUN pip3 install --upgrade pip setuptools 
RUN pip3 install torch===1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install -r requirements.txt
# Add the code as the last Docker layer because it changes the most

ADD Step-5623_checkpoint_lang_pred.pt /app/Step-5623_checkpoint_lang_pred.pt
ADD app.py  /app/app.py

# Run the service
WORKDIR /app
CMD [ "python3", "app.py"]


