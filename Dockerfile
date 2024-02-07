FROM nytimes/blender:2.93-gpu-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y software-properties-common curl
RUN apt-get install -y build-essential
RUN apt-get install -y cmake
RUN apt-get install -y git
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libc++1

RUN apt-get install -y python3.8
RUN apt-get install -y python3-pip
RUN apt-get install -y unzip
RUN apt-get install -y rsync

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /src
COPY requirements.txt /src/
# COPY pointnet2_ops_lib /src/pointnet2_ops_lib
RUN pip3 install -r requirements.txt
# RUN pip3 install pointnet2_ops_lib/.
COPY . /src/