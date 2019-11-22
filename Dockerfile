#FROM  aaftio/face_recognition:latest
FROM ubuntu:latest

WORKDIR /app

COPY requirement.txt .

RUN apt-get update && apt-get install python3-pip -y 

RUN  apt-get install -y cmake

RUN pip3 install -r requirement.txt 

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev

COPY cv_api/ .