# hash:sha256:45850247e3b62e9057d7d845ee2ec03fec2049e239230267285cf1b8ef412f4c
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        jupyter==1.0.0 \
        jupyter_console==6.4.4 \
        keras==2.4.3 \
        nbconvert==6.5.4 \
        nltk==3.7 \
        pandas==1.3.5 \
        scikit-learn==1.0.2 \
        scipy==1.4.1 \
        tensorflow==2.3.0 \
        tensorflow-estimator==2.3.0 \
    && conda clean -ya
