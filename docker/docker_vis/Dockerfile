FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel
LABEL maintainer="nishimura"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsm6 \
        libxext6 \
        libgtk2.0-dev \
	libgl1-mesa-glx \
	git \
	curl \ 
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        opencv-python \
	#opencv-contrib-python \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        keras_applications \
        keras_preprocessing \
        matplotlib \
        pandas \
        scipy \
        sklearn \
	torchvision \
	pulp \
	opencv-python \
	scikit-image \
	tqdm \
    	visdom \
        && \
	python -m ipykernel.kernelspec

COPY jupyter_notebook_config.py /root/.jupyter/

RUN git clone https://github.com/facebookresearch/visdom.git /root/visdom
RUN cd /root/visdom && \
    git checkout $(cat /root/commitish)
LABEL io.k8s.description="Visdom server" \
      io.k8s.display-name="Visdom server"

ADD ./commitish /root/
ADD ./visdom_connection_test.py /root/

RUN cd /root/visdom && \
    git checkout $(cat /root/commitish)

LABEL io.k8s.description="Visdom server" \
      io.k8s.display-name="Visdom server"

ENV HOSTNAME='localhost'
ENV PORT=8097
ENV ENV_PATH="~/.visdom/"
ENV LOGGING_LEVEL='INFO'
ENV READONLY="True"
ENV ENABLE_LOGIN="False"
ENV FORCE_NEW_COOKIE="False"
ENV BASE_URL="/"

RUN cd /root/visdom && pip install .

EXPOSE $PORT

COPY jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /

CMD python -m visdom.server \
    --hostname ${HOSTNAME} \
    -port ${PORT} \
    -base_url ${BASE_URL} \
    -env_path ${ENV_PATH} \
    -logging_level ${LOGGING_LEVEL} \
    `if [ "x$READONLY" = "xTrue" ];then echo "-readonly";fi` \
    `if [ "x$FORCE_NEW_COOKIE" = "xTrue" ];then echo "-force_new_cookie";fi` \
    `if [ "x$ENABLE_LOGIN" = "xTrue" ];then echo "-enable_login";fi`
