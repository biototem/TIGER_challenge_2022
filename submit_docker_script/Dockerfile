FROM ubuntu:20.04

ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV BASE_INSTALL=pkgs
COPY $BASE_INSTALL/ /$BASE_INSTALL

# Install python3.8

RUN : \
    && apt-get update \
    && seq 3 | xargs -i apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && seq 3 | xargs -i apt-get install -y --no-install-recommends python3.8-venv \
    && seq 3 | xargs -i apt-get install libpython3.8-dev -y \
    && apt-get clean \
    && :
    
# Add env to PATH
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install ASAP
RUN : \
    && apt-get update \
    && seq 3 | xargs -i apt install -y libboost-program-options1.71.0 libboost-regex1.71.0 libdcmtk14 \
    && dpkg --install /$BASE_INSTALL/ASAP-2.0-py38-Ubuntu2004.deb || true \
    && apt-get -f install --fix-missing --fix-broken --assume-yes \
    && dpkg -i /$BASE_INSTALL/ASAP-2.0-py38-Ubuntu2004.deb \
    && dpkg -i /$BASE_INSTALL/libpixman-1-0_0.40.0-1build2_amd64.deb \
    && dpkg -i /$BASE_INSTALL/libpixman-1-dev_0.40.0-1build2_amd64.deb \
    && ldconfig -v \
    && apt-get clean \
    && echo "/opt/ASAP/bin" > /venv/lib/python3.8/site-packages/asap.pth \
#    && :

# # Install algorithm
#RUN mkdir -p /home/user/pathology-tiger-algorithm
#COPY setup.py /home/user/pathology-tiger-algorithm
RUN : \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple wheel==0.37.0 \
#    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple /home/user/pathology-tiger-algorithm \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.20.2 \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm==4.62.3 \
#    && rm -r /home/user/pathology-tiger-algorithm \
    && :

# Install python other requirements
RUN : \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pillow \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple /$BASE_INSTALL/torch-1.10.2+cu113-cp38-cp38-linux_x86_64.whl \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple /$BASE_INSTALL/torchvision-0.11.3+cu113-cp38-cp38-linux_x86_64.whl \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple shapely \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple imageio \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-image \
    && :


RUN rm -r $BASE_INSTALL


RUN : \
      && seq 3 | xargs -i apt-get install -y sudo
    
# Make user
RUN  mkdir /home/user \
     && groupadd -r user && useradd -r -g user user \
     && chown user /home/user/ \
     && :
RUN mkdir /output/
RUN chown user /output/
ENV passwd=tigerroot

RUN usermod -aG sudo user \
    && echo "user   ALL=(ALL:ALL) NOPASSWD:ALL" >>/etc/sudoers \
    && chmod u+w /etc/sudoers

RUN mkdir /input \
    && chown user /input


ENV ALGORITHMSCRIPT=tigeralgorithmexample
COPY $ALGORITHMSCRIPT/ /home/user/tigeralgorithmexample/
RUN chmod 777 -R /home/user/tigeralgorithmexample

USER user
WORKDIR /home/user/tigeralgorithmexample


ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute


#CMD ["/bin/bash"]
# Cmd and entrypoint
CMD ["__main__.py"]
ENTRYPOINT ["python"]

# Compute requirements
LABEL processor.cpus="1"
LABEL processor.cpu.capabilities="null"
LABEL processor.memory="15G"
LABEL processor.gpu_count="1"
LABEL processor.gpu.compute_capability="null"
LABEL processor.gpu.memory="11G"
