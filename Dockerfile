FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.2.2-devel-ubuntu20.04
ENV PATH=/opt/conda/bin:$PATH
WORKDIR /opt/app

# 更换 Ubuntu 软件源为清华源
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

RUN apt-get update --fix-missing && \
    apt-get install -y wget git && \
    apt-get clean

# 使用清华源下载 Miniconda
RUN wget --quiet https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh 
RUN /bin/bash ~/miniconda.sh -b -p /opt/conda 

# 配置 conda 使用清华源
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2 && \
    conda config --set show_channel_urls yes

RUN echo "source activate base" > ~/.bashrc
RUN conda install -y python=3.9

RUN conda install pytorch=2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# 配置 pip 使用清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install transformers==4.38.0 accelerate==0.27.2 datasets==2.17.1 deepspeed==0.13.2 sentencepiece wandb "numpy<2.0"
COPY flash_attn-2.3.4+cu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl /opt/app
RUN pip install flash_attn-2.3.4+cu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
CMD ["bash"]