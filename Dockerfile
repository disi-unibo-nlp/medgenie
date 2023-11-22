FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
LABEL maintainer="disi-unibo-nlp"

# Zero interaction (default answers to all questions)
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /genRead

# Install general-purpose dependencies
RUN apt-get update -y && \
    apt-get install -y curl \
    git \
    bash \
    nano \
    wget \
    python3.8 \
    python3-pip && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install wrapt --upgrade --ignore-installed
RUN pip install gdown

# Install PyTorch and related packages (part 1)
RUN pip install torch torchvision -f https://download.pytorch.org/whl/cu118/torch_stable.html

# Install other Python packages
RUN pip3 install --upgrade transformers==4.35.0
RUN pip3 install --upgrade peft==0.5.0
RUN pip3 install --upgrade datasets==2.14.5
RUN pip3 install --upgrade wandb==0.15.7
RUN pip3 install --upgrade tokenizers==0.13.3
RUN pip3 install --upgrade tqdm==4.63.1
RUN pip3 install --upgrade nltk==3.7
RUN pip3 install --upgrade scipy==1.10.1
RUN pip3 install --upgrade huggingface_hub==0.18.0

# required for flash attention
RUN pip3 install -q accelerate optimum
RUN pip3 install packaging
RUN pip3 install ninja
RUN pip3 install flash-attn --no-build-isolation
RUN pip3 install bitsandbytes

RUN pip3 install vllm

# Back to default frontend
ENV DEBIAN_FRONTEND=dialog