FROM pytorch/pytorch:latest

# Set working directory
WORKDIR /workspace

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install --no-cache-dir \
    numpy==1.23.3 \
    opencv-python>=4.1.2 \
    PyYAML>=5.3.1 \
    scipy>=1.4.1 \
    tqdm>=4.41.0 \
    addict>=2.4.0 \
    tensorboard>=2.7.0 \
    pycocotools>=2.0 \
    onnx>=1.10.0 \
    onnx-simplifier>=0.3.6 \
    thop \
    dvc \
    dvc-s3 \
    boto3

CMD tail -f /dev/null
