# BASE IMAGE
# Lưu ý: Sử dụng đúng phiên bản CUDA 12.2 để khớp với Server BTC
# ------------------------------------------------------------
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# SYSTEM DEPENDENCIES
# Cài đặt Python, Pip và các gói hệ thống cần thiết
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Link python3 thành python
RUN ln -s /usr/bin/python3 /usr/bin/python

# ------------------------------------------------------------
# PROJECT SETUP
# ------------------------------------------------------------
WORKDIR /code

# Copy requirements first for better cache
COPY requirements.txt /code/

# ------------------------------------------------------------
# INSTALL LIBRARIES
# ------------------------------------------------------------
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /code/

# ------------------------------------------------------------
# EXECUTION
# Pipeline đọc /code/private_test.json và xuất submission.csv
# ------------------------------------------------------------
CMD ["bash", "inference.sh"]
