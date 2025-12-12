# ------------------------------------------------------------
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# SYSTEM DEPENDENCIES
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# ------------------------------------------------------------
# PROJECT SETUP
# ------------------------------------------------------------
WORKDIR /code

# Copy requirements first for better cache
COPY requirements.txt /code/

# ------------------------------------------------------------
# INSTALL LIBRARIES
# ------------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /code/

# ------------------------------------------------------------
# EXECUTION
# Pipeline đọc /code/private_test.json và xuất submission.csv
# ------------------------------------------------------------
CMD ["bash", "inference.sh"]
