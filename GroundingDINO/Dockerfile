FROM jittor/jittor-cuda:11.1-16.04
ARG DEBIAN_FRONTEND=noninteractive

# 设置环境变量
ENV CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1 \
    WORKDIR=/workspace

# 安装系统依赖
RUN apt-get -y update && apt-get install -y --no-install-recommends \
        wget \
        build-essential \
        curl \
        ca-certificates \
        libgl1-mesa-glx \
        libglib2.0-0 \
        python3-opencv && \
    rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR $WORKDIR

# 复制本地GroundingDINO项目到容器
COPY GroundingDINO/ ./GroundingDINO/

# 安装Python依赖
RUN cd GroundingDINO && \
    if [ -f requirements.txt ]; then \
        pip install -r requirements.txt; \
    fi

# 安装GroundingDINO包
RUN cd GroundingDINO && \
    python setup.py develop || \
    pip install -e . || \
    python -m pip install .

# 复制测试文件
COPY local_test.py .

# 设置Python路径
ENV PYTHONPATH="${PYTHONPATH}:$WORKDIR:$WORKDIR/GroundingDINO"

# Copy local ModelScope cache (prepare LLM/bert-base/uncased beforehand if needed).
COPY ./LLM/bert-base/uncased /root/.cache/modelscope/hub/models/google-bert/bert-base-uncased

# 运行测试
CMD ["python", "local_test.py"]
