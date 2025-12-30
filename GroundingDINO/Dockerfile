# 第一阶段：安装从 Modelscope 获取的模型，以及权重
FROM python:3.14 AS build

RUN pip install --upgrade pip
RUN pip install modelscope

RUN modelscope download --model google-bert/bert-base-uncased --local_dir /modelscope
RUN wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# 第二阶段：
FROM python:3.8

# 设置环境变量
ENV WORKDIR=/workspace

# CUDA设置
RUN rm -rf /root/.cache/jittor/jt1.3.10

ENV NVCC_PATH=/home/cslabuser/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux/bin/nvcc \
    nvcc_path=/home/cslabuser/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux/bin/nvcc \
    PATH=/home/cslabuser/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux/bin:$PATH \
    DISABLE_MULTIPROCESSING=1 \
    JT_USE_CUDA=1 \
    CUDA_VISIBLE_DEVICES=0

# 安装系统依赖
RUN apt-get -y update
RUN apt-get -y install libgl1

# 获取第一阶段下载的模型和权重
COPY --from=build /modelscope /modelscope/hub/models/google-bert/bert-base-uncased
COPY --from=build /groundingdino_swint_ogc.pth $WORKDIR/weights/groundingdino_swint_ogc.pth

# 设置工作目录
WORKDIR $WORKDIR

# 复制本地GroundingDINO项目到容器
COPY . .


# 安装Python依赖
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .

# 运行测试
CMD ["python", "local_test.py"]
