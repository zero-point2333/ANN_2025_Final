# 第一阶段：安装 Modelscope
FROM python:3.14 AS build

RUN pip install --upgrade pip
RUN pip install modelscope

RUN modelscope download --model google-bert/bert-base-uncased --local_dir /modelscope

# 第二阶段：
FROM python:3.8

# 设置环境变量
ENV WORKDIR=/workspace

# 安装系统依赖
RUN apt-get -y update
RUN apt-get -y install libgl1

# 获取第一阶段下载的模型
COPY --from=build /modelscope /modelscope/google-bert/bert-base-uncased

# 设置工作目录
WORKDIR $WORKDIR

# 复制本地GroundingDINO项目到容器
COPY . .

# 安装Python依赖
RUN pip install --upgrade pip
RUN pip install -e .

# 运行测试
CMD ["python", "local_test.py"]