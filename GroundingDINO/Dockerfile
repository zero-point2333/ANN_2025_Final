FROM python:3.8

# 设置环境变量
ENV WORKDIR=/workspace

# 安装系统依赖
RUN apt-get -y update

# 设置工作目录
WORKDIR $WORKDIR

# 复制本地GroundingDINO项目到容器
COPY . .

# 安装Python依赖
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .

# 设置Python路径
ENV PYTHONPATH="${PYTHONPATH}:$WORKDIR"

RUN modelscope download --model "google-bert/bert-base-uncased"

# 运行测试
CMD ["python", "local_test.py"]