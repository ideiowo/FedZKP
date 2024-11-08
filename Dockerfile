# 使用官方的 Python 基礎映像
FROM python:3.9-slim

# 更新並安裝必要的工具
RUN apt-get update && \
    apt-get install -y curl git build-essential

# 安裝 Rust
RUN curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 安裝 Node.js 和 npm
RUN apt-get install -y nodejs npm

# 安裝 snarkjs
RUN npm install -g snarkjs

# 克隆 circom 存儲庫並安裝 circom
RUN git clone https://github.com/iden3/circom.git && \
    cd circom && \
    cargo build --release && \
    cd .. && \
    cargo install --path circom/circom

# 修改 ffjavascript 模塊的主文件
RUN sed -i '/const nPoints = Math.floor(buffBases.byteLength \/ sGIn);/a if (nPoints == 0) return G.zero;' $(npm root -g)/snarkjs/node_modules/ffjavascript/build/main.cjs


# 設置工作目錄
WORKDIR /app

# 複製當前目錄的所有內容到容器的 /app 目錄
COPY . /app

# 安裝必要的 Python 包
RUN pip install --no-cache-dir -r requirements.txt

# 設置環境變量
ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/app/tools:/root/.cargo/bin:/usr/local/bin"

# 指定執行命令
CMD ["python", "main.py"]
