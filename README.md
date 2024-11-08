# FedZKP Project

## 專案描述
這個專案實現了聯邦學習中的 FedAvg 算法，並結合了零知識證明（Zero-Knowledge Proof）技術，允許多個客戶端協同訓練一個共享的機器學習模型，而無需將數據集中到一個位置，同時確保數據隱私和安全性。

## 安裝指南

首先克隆此專案到本地，然後安裝必要的依賴。

### 使用 Docker

#### 構建 Docker 映像

在專案根目錄中運行以下命令來構建 Docker 映像：

```bash
docker build -t fedzkp-app .
```

#### 運行 Docker 容器

構建完成後，運行以下命令來啟動 Docker 容器：

```bash
docker run --rm -it fedzkp-app
```

## 文件結構

- `main.py` - 主程序文件，用於執行 FedAvg 訓練流程。
- `requirements.txt` - 包含所有依賴的清單。
- `.gitignore` - 指定 git 要忽略的文件格式。
- `README.md` - 專案的 README 文件。

### 目錄結構

```
FedZKP/
│  .gitignore
│  main.py
│  README.md
│  requirements.txt
│
├─client/
│  │  client.py
│  │  __init__.py
│
├─data/
│  ├─FashionMNIST/
│  ├─MNIST/
│
├─models/
│  │  architecture.py
│  │  __init__.py
│
├─output/
│
├─server/
│  │  server.py
│  │  __init__.py
│
├─utils/
│  │  data_utils.py
│  │  __init__.py
│
└─zkBFT/
    │  block.py
    │  blockchain.py
    │  __init__.py
    │
└─ZKP/
    │  Aggregate.circom
    │  Aggregate.r1cs
    │  Aggregate.sym
    │  Aggregate_0000.zkey
    │  pot12_0000.ptau
    │  pot12_final.ptau
    │  proof.json
    │  public.json
    │  verification_key.json
    │  witness.wtns
    │  Aggregate_js/
    │      Aggregate.wasm
    │      generate_witness.js
    │      witness_calculator.js
```

### 使用方法

執行以下命令來啟動訓練流程：

```bash
python main.py
```

### 注意事項

1. 請確保您已安裝所有必要的依賴項。
2. 如果使用 Docker，請確保 Docker 已啟動並正常運行。
3. 執行 `main.py` 前，請確認所有配置和路徑均已正確設置。


## 授權協議

本專案基於 MIT 許可協議進行分發和使用。更多信息請參閱以下內容。

```
MIT License

Copyright (c) [2024] [NYCU ideiowo]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
