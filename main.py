import os
import torch
import torch.nn as nn
import torch.optim as optim
from client import Client
from server import Server
from utils.data_utils import load_MNIST, load_FMNIST, load_CIFAR10
from models.architecture import DNN,CNN,LeNet5,LeNet5_CIFAR10
from batch import batch_process_gradients, compute_R_MAXS
import numpy as np
import subprocess
from datetime import datetime
import platform
from aggregator import FedAvg, ClippedClustering


def get_snarkjs_path():
    if platform.system() == 'Windows':
        command = 'where snarkjs'
    else:
        command = 'which snarkjs'
    
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
        paths = result.stdout.strip().split('\n')
        
        # 優先選擇 .cmd 文件（主要是 Windows 環境使用）
        if platform.system() == 'Windows':
            for path in paths:
                if path.endswith('.cmd'):
                    return path.strip()
        
        # 返回第一個結果
        return paths[0].strip()
    except subprocess.CalledProcessError as e:
        print(f"Error finding snarkjs path: {e}")
        return None

   
# 聯邦學習設定參數
NUM_CLIENTS = 10
ROUNDS = 50
EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.001
R_MAX = 8
R_MAXS = None
wasm_path = "ZKP_LeNet5/Aggregate_js/Aggregate.wasm"
zkey_path = "ZKP_LeNet5/Aggregate_0000.zkey"
snarkjs_path = get_snarkjs_path()
project_base_path = "./gradients"
verification_key_path = "ZKP_LeNet5/verification_key.json"
aggregator = FedAvg()

# 初始化裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加載數據
client_loaders, client_test_loaders = load_FMNIST(num_clients=NUM_CLIENTS, batch_size=BATCH_SIZE)

# 初始化全局模型
initial_model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(initial_model.parameters(), lr=LEARNING_RATE)

# 創建服務器實例
server = Server(initial_model, aggregator)

# 創建客戶端實例
clients = []
for i in range(NUM_CLIENTS):
    client_model = LeNet5().to(device)
    client_model.load_state_dict(initial_model.state_dict())
    client_optimizer = optim.Adam(client_model.parameters(), lr=LEARNING_RATE)
    client_optimizer.load_state_dict(optimizer.state_dict())
    client = Client(client_id=i, data_loader=client_loaders[i], model=client_model, optimizer=client_optimizer, criterion=criterion)
    clients.append(client)

# 用來儲存每輪的結果
results = []

# 訓練迴圈
for round in range(ROUNDS):
    
    print(f"Round {round + 1}/{ROUNDS}")
    client_gradients = []

    # 訓練每個客戶端並收集梯度
    for client_id, client in enumerate(clients):
        print(f"Training client {client_id + 1}...")
        gradients, train_loss = client.train(epochs=EPOCHS)
        client_gradients.append(gradients)
   
    server.aggregate(client_gradients)
    '''
    # 直接在這裡進行梯度的批處理
    processed_gradients = batch_process_gradients(client_gradients, bit_width=16, r_maxs=R_MAXS, batch_size=13, pad_zero=3)
    
    # 執行處理
    public_file_path = server.generate_zero_knowledge_proof(round, processed_gradients, snarkjs_path, wasm_path, zkey_path)
    server.batch_aggregate(processed_gradients, public_file_path, lr=LEARNING_RATE, bit_width=16, r_maxs=R_MAXS, batch_size=13, pad_zero=3)

    R_MAXS = compute_R_MAXS(client_gradients, bit_width=16)
    # 輸出每一層的 R_MAXS 值
    print("各層的 R_MAXS 值：")
    for name, r_max in R_MAXS.items():
        print(f"{name}: R_MAX = {r_max}")
    # 構建 proof_file_path
    proof_file_path = f"{project_base_path}/round{round + 1}/proof.json"
    '''

    # 將更新後的全局模型分發給每個客戶端
    global_model = server.get_global_model()
    
    for client in clients:
        client.update_model(global_model)

    # 驗證全局模型性能
    global_model.eval()  # 設置模型為評估模式
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in client_test_loaders:  # 使用驗證加載器來測試
            data, labels = data.to(device), labels.to(device)
            outputs = global_model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    average_loss = total_loss / total
    # 儲存每輪的結果
    results.append((round + 1, accuracy, average_loss))
    print(f"驗證結果：準確率 {accuracy}%, 平均損失 {average_loss}")

# 設置結果保存的路徑
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 在所有訓練迴圈結束後，將結果寫入文件
results_path = os.path.join(output_dir, 'FMNIST_50_renew.csv')
with open(results_path, 'w') as f:
    f.write('Round,Accuracy,Average Loss\n')
    for round, accuracy, average_loss in results:
        f.write(f"{round},{accuracy:.2f},{average_loss:.4f}\n")