import os
import torch
import torch.nn as nn
import torch.optim as optim
from blockchain.node import Node  # 從 blockchain 資料夾中引入 Node 類
from blockchain.blockchain import Blockchain  # 確保引入 Blockchain 類別
from blockchain.consensus import Consensus  # 引入 Consensus 類別
from utils.data_utils import load_MNIST, load_FMNIST, load_CIFAR10
from models.architecture import DNN, CNN, LeNet5, LeNet5_CIFAR10, SqueezeNet_CIFAR10, ResNet20, CustomCNN
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

def rotate_primary(round_number, total_nodes):
    return 0

# 聯邦學習設定參數
NUM_NODES = 11  # 節點數量，包括主節點和副本節點
ROUNDS = 125
EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.001
R_MAXS = None
wasm_path = "ZKP_LeNet5/Aggregate_js/Aggregate.wasm"
zkey_path = "ZKP_LeNet5/Aggregate_0000.zkey"
snarkjs_path = get_snarkjs_path()
project_base_path = "./gradients"
verification_key_path = "ZKP_LeNet5/verification_key.json"
aggregator = FedAvg()
# 設定聚合方式
use_batch_aggregation = False  # 設為 True 使用批次聚合， False 使用一般聚合器聚合

# 初始化裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 創建 NUM_NODES - 1 個資料加載器
client_loaders, client_test_loaders = load_CIFAR10(num_clients=NUM_NODES - 1, batch_size=BATCH_SIZE)

# 初始化資料加載器池
data_loader_pool = client_loaders.copy()

# 初始化全局模型
initial_model = CustomCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(initial_model.parameters(), lr=LEARNING_RATE)

# 創建節點實例
nodes = []
for i in range(NUM_NODES):
    node_model = CustomCNN().to(device)
    node_model.load_state_dict(initial_model.state_dict())
    node_optimizer = optim.Adam(node_model.parameters(), lr=LEARNING_RATE)
    node_optimizer.load_state_dict(optimizer.state_dict())
    
    if i == 0:
        # 初始化主節點，不分配資料加載器
        role = 'primary'
        data_loader = None

    else:
        # 初始化副本節點，從資料加載器池中分配資料加載器
        role = 'replica'
        data_loader = data_loader_pool.pop(0)
    
    node = Node(node_id=i, data_loader=data_loader, model=node_model, optimizer=node_optimizer, criterion=criterion, aggregator=aggregator, role=role)
    nodes.append(node)




# 用來儲存每輪的結果
results = []

# 訓練迴圈
for round in range(ROUNDS):
    print(f"Round {round + 1}/{ROUNDS}")
    
    # 主節點輪換
    current_primary_id = rotate_primary(round, NUM_NODES)
    
    # 先處理主節點
    primary_node = nodes[current_primary_id]
    primary_node.become_primary()
    print(f"Node {primary_node.node_id} is now the primary node.")
    # 如果主節點持有資料加載器，放回資料加載器池
    if primary_node.data_loader is not None:
        data_loader_pool.append(primary_node.data_loader)
        primary_node.data_loader = None
    else:
        print(f"Node {primary_node.node_id} did not have a data loader to release.")

    # 然後處理副本節點
    for node in nodes:
        if node.node_id != current_primary_id:
            node.become_replica()
            # 如果該節點沒有資料加載器，從資料加載器池中獲取
            if node.data_loader is None:
                if data_loader_pool:
                    node.data_loader = data_loader_pool.pop(0)
                else:
                    print(f"警告：無法為節點 {node.node_id} 分配資料加載器。")
            

    client_gradients = []

    # 副本節點進行訓練
    for node in nodes:
        if node.role == 'replica':
            print(f"Training node {node.node_id}...")
            gradients, train_loss = node.train_model_exdp_C(epochs=EPOCHS, epsilon=1)
            client_gradients.append(gradients)

    # 主節點進行聚合
    primary_node = nodes[current_primary_id]
    
    if use_batch_aggregation:
        # 使用批次聚合
        R_MAXS = compute_R_MAXS(client_gradients, bit_width=16)
        
        processed_gradients = batch_process_gradients(client_gradients, bit_width=16, r_maxs=R_MAXS, batch_size=13, pad_zero=3)
        
        # 主節點生成零知識證明
        public_file_path = primary_node.generate_zero_knowledge_proof(round, processed_gradients, snarkjs_path, wasm_path, zkey_path)
        primary_node.batch_aggregate(processed_gradients, public_file_path, lr=LEARNING_RATE, bit_width=16, r_maxs=R_MAXS, batch_size=13, pad_zero=3)
        
        # 輸出每一層的 R_MAXS 值
        print("各層的 R_MAXS 值：")
        for name, r_max in R_MAXS.items():
            print(f"{name}: R_MAX = {r_max}")
    else:
        # 使用一般聚合器聚合
        primary_node.aggregate_with_aggrator(client_gradients, lr=LEARNING_RATE)

    # 將更新後的全局模型分發給每個副本節點
    global_model = primary_node.get_global_model()
    for node in nodes:
        if node.role == 'replica':
            node.update_model(global_model.state_dict())

    # 驗證全局模型性能（使用主節點的模型）
    primary_node.global_model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in client_test_loaders:
            data, labels = data.to(device), labels.to(device)
            outputs = primary_node.global_model(data)
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
results_path = os.path.join(output_dir, 'CIFAR10_200_modeldp_1C.csv')
with open(results_path, 'w') as f:
    f.write('Round,Accuracy,Average Loss\n')
    for round_num, accuracy, average_loss in results:
        f.write(f"{round_num},{accuracy:.2f},{average_loss:.4f}\n")
