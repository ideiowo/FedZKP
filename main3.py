import os
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from blockchain.node import Node  # 從 blockchain 資料夾中引入 Node 類
from blockchain.blockchain import Blockchain  # 確保引入 Blockchain 類別
from blockchain.consensus import Consensus  # 引入 Consensus 類別
from utils.data_utils import load_MNIST, load_FMNIST, load_CIFAR10
from models.architecture import DNN, CNN, LeNet5, LeNet5_CIFAR10
from batch import batch_process_gradients, compute_R_MAXS, compute_R_MAXS_from_layer_stats
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
NUM_REPLICAS = 10  # 副本節點數量 (NUM_NODES = NUM_REPLICAS + 2 主節點 + Supervisor)
ROUNDS = 200
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
use_batch_aggregation = True  # 設為 True 使用批次聚合， False 使用一般聚合器聚合
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 創建 NUM_NODES - 1 個資料加載器
client_loaders, client_test_loaders = load_FMNIST(num_clients=NUM_REPLICAS , batch_size=BATCH_SIZE)
# 初始化資料加載器池
data_loader_pool = client_loaders.copy()

# 初始化全局模型
initial_model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(initial_model.parameters(), lr=LEARNING_RATE)

# 初始化區塊鏈實例
blockchain = Blockchain()

# 初始化共識機制
f = (NUM_REPLICAS + 2) // 2  # 根據 BZ-BFT 的容忍計算 f < n/2 (n = NUM_REPLICAS + 2 主節點 + Supervisor)
consensus = Consensus(blockchain=blockchain, f=f)

# 用來儲存每輪的結果
results = []

# 請求階段
consensus.request_phase(
    round_number=1,  # 初始化輪數為1
    aggregator=aggregator,
    device=device,
    initial_model=initial_model,
    criterion=criterion,
    LEARNING_RATE=LEARNING_RATE,
    NUM_REPLICAS=NUM_REPLICAS,
    data_loader_pool=data_loader_pool
)


# 訓練迴圈
for round in range(ROUNDS):
    print(f"Round {round + 1}/{ROUNDS}")

    # Pre-prepare 階段 - 創建或更新副本節點
    replica_nodes = consensus.pre_prepare_phase(
        NUM_REPLICAS=NUM_REPLICAS,
        device=device,
        initial_model=initial_model,
        LEARNING_RATE=LEARNING_RATE,
        criterion=criterion,
        aggregator=aggregator,
        data_loader_pool=data_loader_pool
    )

    # Prepare 階段 - 副本節點進行訓練
    train_function_name = 'train'  # 可更改為 'train_model_exdp' 等其他訓練函數
    
    if use_batch_aggregation:
        processed_gradients, layer_stats = consensus.prepare_phase(
            replica_nodes=replica_nodes,
            train_function_name=train_function_name,
            epochs=EPOCHS,
            use_batch_aggregation=True,
            bit_width=16,
            batch_size=13,
            pad_zero=3,
            RMAXS=R_MAXS
        )

    else:
        client_gradients = consensus.prepare_phase(
            replica_nodes=replica_nodes,
            train_function_name=train_function_name,
            epochs=EPOCHS,
            use_batch_aggregation=False
        )

    # 主節點進行 Pre-commit 階段的聚合
    primary_node = consensus.primary_node
    if use_batch_aggregation:
        # 更新 R_MAXS 值，供下一輪使用
        R_MAXS = compute_R_MAXS_from_layer_stats(layer_stats, bit_width=16)
    
    # 調用 pre_commit_phase，執行聚合並生成聚合結果
    proof_file_path, public_file_path  = consensus.pre_commit_phase(
        primary_node=primary_node,
        processed_gradients=processed_gradients if use_batch_aggregation else None,
        client_gradients=client_gradients if not use_batch_aggregation else None,
        use_batch_aggregation=use_batch_aggregation,
        R_MAXS=R_MAXS,
        round=round,
        snarkjs_path=snarkjs_path,
        wasm_path=wasm_path,
        zkey_path=zkey_path,
        LEARNING_RATE=LEARNING_RATE,
        bit_width=16,
        batch_size=13,
        pad_zero=3
    )
   
    # Commit 階段 - 副本節點驗證主節點的證明
    consensus_result, verified_nodes, failed_nodes = consensus.commit_phase(
        replica_nodes=replica_nodes,
        snarkjs_path=snarkjs_path,
        verification_key_path=verification_key_path,
        public_file_path=public_file_path,
        proof_file_path=proof_file_path
    )

    if consensus_result:
        print("The consensus was successful. Proceeding to the next round.")

        aggregated_gradients_hash = hashlib.sha256(str(processed_gradients).encode()).hexdigest()
        verification_proof_hash = hashlib.sha256(open(proof_file_path, 'rb').read()).hexdigest()

        blockchain.add_block(
            aggregated_gradients_hash=aggregated_gradients_hash,
            verification_proof_hash=verification_proof_hash,
            consensus_votes={
                "verified_nodes": verified_nodes,
                "failed_nodes": failed_nodes
            }
        )

    else:
        print("The consensus failed. Taking corrective action.")



    # 將更新後的全局模型分發給每個副本節點
    global_model = primary_node.get_global_model()
    for node in replica_nodes:
        node.update_model(global_model.state_dict())

    # 主節點輪換
    total_nodes = len(consensus.nodes)  # 包含 primary 和 Supervisor
    current_primary_id = rotate_primary(round, total_nodes)

    # 設置當前主節點
    consensus.primary_node = consensus.nodes[current_primary_id]
    
    primary_node = consensus.primary_node
    primary_node.become_primary()
    print(f"Node {primary_node.node_id} is now the primary node.")

    # 如果主節點持有資料加載器，放回資料加載器池
    if primary_node.data_loader is not None:
        data_loader_pool.append(primary_node.data_loader)
        primary_node.data_loader = None
    else:
        print(f"Node {primary_node.node_id} did not have a data loader to release.")

    # 處理副本節點（不包含 Supervisor 和 primary node）
    for node in replica_nodes:
        node.become_replica()
        # 如果該節點沒有資料加載器，從資料加載器池中獲取
        if node.data_loader is None:
            if data_loader_pool:
                node.data_loader = data_loader_pool.pop(0)
            else:
                print(f"警告：無法為節點 {node.node_id} 分配資料加載器。")

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

    # 標記輪次結束
    consensus.round_active = False

# 設置結果保存的路徑
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 在所有訓練迴圈結束後，將結果寫入文件
results_path = os.path.join(output_dir, 'FMNIST_batch_dp20.csv')
with open(results_path, 'w') as f:
    f.write('Round,Accuracy,Average Loss\n')
    for round_num, accuracy, average_loss in results:
        f.write(f"{round_num},{accuracy:.2f},{average_loss:.4f}\n")