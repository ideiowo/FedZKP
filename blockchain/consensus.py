# consensus.py

from datetime import datetime
import torch
from models.architecture import LeNet5  # 確保正確的導入路徑
from blockchain.node import Node  # 確保正確的導入路徑
from blockchain.blockchain import Blockchain  # 確保正確的導入路徑
import torch.optim as optim
from batch import batch_process_gradients
import numpy as np

class Consensus:
    def __init__(self, blockchain, f):
        """
        初始化 Consensus 類別。

        :param blockchain: Blockchain 類別的實例
        :param f: 系統容忍的拜占庭節點數量（f < n/2）
        """
        self.blockchain = blockchain
        self.nodes = []  # 所有節點的列表，包括 Supervisor 和主節點
        self.f = f  # 容忍的拜占庭節點數量
        self.current_view = 0
        self.sequence_number = 0
        self.votes = {}  # 存儲投票結果，例如 {'node_id': vote}
        self.current_round = 0  # 當前訓練輪數
        self.primary_node = None  # 當前主節點
        self.supervisor_node = None  # Supervisor 節點
        self.round_active = False  # 當前輪次是否活躍
        self.replicas_initialized = False  # 標記副本節點是否已經創建

    def request_phase(self, round_number, aggregator, device, initial_model, criterion, LEARNING_RATE, NUM_REPLICAS, data_loader_pool):
        """
        請求階段：客戶端請求開始一輪新的聯邦學習訓練。

        :param round_number: 當前訓練輪數
        :param aggregator: 聚合器實例
        :param device: 訓練設備（CPU 或 GPU）
        :param initial_model: 初始全局模型
        :param criterion: 損失函數
        :param LEARNING_RATE: 學習率
        :param NUM_REPLICAS: 副本節點數量
        :param data_loader_pool: 資料加載器池
        :return: bool，是否成功啟動請求階段
        """
        if self.round_active:
            print("當前已有活躍的輪次，無法啟動新的輪次。")
            return False

        self.current_round = round_number
        self.round_active = True

        print(f"開始請求階段，輪數：{self.current_round}")

        # 初始化全局模型
        self.global_model = initial_model.to(device)
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=LEARNING_RATE)

        # 創建主節點
        primary_node_model = LeNet5().to(device)
        primary_node_model.load_state_dict(initial_model.state_dict())
        primary_node_optimizer = torch.optim.Adam(primary_node_model.parameters(), lr=LEARNING_RATE)
        primary_node_optimizer.load_state_dict(self.optimizer.state_dict())

        primary_node = Node(
            node_id=0,
            data_loader=None,
            model=primary_node_model,
            optimizer=primary_node_optimizer,
            criterion=self.criterion,
            aggregator=aggregator,
            role='primary'
        )
        self.primary_node = primary_node
        self.nodes.append(primary_node)
        primary_node.become_primary()
        print(f"主節點 {primary_node.node_id} 已設置為 primary node.")

        # 創建 Supervisor 節點
        supervisor_node_model = LeNet5().to(device)
        supervisor_node_model.load_state_dict(initial_model.state_dict())
        supervisor_node_optimizer = torch.optim.Adam(supervisor_node_model.parameters(), lr=LEARNING_RATE)
        supervisor_node_optimizer.load_state_dict(self.optimizer.state_dict())

        supervisor_node = Node(
            node_id=1,
            data_loader=None,
            model=supervisor_node_model,
            optimizer=supervisor_node_optimizer,
            criterion=self.criterion,
            aggregator=aggregator,
            role='supervisor'
        )
        self.supervisor_node = supervisor_node
        self.nodes.append(supervisor_node)
        print(f"Supervisor 節點 {supervisor_node.node_id} 已設置。")

        return True

    def pre_prepare_phase(self, NUM_REPLICAS, device, initial_model, LEARNING_RATE, criterion, aggregator, data_loader_pool):
        """
        Pre-prepare 階段：僅在初次調用時創建副本節點，後續調用僅更新模型。

        :param NUM_REPLICAS: 副本節點數量
        :param device: 訓練設備（CPU 或 GPU）
        :param initial_model: 初始全局模型
        :param LEARNING_RATE: 學習率
        :param criterion: 損失函數
        :param aggregator: 聚合器實例
        :param data_loader_pool: 資料加載器池
        :return: List of replica nodes
        """
        if not self.replicas_initialized:
            # 初次調用，創建副本節點
            self.replica_nodes = []
            for i in range(NUM_REPLICAS):
                node_model = LeNet5().to(device)
                node_model.load_state_dict(initial_model.state_dict())
                node_optimizer = optim.Adam(node_model.parameters(), lr=LEARNING_RATE)
                node_optimizer.load_state_dict(self.primary_node.optimizer.state_dict())
                
                role = 'replica'
                data_loader = data_loader_pool.pop(0) if data_loader_pool else None
                
                node = Node(
                    node_id=i+2,  # node_id 從2開始，因為主節點和 Supervisor 會在 request_phase 中創建
                    data_loader=data_loader,
                    model=node_model,
                    optimizer=node_optimizer,
                    criterion=criterion,
                    aggregator=aggregator,
                    role=role
                )
                self.replica_nodes.append(node)
            
            self.replicas_initialized = True  # 設置已初始化標記
            print(f"已成功創建 {NUM_REPLICAS} 個副本節點。")
        else:
            # 後續調用，更新副本節點的模型
            global_model = self.primary_node.get_global_model()
            for node in self.replica_nodes:
                node.update_model(global_model.state_dict())
            print("已成功更新副本節點的模型。")

        return self.replica_nodes
  
    def prepare_phase(self, replica_nodes, train_function_name, epochs, use_batch_aggregation, bit_width=16, batch_size=13, pad_zero=3, RMAXS = None):
        """
        Prepare 階段：讓副本節點使用指定的訓練函數進行本地訓練，收集梯度，並根據需要進行批次處理。

        :param replica_nodes: 副本節點列表
        :param train_function_name: 使用的訓練函數名稱，例如 'train_exdp' 或 'train_model_exdp'
        :param epochs: 訓練的輪數
        :param use_batch_aggregation: 是否使用批次聚合
        :param bit_width: 批次處理的位元寬度
        :param batch_size: 批次大小
        :param pad_zero: 填充零的數量
        :return: 若使用批次處理，返回 (processed_gradients, layer_stats)，否則返回 client_gradients
        """
        client_gradients = []

        # 副本節點訓練
        for node in replica_nodes:
            print(f"Training node {node.node_id} with {train_function_name}...")
            
            # 獲取訓練函數
            train_function = getattr(node, train_function_name, None)
            if train_function is None:
                raise ValueError(f"Node {node.node_id} does not have a method named {train_function_name}")

            # 執行訓練並獲取梯度
            gradients, train_loss = train_function(epochs=epochs)
            client_gradients.append(gradients)

        # 判斷是否需要批次聚合
        if use_batch_aggregation:
            print("使用批次處理梯度。")
            
            # 初始化每層的最大值、最小值和尺寸資訊
            layer_names = client_gradients[0].keys()
            layer_stats = {name: {'max_values': [], 'min_values': [], 'size': 0} for name in layer_names}
            num_clients = len(client_gradients)

            # 收集每層的最大值、最小值和尺寸資訊
            for gradients in client_gradients:
                for name, gradient in gradients.items():
                    if gradient is None:
                        continue
                    gradient_flat = gradient.flatten().detach().cpu().numpy()
                    if not np.all(np.isfinite(gradient_flat)):
                        continue
                    layer_stats[name]['max_values'].append(np.max(gradient_flat))
                    layer_stats[name]['min_values'].append(np.min(gradient_flat))
                    if layer_stats[name]['size'] == 0:
                        layer_stats[name]['size'] = gradient_flat.size * num_clients

            # 將收集到的資訊用於批次處理
            processed_gradients = batch_process_gradients(
                client_gradients=client_gradients,
                bit_width=bit_width,
                r_maxs=RMAXS,
                batch_size=batch_size,
                pad_zero=pad_zero
            )

            # 返回處理後的梯度和層的統計資訊
            return processed_gradients, layer_stats
        else:
            # 若未使用批次處理，則返回原始梯度
            return client_gradients
    
    def pre_commit_phase(self, primary_node, processed_gradients, client_gradients, use_batch_aggregation, R_MAXS, round, snarkjs_path, wasm_path, zkey_path, LEARNING_RATE, bit_width=16, batch_size=13, pad_zero=3):
        """
        Pre-commit 階段：主節點進行聚合並生成聚合結果。

        :param primary_node: 主節點實例
        :param processed_gradients: 批次處理後的梯度（若使用批次聚合）
        :param client_gradients: 客戶端的原始梯度列表（若不使用批次聚合）
        :param use_batch_aggregation: 是否使用批次聚合
        :param R_MAXS: 每層的剪枝閾值
        :param round: 當前輪數
        :param snarkjs_path: snarkjs 的路徑
        :param wasm_path: WASM 文件的路徑
        :param zkey_path: zkey 文件的路徑
        :param LEARNING_RATE: 學習率
        :param bit_width: 批次處理的位元寬度
        :param batch_size: 批次大小
        :param pad_zero: 填充零的數量
        :return: 聚合結果或相關資訊
        """
        if use_batch_aggregation:
            # 使用批次聚合並生成零知識證明
            print("主節點使用批次聚合並生成零知識證明。")
            proof_file_path, public_file_path = primary_node.generate_zero_knowledge_proof(
                round=round,
                processed_gradients=processed_gradients,
                snarkjs_path=snarkjs_path,
                wasm_path=wasm_path,
                zkey_path=zkey_path
            )
            primary_node.batch_aggregate(
                processed_gradients=processed_gradients,
                public_json_path=public_file_path,
                lr=LEARNING_RATE,
                bit_width=bit_width,
                r_maxs=R_MAXS,
                batch_size=batch_size,
                pad_zero=pad_zero
            )
            return proof_file_path, public_file_path   # 返回 ZKP 的文件路徑作為聚合結果
        else:
            # 使用一般聚合器進行聚合
            print("主節點使用一般聚合器進行聚合。")
            primary_node.aggregate_with_aggrator(client_gradients, lr=LEARNING_RATE)
            return None  # 若不生成 ZKP，則返回 None

    def commit_phase(self, replica_nodes, snarkjs_path, verification_key_path, public_file_path, proof_file_path):
        """
        Commit 階段：主節點收集副本節點的驗證結果並產生共識結果。

        :param replica_nodes: 副本節點列表
        :param snarkjs_path: snarkjs 的路徑
        :param verification_key_path: 驗證鍵的路徑
        :param public_file_path: 公共輸出文件的路徑
        :param proof_file_path: 證明文件的路徑
        :return: Tuple (共識結果, 驗證成功的節點列表, 驗證失敗的節點列表)
        """
        verified_nodes = []
        failed_nodes = []

        # 副本節點驗證主節點的證明
        for replica in replica_nodes:
            node_id, is_verified, message = replica.verify_proof(
                snarkjs_path=snarkjs_path,
                verification_key_path=verification_key_path,
                public_file_path=public_file_path,
                proof_file_path=proof_file_path
            )
            if is_verified:
                print(f"Node {node_id} successfully verified the proof: {message}")
                verified_nodes.append(node_id)
            else:
                print(f"Node {node_id} failed to verify the proof: {message}")
                failed_nodes.append(node_id)

        # 決定共識結果
        if len(verified_nodes) > len(failed_nodes):
            consensus_result = True
            print("Consensus reached: Majority of nodes verified the proof successfully.")
        else:
            consensus_result = False
            print("Consensus failed: Majority of nodes failed to verify the proof.")

        return consensus_result, verified_nodes, failed_nodes


    # 後續的共識流程方法將在未來實現
