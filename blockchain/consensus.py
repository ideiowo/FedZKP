# consensus.py

from datetime import datetime
import torch
from models.architecture import LeNet5  # 確保正確的導入路徑
from blockchain.node import Node  # 確保正確的導入路徑
from blockchain.blockchain import Blockchain  # 確保正確的導入路徑

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

    # 後續的共識流程方法將在未來實現
