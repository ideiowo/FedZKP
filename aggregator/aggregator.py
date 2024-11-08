import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering

class FedAvg:
    def __call__(self, client_gradients):
        if not client_gradients:
            print("警告：client_gradients 列表是空的！")
            return None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_clients = len(client_gradients)

        # 初始化聚合梯度字典
        aggregated_gradients = {name: torch.zeros_like(gradient).to(device) for name, gradient in client_gradients[0].items()}

        # 累加所有客戶端的梯度
        for name in client_gradients[0]:
            # 累加所有客户端的梯度
            aggregated_gradients[name] = sum(gradients[name].to(device) for gradients in client_gradients) / num_clients

        return aggregated_gradients

class ClippedClustering:
    def __init__(self, max_tau=1e5):
        self.tau = max_tau
        self.l2norm_his = []

    def __call__(self, client_gradients):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_clients = len(client_gradients)
        if not client_gradients:
            print("警告：client_gradients 列表是空的！")
            return None

        # 剪裁更新向量
        updates = self._clip_updates(client_gradients, device)

        # 計算兩兩餘弦相似度
        dis_max = self._compute_cosine_similarity(updates, device)

        # 聚類更新向量
        selected_idxs = self._cluster_updates(dis_max)

        # 計算最終聚合值
        aggregated_gradients = self._compute_values(selected_idxs, updates)

        return aggregated_gradients

    def _clip_updates(self, client_gradients, device):
        l2norms = [self._calculate_norm(gradient) for gradient in client_gradients]
        self.l2norm_his.extend(l2norms)
        threshold = np.median(self.l2norm_his)
        threshold = min(threshold, self.tau)

        # 打印歷史範數、範數的中位數和剪裁閾值
        #print(f"歷史範數: {self.l2norm_his}")
        print(f"範數的中位數: {threshold}")
        print(f"剪裁閾值: {threshold}")

        clipped_updates = []
        for gradients in client_gradients:
            l2norm = self._calculate_norm(gradients)
            scale = min(1, threshold / l2norm)
            clipped_gradients = {name: gradient.to(device) * scale for name, gradient in gradients.items()}
            clipped_updates.append(clipped_gradients)

        return clipped_updates

    def _calculate_norm(self, gradients):
        return torch.sqrt(sum(torch.sum(gradient ** 2) for gradient in gradients.values())).item()

    def _compute_cosine_similarity(self, updates, device):
        num = len(updates)
        dis_max = np.zeros((num, num))
        for i in range(num):
            for j in range(i + 1, num):
                dot_product = sum(torch.sum(updates[i][name] * updates[j][name]) for name in updates[i])
                norm_i = torch.sqrt(sum(torch.sum(updates[i][name] ** 2) for name in updates[i])).item()
                norm_j = torch.sqrt(sum(torch.sum(updates[j][name] ** 2) for name in updates[j])).item()
                dis_max[i, j] = 1 - dot_product / (norm_i * norm_j)
                dis_max[j, i] = dis_max[i, j]
        return dis_max

    def _cluster_updates(self, dis_max):
        clustering = AgglomerativeClustering(
            metric='precomputed', linkage='average', n_clusters=2
        )
        clustering.fit(dis_max)

        labels = clustering.labels_
        cluster_0 = [i for i, label in enumerate(labels) if label == 0]
        cluster_1 = [i for i, label in enumerate(labels) if label == 1]

        if len(cluster_0) >= len(cluster_1):
            selected_idxs = cluster_0
        else:
            selected_idxs = cluster_1

        # 打印最終選擇的更新索引
        print(f"最終選擇的更新索引 (selected_idxs): {selected_idxs}")

        return selected_idxs

    def _compute_values(self, selected_idxs, updates):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        aggregated_gradients = {name: torch.zeros_like(gradient).to(device) for name, gradient in updates[0].items()}
        for idx in selected_idxs:
            for name, gradient in updates[idx].items():
                aggregated_gradients[name] += gradient / len(selected_idxs)
        return aggregated_gradients