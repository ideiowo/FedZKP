import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import subprocess
import re
import json
import os
import numpy as np
import csv
from batch import (
    compute_R_MAXS,
    batch_decode_from_big_int,
    dequantize_from_twos_complement,
)
from aggregator import FedAvg  # 假設您有一個aggregator.py文件

# 讀取 public.json 文件
def read_public_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def remove_ansi_escape_sequences(text):
    ansi_escape = re.compile(r'\x1b\[.*?m')
    return ansi_escape.sub('', text)

def calculate_witness(round, json_file_path, round_folder_path, snarkjs_path, wasm_path):
    output_witness_path = f"{round_folder_path}/witness.wtns"
    try:
        result = subprocess.run(
            [snarkjs_path, "wtns", "calculate", wasm_path, json_file_path, output_witness_path],
            check=True, text=True, capture_output=True
        )
        print(f"Successfully created witness for round {round + 1}: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error calculating witness for round {round + 1}: {e.stderr}")
        print(f"Standard Output: {e.stdout}")
        print(f"Return Code: {e.returncode}")
        return False

    return output_witness_path

def create_proof_and_public(round, output_witness_path, round_folder_path, snarkjs_path, zkey_path):
    proof_file_path = os.path.join(round_folder_path, 'proof.json')
    public_file_path = os.path.join(round_folder_path, 'public.json')
    try:
        result = subprocess.run(
            [snarkjs_path, "groth16", "prove", zkey_path, output_witness_path, proof_file_path, public_file_path],
            check=True, text=True, capture_output=True
        )
        print(f"Successfully created proof and public data for round {round + 1}: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating proof for round {round + 1}: {e.stderr}")
        print(f"Command: {e.cmd}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

    return proof_file_path, public_file_path

def save_gradients_to_json(round, processed_gradients, snarkjs_path, wasm_path, zkey_path):
    # 將處理後的梯度轉換為可序列化的格式
    serializable_gradients = flatten_and_convert_to_serializable(processed_gradients)

    # 將處理後的梯度轉換為 JSON 格式並儲存
    gradients_json = json.dumps({"values": serializable_gradients}, indent=4)

    round_folder_path = f"./gradients/round{round + 1}"
    json_file_path = f"{round_folder_path}/batch_gradient{round + 1}.json"

    try:
        os.makedirs(round_folder_path, exist_ok=True)
        with open(json_file_path, 'w') as f:
            f.write(gradients_json)
        print(f"Successfully saved: round{round + 1}.json")
    except Exception as e:
        print(f"Error writing file for round {round + 1}: {e}")
        return False, None

    return json_file_path, round_folder_path

def get_layer_type(name):
    """ 根據參數名稱返回層的類型 """
    if "conv" in name or "features" in name:
        return "Convolutional Layer"
    elif "fc" in name or "classifier" in name:
        return "Fully Connected Layer"
    else:
        return "Other Layer"

def write_layer_stats_to_csv(layer_type, eta, sensitivity, max_min_diff):
    """ 寫入層級統計數據到 CSV 文件 """
    with open('node_3_gradients_data.csv', 'a', newline='') as csvfile:
        # 檢查文件是否是空的，以決定是否寫入標頭
        csvfile.seek(0, 2)  # 移動到文件的末尾
        is_empty = csvfile.tell() == 0

        writer = csv.DictWriter(csvfile, fieldnames=["Layer Type", "Eta", "Sensitivity (1/eta)", "Max-Min Difference"])

        if is_empty:
            writer.writeheader()  # 文件是空的，寫入標頭

        # 寫入當前層的數據
        writer.writerow({
            "Layer Type": layer_type,
            "Eta": eta,
            "Sensitivity (1/eta)": sensitivity,
            "Max-Min Difference": max_min_diff
        })
# 遞歸地將 NumPy 陣列或 PyTorch 張量轉換為列表，並展平成單一列表

def flatten_and_convert_to_serializable(data):
    if isinstance(data, np.ndarray):
        return [str(x) for x in data.flatten().tolist()]
    elif isinstance(data, torch.Tensor):
        return [str(x) for x in data.detach().cpu().numpy().flatten().tolist()]
    elif isinstance(data, dict):
        flat_list = []
        for v in data.values():
            flat_list.extend(flatten_and_convert_to_serializable(v))
        return flat_list
    elif isinstance(data, list):
        flat_list = []
        for elem in data:
            flat_list.extend(flatten_and_convert_to_serializable(elem))
        return flat_list
    else:
        return [str(data)]

class Node:
    def __init__(self, node_id, data_loader=None, model=None, optimizer=None, criterion=None, aggregator=None, role='replica'):
        self.node_id = node_id
        self.data_loader = data_loader      
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device) if model else None
        self.optimizer = optimizer
        self.criterion = criterion
        self.aggregator = aggregator
        self.role = role  # 'primary', 'replica', 'supervisor'
        self.global_model = model.to(self.device) if model else None  # 用於主節點
        self.previous_model_updates = None # 用於差分隱私的訓練

    def train(self, epochs=1):
        if self.role != 'replica':
            print(f"Node {self.node_id} is not a replica node and cannot train.")
            return None, None

        self.model.to(self.device)
        
        # 1. 記錄初始參數
        initial_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(self.data_loader)
            print(f"Node {self.node_id} - Epoch {epoch+1} - Train Loss: {train_loss}")

        # 2. 計算參數差異
        delta_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                delta_params[name] = param.clone().detach() - initial_params[name]

        return delta_params, train_loss

    def train_exdp(self, epochs=1, N=10, epsilon=0.013):
        if self.role != 'replica':
            print(f"Node {self.node_id} is not a replica node and cannot train with differential privacy.")
            return None, None

        self.model.to(self.device)
        
        # 1. 記錄初始參數
        initial_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}

        # 初始化 previous_gradients 為零張量
        previous_gradients = {name: torch.zeros_like(param).to(self.device) for name, param in self.model.named_parameters()}

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            # 重置 previous_gradients 為零張量，因為每個 epoch 被視為新的迭代
            previous_gradients = {name: torch.zeros_like(param).to(self.device) for name, param in self.model.named_parameters()}

            for batch_idx, (data, target) in enumerate(tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                # 初始化擾動後的梯度
                perturbed_gradients = {}

                # 對每個參數進行指數機制的梯度擾動
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        continue

                    # 獲取當前梯度和上一個迭代的梯度
                    current_grad = param.grad.clone().to(self.device)
                    prev_grad = previous_gradients[name]

                    # 計算取樣界限 C
                    C = current_grad - prev_grad

                    # 確定取樣範圍 R
                    R_lower = torch.where(C < 0, current_grad - C.abs(), current_grad)
                    R_upper = torch.where(C < 0, current_grad, current_grad + C)

                    # 生成候選集 S
                    steps = torch.linspace(0, 1, steps=N).to(self.device)
                    steps_shape = [N] + [1]*(current_grad.dim())  # 調整 steps 的形狀以進行廣播
                    steps = steps.view(steps_shape)

                    R_lower_expanded = R_lower.unsqueeze(0)
                    R_upper_expanded = R_upper.unsqueeze(0)

                    # S 的形狀為 (N, *param.shape)
                    S = R_lower_expanded + (R_upper_expanded - R_lower_expanded) * steps

                    # 計算距離 d，這裡使用絕對值差異
                    distances = torch.abs(current_grad.unsqueeze(0) - S)
                    
                    # 計算學習率和敏感度
                    learning_rate = self.optimizer.param_groups[0]['lr']
                    eta = learning_rate
                    
                    # 檢查該參數是否在優化器狀態中並且包含所需的鍵
                    if param in self.optimizer.state and 'exp_avg' in self.optimizer.state[param] and 'exp_avg_sq' in self.optimizer.state[param]:
                        # 計算更新步長
                        m_t = self.optimizer.state[param]['exp_avg']
                        v_t = self.optimizer.state[param]['exp_avg_sq']
                        step_size = self.optimizer.param_groups[0]['lr'] * (m_t / (v_t.sqrt() + self.optimizer.param_groups[0]['eps']))
                        eta = step_size

                    exponent = - (eta * epsilon / 2) * distances

                    # 為了數值穩定性，減去最大值
                    max_exponent = exponent.max(dim=0, keepdim=True)[0]
                    probabilities = torch.exp(exponent - max_exponent)
                    probabilities = probabilities / probabilities.sum(dim=0, keepdim=True)

                    # 將概率展平以便進行抽樣
                    flat_probabilities = probabilities.view(N, -1).transpose(0, 1)  # 形狀為 (num_elements, N)

                    # 對每個元素進行抽樣
                    indices = torch.multinomial(flat_probabilities, num_samples=1).squeeze(1)  # 形狀為 (num_elements,)

                    # 從候選集 S 中選擇擾動後的值
                    S_flat = S.view(N, -1).transpose(0, 1)  # 形狀為 (num_elements, N)
                    perturbed_values = S_flat[torch.arange(S_flat.size(0)), indices]

                    # 恢復擾動梯度的形狀
                    perturbed_grad = perturbed_values.view_as(current_grad)

                    # 保存擾動後的梯度
                    perturbed_gradients[name] = perturbed_grad

                    # 更新參數的梯度為擾動後的梯度
                    param.grad = perturbed_grad

                    # 更新 previous_gradients
                    previous_gradients[name] = current_grad.clone().detach()

                self.optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(self.data_loader)
            print(f"Node {self.node_id} - Epoch {epoch+1} - Train Loss: {train_loss}")

        # 2. 計算參數差異
        delta_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                delta_params[name] = param.clone().detach() - initial_params[name]

        return delta_params, train_loss

    def train_exdp_C(self, epochs=1, N=10, epsilon=0.013):
        if self.role != 'replica':
            print(f"節點 {self.node_id} 不是副本節點，無法進行差分隱私訓練。")
            return None, None

        self.model.to(self.device)
        
        # 1. 記錄初始參數
        initial_params = {name: param.clone().detach().to(self.device) for name, param in self.model.named_parameters()}

        # 初始化 previous_gradients 為零張量
        previous_gradients = {name: torch.zeros_like(param).to(self.device) for name, param in self.model.named_parameters()}

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            # 重置 previous_gradients 為零張量，因為每個 epoch 被視為新的迭代
            previous_gradients = {name: torch.zeros_like(param).to(self.device) for name, param in self.model.named_parameters()}

            for batch_idx, (data, target) in enumerate(tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                # 初始化擾動後的梯度
                perturbed_gradients = {}

                # 對每個參數進行指數機制的梯度擾動
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        continue

                    # 獲取當前梯度和上一個迭代的梯度
                    current_grad = param.grad.clone().detach().to(self.device)
                    prev_grad = previous_gradients[name]

                    # 計算取樣界限 C
                    C = current_grad - prev_grad

                    # 確定取樣範圍 R
                    R_lower = torch.where(C < 0, current_grad + C, current_grad)
                    R_upper = torch.where(C < 0, current_grad, current_grad + C)

                    # 生成候選集 S
                    steps = torch.linspace(0, 1, steps=N).to(self.device)
                    steps_shape = [N] + [1]*(current_grad.dim())  # 調整 steps 的形狀以進行廣播
                    steps = steps.view(steps_shape)

                    R_lower_expanded = R_lower.unsqueeze(0)
                    R_upper_expanded = R_upper.unsqueeze(0)

                    # S 的形狀為 (N, *param.shape)
                    S = R_lower_expanded + (R_upper_expanded - R_lower_expanded) * steps

                    # 計算距離 d，這裡使用絕對值差異
                    distances = torch.abs(current_grad.unsqueeze(0) - S)
                    
                    # 使用 C 的絕對值作為敏感度 delta_g
                    delta_g = torch.abs(C)
                    delta_g = delta_g + 1e-10  # 防止敏感度為零

                    # 將 delta_g 擴展到與 distances 相同的形狀
                    delta_g_expanded = delta_g.unsqueeze(0)

                    # 計算指數項
                    exponent = - (epsilon / (2 * delta_g_expanded)) * distances

                    # 為了數值穩定性，減去最大值
                    max_exponent = exponent.max(dim=0, keepdim=True)[0]
                    probabilities = torch.exp(exponent - max_exponent)
                    probabilities = probabilities / probabilities.sum(dim=0, keepdim=True)

                    # 將概率展平以便進行抽樣
                    flat_probabilities = probabilities.view(N, -1).transpose(0, 1)  # 形狀為 (num_elements, N)

                    # 對每個元素進行抽樣
                    indices = torch.multinomial(flat_probabilities, num_samples=1).squeeze(1)  # 形狀為 (num_elements,)

                    # 從候選集 S 中選擇擾動後的值
                    S_flat = S.view(N, -1).transpose(0, 1)  # 形狀為 (num_elements, N)
                    perturbed_values = S_flat[torch.arange(S_flat.size(0)), indices]

                    # 恢復擾動梯度的形狀
                    perturbed_grad = perturbed_values.view_as(current_grad)

                    # 保存擾動後的梯度
                    perturbed_gradients[name] = perturbed_grad

                    # 更新參數的梯度為擾動後的梯度
                    param.grad = perturbed_grad

                    # 更新 previous_gradients
                    previous_gradients[name] = current_grad.clone().detach()

                # 更新模型參數
                self.optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(self.data_loader)
            print(f"節點 {self.node_id} - Epoch {epoch+1} - 訓練損失: {train_loss}")

        # 2. 計算參數差異
        delta_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                delta_params[name] = param.clone().detach().to(self.device) - initial_params[name]

        return delta_params, train_loss


    def train_exdp_clamp(self, epochs=1, N=10, epsilon=0.013, bit_width=16):
        """
        使用指數機制（ExDP）對梯度進行差分隱私訓練，
        在每個批次中計算梯度的剪裁範圍，並使用該範圍作為敏感度。

        參數：
        - epochs (int): 訓練的輪數。
        - N (int): 每個取樣範圍內的候選數量。
        - epsilon (float): 差分隱私的隱私參數。
        - bit_width (int): 用於計算 R_MAXS 的位元寬度。

        返回：
        - delta_params (dict): 模型參數的變化量。
        - train_loss (float): 訓練損失。
        """
        if self.role != 'replica':
            print(f"Node {self.node_id} is not a replica node and cannot train with differential privacy.")
            return None, None

        self.model.to(self.device)
        
        # 記錄初始參數
        initial_params = {name: param.clone().detach().to(self.device) for name, param in self.model.named_parameters()}

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            # 初始化 previous_gradients 為零張量
            previous_gradients = {name: torch.zeros_like(param).to(self.device) for name, param in self.model.named_parameters()}

            for batch_idx, (data, target) in enumerate(tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                # 初始化擾動後的梯度
                perturbed_gradients = {}

                # 收集當前批次的梯度
                current_gradients = {name: param.grad.clone().detach().to(self.device) for name, param in self.model.named_parameters() if param.grad is not None}

                # 計算當前批次的 R_MAXS
                batch_gradients_list = [current_gradients]
                R_MAXS = compute_R_MAXS(batch_gradients_list, bit_width=bit_width)

                # 對每個參數進行剪裁和指數機制的梯度擾動
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        continue

                    # 獲取當前梯度和上一個迭代的梯度
                    current_grad = current_gradients[name]
                    prev_grad = previous_gradients[name]

                    # 從 R_MAXS 中獲取剪裁範圍
                    max_value = R_MAXS.get(name, None)
                    if max_value is not None:
                        # 對梯度進行剪裁
                        current_grad = torch.clamp(current_grad, min=-max_value, max=max_value)
                    else:
                        print(f"警告：層 {name} 沒有找到對應的 R_MAXS，將跳過此層的剪裁。")
                        continue

                    # 計算取樣界限 C
                    C = current_grad - prev_grad

                    # 確定取樣範圍 R
                    R_lower = torch.where(C < 0, current_grad - torch.abs(C), current_grad)
                    R_upper = torch.where(C < 0, current_grad, current_grad + C)

                    # 生成候選集 S
                    steps = torch.linspace(0, 1, steps=N).to(self.device)
                    steps_shape = [N] + [1]*(current_grad.dim())  # 調整 steps 的形狀以進行廣播
                    steps = steps.view(steps_shape)

                    R_lower_expanded = R_lower.unsqueeze(0)
                    R_upper_expanded = R_upper.unsqueeze(0)

                    # S 的形狀為 (N, *param.shape)
                    S = R_lower_expanded + (R_upper_expanded - R_lower_expanded) * steps

                    # 計算距離 d，這裡使用絕對值差異
                    distances = torch.abs(current_grad.unsqueeze(0) - S)
                    
                    # 使用剪裁範圍作為敏感度 delta_g
                    if max_value is not None and max_value > 0:
                        delta_g = 2 * max_value  # 敏感度為剪裁範圍的兩倍
                       
                    else:
                        print(f"警告：層 {name} 的 max_value 無效，將跳過此層的擾動。")
                        continue

                    # 計算指數項
                    exponent = - (epsilon / (2 * delta_g)) * distances

                    # 為了數值穩定性，減去最大值
                    max_exponent = exponent.max(dim=0, keepdim=True)[0]
                    probabilities = torch.exp(exponent - max_exponent)
                    probabilities = probabilities / probabilities.sum(dim=0, keepdim=True)

                    # 將概率展平以便進行抽樣
                    flat_probabilities = probabilities.view(N, -1).transpose(0, 1)  # 形狀為 (num_elements, N)

                    # 對每個元素進行抽樣
                    indices = torch.multinomial(flat_probabilities, num_samples=1).squeeze(1)  # 形狀為 (num_elements,)

                    # 從候選集 S 中選擇擾動後的值
                    S_flat = S.view(N, -1).transpose(0, 1)  # 形狀為 (num_elements, N)
                    perturbed_values = S_flat[torch.arange(S_flat.size(0)), indices]

                    # 恢復擾動梯度的形狀
                    perturbed_grad = perturbed_values.view_as(current_grad)

                    # 保存擾動後的梯度
                    perturbed_gradients[name] = perturbed_grad

                    # 更新參數的梯度為擾動後的梯度
                    param.grad = perturbed_grad

                    # 更新 previous_gradients
                    previous_gradients[name] = current_grad.clone().detach()

                # 使用擾動後的梯度更新模型參數
                self.optimizer.step()
                total_loss += loss.item()

            # 更新訓練損失
            train_loss = total_loss / len(self.data_loader)
            print(f"Node {self.node_id} - Epoch {epoch+1} - Train Loss after DP: {train_loss}")

        # 計算參數差異
        delta_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                delta_params[name] = param.clone().detach().to(self.device) - initial_params[name]

        return delta_params, train_loss                                                                                     

    def train_clamp(self, epochs=1, bit_width=16):
        """
        訓練模型，對梯度進行剪裁，剪裁範圍使用 compute_R_MAXS 計算得出的值。

        參數：
        - epochs (int): 訓練的輪數。
        - bit_width (int): 用於計算 R_MAXS 的位元寬度。

        返回：
        - delta_params (dict): 模型參數的變化量。
        - train_loss (float): 訓練損失。
        """
        if self.role != 'replica':
            print(f"Node {self.node_id} is not a replica node and cannot train with gradient clipping.")
            return None, None

        self.model.to(self.device)
        
        # 1. 記錄初始參數
        initial_params = {name: param.clone().detach().to(self.device) for name, param in self.model.named_parameters()}

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for batch_idx, (data, target) in enumerate(tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                # 收集當前批次的梯度
                current_gradients = {name: param.grad.clone().detach().to(self.device) for name, param in self.model.named_parameters() if param.grad is not None}

                # 計算當前批次的 R_MAXS
                batch_gradients_list = [current_gradients]
                R_MAXS = compute_R_MAXS(batch_gradients_list, bit_width=bit_width)

                # 對梯度進行剪裁
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        max_value = R_MAXS.get(name, None)
                        if max_value is not None:
                            param.grad.data = torch.clamp(param.grad.data, min=-max_value, max=max_value)
                        else:
                            print(f"警告：層 {name} 沒有找到對應的 R_MAXS，將跳過此層的剪裁。")

                # 更新模型參數
                self.optimizer.step()
                total_loss += loss.item()

            # 計算並輸出當前 epoch 的平均訓練損失
            train_loss = total_loss / len(self.data_loader)
            print(f"Node {self.node_id} - Epoch {epoch+1} - Train Loss: {train_loss}")

        # 2. 計算參數差異
        delta_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                delta_params[name] = param.clone().detach().to(self.device) - initial_params[name]

        return delta_params, train_loss

    def train_exdp_record(self, epochs=1, N=10, epsilon=0.013):
        """
        使用差分隱私技術訓練模型，並在特定節點即時記錄未擾動前的梯度特徵值。

        參數:
            epochs (int): 訓練的總輪數，默認為 1。
            N (int): 候選梯度的數量，用於指數機制，默認為 10。
            epsilon (float): 差分隱私的隱私參數，默認為 0.013。

        返回:
            tuple: (perturbed_gradients, train_loss)
                - perturbed_gradients (dict): 擾動後的梯度字典。
                - train_loss (float): 每個 epoch 的平均損失。
        """
        if self.role != 'replica':
            print(f"Node {self.node_id} 不是副本節點，無法使用差分隱私進行訓練。")
            return None, None

        # 將模型移動到指定設備（如 GPU 或 CPU）
        self.model.to(self.device)

        # 記錄初始參數（可選）
        initial_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}

        # 初始化 previous_gradients 為零張量，用於存儲每個參數在上一輪的梯度
        previous_gradients = {name: torch.zeros_like(param).to(self.device) for name, param in self.model.named_parameters()}

        # 如果 node_id 為 3，準備即時寫入特徵值的文件
        # **修改部分：移除 node_id 的限制或調整為包含 node_id=1**
        # 選項 1：包含 node_id=1 和 node_id=3
        if self.node_id in [1, 3]:
            # 確保保存特徵值的目錄存在
            os.makedirs('gradient_features', exist_ok=True)
            # 定義特徵值文件的路徑，使用節點 ID 來區分不同節點的文件
            feature_file_path = f'gradient_features/gradient_features_node_{self.node_id}.json'
            # 初始化文件，寫入開頭的列表標誌
            with open(feature_file_path, 'w', encoding='utf-8') as f:
                f.write('[\n')  # 開始一個 JSON 數組

        # 初始化變量以存儲最終的擾動後梯度和損失
        final_perturbed_gradients = {}
        final_train_loss = 0.0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            # 重置 previous_gradients 為零張量，因為每個 epoch 被視為新的迭代
            previous_gradients = {name: torch.zeros_like(param).to(self.device) for name, param in self.model.named_parameters()}

            # 遍歷數據加載器中的每個批次，並顯示訓練進度
            for batch_idx, (data, target) in enumerate(tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                # 將數據和目標移動到指定設備
                data, target = data.to(self.device), target.to(self.device)
                # 清除優化器的梯度
                self.optimizer.zero_grad()
                # 前向傳播
                output = self.model(data)
                # 計算損失
                loss = self.criterion(output, target)
                # 反向傳播，計算梯度
                loss.backward()

                # 初始化擾動後的梯度字典
                perturbed_gradients = {}

                # 對每個參數進行指數機制的梯度擾動
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        continue  # 如果參數沒有梯度，跳過

                    # 獲取當前梯度並移動到指定設備
                    current_grad = param.grad.clone().to(self.device)
                    # 獲取前一輪的梯度
                    prev_grad = previous_gradients[name]
                    # 計算取樣界限 C
                    C = current_grad - prev_grad
                    # 根據 C 的符號確定取樣下界和上界
                    R_lower = torch.where(C < 0, current_grad - C.abs(), current_grad)
                    R_upper = torch.where(C < 0, current_grad, current_grad + C)

                    # 生成 N 個均勻分布的步長
                    steps = torch.linspace(0, 1, steps=N).to(self.device)
                    steps_shape = [N] + [1] * (current_grad.dim())  # 調整步長的形狀以進行廣播
                    steps = steps.view(steps_shape)

                    # 擴展 R_lower 和 R_upper 的形狀，以便與步長進行廣播運算
                    R_lower_expanded = R_lower.unsqueeze(0)
                    R_upper_expanded = R_upper.unsqueeze(0)

                    # 計算候選集 S，形狀為 (N, *param.shape)
                    S = R_lower_expanded + (R_upper_expanded - R_lower_expanded) * steps

                    # 計算距離 d，使用絕對值差異
                    distances = torch.abs(current_grad.unsqueeze(0) - S)

                    # 獲取優化器的學習率
                    learning_rate = self.optimizer.param_groups[0]['lr']
                    eta = learning_rate

                    # 如果使用的是自適應優化器（如 Adam），根據一階和二階矩估計調整步長
                    if param in self.optimizer.state and 'exp_avg' in self.optimizer.state[param] and 'exp_avg_sq' in self.optimizer.state[param]:
                        m_t = self.optimizer.state[param]['exp_avg']
                        v_t = self.optimizer.state[param]['exp_avg_sq']
                        step_size = self.optimizer.param_groups[0]['lr'] * (m_t / (v_t.sqrt() + self.optimizer.param_groups[0]['eps']))
                        eta = step_size

                    # 確保 eta 是 float 類型，無論其原始類型為何
                    if isinstance(eta, torch.Tensor):
                        # 如果 eta 是多元素張量，取其平均值作為單一浮點數
                        if eta.numel() == 1:
                            eta = eta.item()
                        else:
                            eta = eta.mean().item()

                    # 在進行擾動之前，記錄未擾動前的特徵值
                    if self.node_id in [1, 3]:
                        # 計算有效步長（Effective Step Size）
                        effective_step_size = eta  # eta 已轉為 float

                        # 計算敏感度（Sensitivity）
                        sensitivity = (1 / eta) if eta != 0 else float('inf')

                        # 計算當前梯度的值範圍（最小值和最大值）
                        grad_min = current_grad.min().item()
                        grad_max = current_grad.max().item()

                        # 計算範圍值（range_value）和範圍值除以敏感度（range_div_sensitivity）
                        range_value = grad_max - grad_min
                        range_div_sensitivity = range_value / sensitivity if sensitivity != 0 else float('inf')

                        # 構建特徵值的記錄
                        feature_record = {
                            'epoch': epoch + 1,
                            'batch': batch_idx + 1,
                            'parameter': name,
                            'effective_step_size': effective_step_size,
                            'sensitivity': sensitivity,
                            'value_range': {
                                'min': grad_min,
                                'max': grad_max
                            },
                            'range_value': range_value,
                            'range_div_sensitivity': range_div_sensitivity
                        }

                        # 寫入特徵值到文件
                        try:
                            with open(feature_file_path, 'a', encoding='utf-8') as f:
                                # 將記錄轉換為 JSON 字符串
                                json_record = json.dumps(feature_record, ensure_ascii=False)
                                # 添加逗號和換行符，除了第一個記錄
                                if epoch == 0 and batch_idx == 0 and os.path.getsize(feature_file_path) == 2:
                                    f.write(f"  {json_record}\n")
                                else:
                                    f.write(f",\n  {json_record}\n")
                        except IOError as e:
                            print(f"寫入特徵值文件時出錯: {e}")

                    # 計算指數機制的權重
                    exponent = - (eta * epsilon / 2) * distances

                    # 為了數值穩定性，減去最大值
                    max_exponent = exponent.max(dim=0, keepdim=True)[0]
                    probabilities = torch.exp(exponent - max_exponent)
                    probabilities = probabilities / probabilities.sum(dim=0, keepdim=True)

                    # 將概率展平為 (num_elements, N) 的形狀，以便對每個元素進行獨立抽樣
                    flat_probabilities = probabilities.view(N, -1).transpose(0, 1)

                    # 對每個元素根據概率分佈進行抽樣，獲取選擇的索引
                    indices = torch.multinomial(flat_probabilities, num_samples=1).squeeze(1)

                    # 將 S 展平以匹配抽樣的索引
                    S_flat = S.view(N, -1).transpose(0, 1)
                    # 根據抽樣的索引從候選集 S 中選擇擾動後的值
                    perturbed_values = S_flat[torch.arange(S_flat.size(0)), indices]

                    # 恢復擾動梯度的形狀
                    perturbed_grad = perturbed_values.view_as(current_grad)

                    # 保存擾動後的梯度
                    perturbed_gradients[name] = perturbed_grad

                    # 更新參數的梯度為擾動後的梯度
                    param.grad = perturbed_grad

                    # 更新 previous_gradients，保存當前的梯度作為下一輪的前一輪梯度
                    previous_gradients[name] = current_grad.clone().detach()

                # 使用優化器根據擾動後的梯度更新模型參數
                self.optimizer.step()
                # 累積損失值
                total_loss += loss.item()

            # 如果 node_id 為 1 或 3，結束 JSON 數組
            if self.node_id in [1, 3]:
                try:
                    with open(feature_file_path, 'a', encoding='utf-8') as f:
                        f.write('\n]\n')  # 結束 JSON 數組
                    print(f"已完成並關閉特徵值文件：{feature_file_path}")
                except IOError as e:
                    print(f"關閉特徵值文件時出錯: {e}")

            # 計算每個 epoch 的平均損失
            final_train_loss = total_loss / len(self.data_loader)
            print(f"Node {self.node_id} - Train Loss: {final_train_loss}")

            # 收集擾動後的梯度（這裡選擇最後一個 epoch 的梯度作為代表）
            final_perturbed_gradients = perturbed_gradients

            # 返回擾動後的梯度和訓練損失
            return final_perturbed_gradients, final_train_loss

    def train_model_exdp(self, epochs=1, N=10, epsilon=1, bit_width=16):
        """
        訓練模型並應用指數機制差分隱私（ExDP）。

        參數：
        - epochs (int): 訓練的輪數。
        - N (int): 每個取樣範圍內的候選數量。
        - epsilon (float): 差分隱私的隱私參數。
        - bit_width (int): 用於計算 R_MAXS 的位元寬度。

        返回：
        - perturbed_model_updates (dict): 擾動後的模型更新。
        - train_loss (float): 訓練損失。
        """
        if self.role != 'replica':
            print(f"Node {self.node_id} is not a replica node and cannot train with differential privacy.")
            return None, None
        
        self.model.to(self.device)
        
        # 1. 記錄初始參數
        initial_params = {name: param.clone().detach().to(self.device) for name, param in self.model.named_parameters()}
        
        # 初始化第一輪的模型更新參數
        if self.previous_model_updates is None:
            self.previous_model_updates = {name: torch.zeros_like(param).to(self.device) for name, param in self.model.named_parameters()}
        
        # 訓練過程
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(self.data_loader)
            print(f"Node {self.node_id} - Epoch {epoch+1} - Train Loss: {train_loss}")
        
        # 訓練結束後，計算模型更新
        current_model_params = {name: param.clone().detach().to(self.device) for name, param in self.model.named_parameters()}

        # currentmodel_updates 計算
        current_model_updates = {}
        for name, param in current_model_params.items():
            current_model_updates[name] = param - initial_params[name]

        # 計算剪裁範圍 R_MAXS，使用 dACIQ 方法
        current_clipped_update = current_model_updates  # 移除列表包裹，使其保持字典結構
        R_MAXS = compute_R_MAXS([current_clipped_update], bit_width=bit_width)  # 注意這裡的參數仍然需要列表
        print(R_MAXS)
        # 剪裁模型更新，使用 R_MAXS 作為敏感度
        for name in current_clipped_update:
            max_value = R_MAXS.get(name, None)
            if max_value is not None:
                current_clipped_update[name] = torch.clamp(current_clipped_update[name], min=-max_value, max=max_value)
            else:
                print(f"警告：層 {name} 沒有裁剪閾值，將跳過此層的裁剪。")
        
        # 對剪裁後的模型更新應用指數機制
        perturbed_model_updates = {}
        for name, current_update in current_clipped_update.items():
            # 從 previous_model_updates 中獲取對應的層的更新值
            previous_update = self.previous_model_updates.get(name, None)
            if previous_update is None:
                print(f"警告：層 {name} 沒有找到對應的上一輪更新，將跳過此層的計算。")
                continue

            # 計算 C = 當前裁減過的梯度 - 上一輪裁減過的梯度
            C = current_update - previous_update

            # 根據 C 的符號定義取樣範圍 R
            R_lower = torch.where(C < 0, current_update - torch.abs(C), current_update)
            R_upper = torch.where(C < 0, current_update, current_update + C)

            # 生成候選集 S
            N_candidates = N  # 候選值的數量
            param_shape = current_update.shape
            num_elements = current_update.numel()

            # 展平張量
            update_flat = current_update.view(-1)
            R_lower_flat = R_lower.view(-1)
            R_upper_flat = R_upper.view(-1)

            # 對於每個元素，生成 N 個候選值
            steps = torch.linspace(0, 1, steps=N_candidates).to(self.device)
            steps = steps.view(-1, 1)  # 形狀：(N, 1)

            # 擴展 R_lower 和 R_upper
            R_lower_expanded = R_lower_flat.unsqueeze(0)  # 形狀：(1, num_elements)
            R_upper_expanded = R_upper_flat.unsqueeze(0)  # 形狀：(1, num_elements)

            # 計算候選值集合 S，形狀：(N, num_elements)
            S = R_lower_expanded + (R_upper_expanded - R_lower_expanded) * steps

            # 計算距離 d，使用絕對值差異
            distances = torch.abs(update_flat.unsqueeze(0) - S)  # 形狀：(N, num_elements)
            
            max_value = R_MAXS.get(name, None)

            # 使用剪裁範圍作為敏感度 delta_u
            delta_u = 2 * max_value  # 敏感度為裁剪範圍的兩倍

            # 計算擾動概率
            exponent = - (epsilon / (2 * delta_u)) * distances  # 形狀：(N, num_elements)

            # 為了數值穩定性，減去每列的最大值
            max_exponent = exponent.max(dim=0, keepdim=True)[0]
            probabilities = torch.exp(exponent - max_exponent)
            probabilities = probabilities / probabilities.sum(dim=0, keepdim=True)

            # 對每個元素進行抽樣
            indices = torch.multinomial(probabilities.transpose(0, 1), num_samples=1).squeeze(1)  # 形狀：(num_elements,)

            # 從候選集合 S 中選擇擾動後的值
            perturbed_values = S[indices, torch.arange(num_elements)]

            # 恢復擾動後的模型更新形狀
            perturbed_update = perturbed_values.view(param_shape)
            perturbed_model_updates[name] = perturbed_update

            # 更新上一輪的模型更新
            self.previous_model_updates[name] = current_clipped_update[name]

        
        return perturbed_model_updates, train_loss

    def train_model_exdp_C(self, epochs=1, N=10, epsilon=1.0):
        """
        使用 C 作為敏感度，訓練模型並應用指數機制差分隱私（ExDP）。

        參數：
        - epochs (int): 訓練的輪數。
        - N (int): 每個取樣範圍內的候選數量。
        - epsilon (float): 差分隱私的隱私參數。

        返回：
        - perturbed_model_updates (dict): 擾動後的模型更新。
        - train_loss (float): 訓練損失。
        """
        if self.role != 'replica':
            print(f"節點 {self.node_id} 不是副本節點，無法進行差分隱私訓練。")
            return None, None

        self.model.to(self.device)

        # 1. 記錄初始參數
        initial_params = {name: param.clone().detach().to(self.device) for name, param in self.model.named_parameters()}

        # 初始化 previous_model_updates，如果沒有的話
        if not hasattr(self, 'previous_model_updates') or self.previous_model_updates is None:
            self.previous_model_updates = {name: torch.zeros_like(param).to(self.device) for name, param in self.model.named_parameters()}

        # 訓練過程
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        train_loss = total_loss / len(self.data_loader)
        print(f"節點 {self.node_id} - 訓練損失: {train_loss}")

        # 訓練結束後，計算模型更新
        current_model_params = {name: param.clone().detach().to(self.device) for name, param in self.model.named_parameters()}

        # 計算當前模型更新
        current_model_updates = {}
        for name, param in current_model_params.items():
            current_model_updates[name] = param - initial_params[name]

        # 初始化擾動後的模型更新字典
        perturbed_model_updates = {}

        # 對每個參數進行指數機制擾動
        for name, current_update in current_model_updates.items():
            # 從 previous_model_updates 中獲取對應的上一輪更新值
            previous_update = self.previous_model_updates.get(name, torch.zeros_like(current_update))

            # 計算 C = 當前模型更新 - 上一輪模型更新
            C = current_update - previous_update

            # 根據 C 的符號定義取樣範圍 R
            R_lower = torch.where(C < 0, current_update + C, current_update)
            R_upper = torch.where(C < 0, current_update, current_update + C)

            # 生成候選集 S
            N_candidates = N  # 候選值的數量
            param_shape = current_update.shape
            num_elements = current_update.numel()

            # 展平張量
            current_update_flat = current_update.view(-1)
            R_lower_flat = R_lower.view(-1)
            R_upper_flat = R_upper.view(-1)

            # 對於每個元素，生成 N 個候選值
            steps = torch.linspace(0, 1, steps=N_candidates).to(self.device).unsqueeze(1)  # 形狀：(N, 1)

            # 擴展 R_lower 和 R_upper，形狀：(1, num_elements)
            R_lower_expanded = R_lower_flat.unsqueeze(0)
            R_upper_expanded = R_upper_flat.unsqueeze(0)

            # 計算候選集合 S，形狀：(N, num_elements)
            S = R_lower_expanded + (R_upper_expanded - R_lower_expanded) * steps

            # 計算距離 d，使用絕對值差異
            distances = torch.abs(current_update_flat.unsqueeze(0) - S)  # 形狀：(N, num_elements)

            # 計算敏感度 delta_u，等於 C 的絕對值
            delta_u = torch.abs(C.view(-1))  # 形狀：(num_elements,)

            # 避免敏感度為零，添加一個小的常數
            delta_u = delta_u + 1e-10

            # 將 delta_u 擴展以匹配 distances 的形狀
            delta_u_expanded = delta_u.unsqueeze(0)  # 形狀：(1, num_elements)

            # 計算擾動概率
            exponent = - (epsilon / (2 * delta_u_expanded)) * distances  # 形狀：(N, num_elements)

            # 為了數值穩定性，減去每列的最大值
            max_exponent = exponent.max(dim=0, keepdim=True)[0]
            probabilities = torch.exp(exponent - max_exponent)
            probabilities = probabilities / probabilities.sum(dim=0, keepdim=True)

            # 對每個元素進行抽樣
            indices = torch.multinomial(probabilities.transpose(0, 1), num_samples=1).squeeze(1)  # 形狀：(num_elements,)

            # 從候選集合 S 中選擇擾動後的值
            perturbed_values = S[indices, torch.arange(num_elements)]

            # 恢復擾動後的模型更新形狀
            perturbed_update = perturbed_values.view(param_shape)
            perturbed_model_updates[name] = perturbed_update

            # 更新 previous_model_updates
            self.previous_model_updates[name] = current_update.clone().detach()

        return perturbed_model_updates, train_loss

    def verify_proof(self, snarkjs_path, zkey_path, public_file_path, proof_file_path):
        if self.role != 'replica':
            print(f"Node {self.node_id} is not a replica node and cannot verify proofs.")
            return self.node_id, False, "Not a replica node"

        try:
            result = subprocess.run(
                [snarkjs_path, "groth16", "verify", zkey_path, public_file_path, proof_file_path],
                check=True, text=True, capture_output=True
            )
            clean_output = remove_ansi_escape_sequences(result.stdout.strip())
            if "[INFO]  snarkJS: OK!" in clean_output:
                return self.node_id, True, clean_output
            else:
                return self.node_id, False, clean_output
        except subprocess.CalledProcessError as e:
            return self.node_id, False, e.stderr

    def update_model(self, global_model_state_dict):
        self.model.load_state_dict(global_model_state_dict)

    def aggregate_with_aggrator(self, client_gradients, lr=0.001):
        if self.role != 'primary':
            print(f"Node {self.node_id} is not a primary node and cannot aggregate with aggregator.")
            return

        aggregated_gradients = self.aggregator(client_gradients)
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_gradients:
                    param += aggregated_gradients[name]

    def batch_aggregate(self, processed_gradients, public_json_path, lr=0.001, bit_width=8, r_maxs=None, batch_size=50, pad_zero=3):
        if self.role != 'primary':
            print(f"Node {self.node_id} is not a primary node and cannot perform batch aggregation.")
            return
        num_clients = len(processed_gradients)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        first_gradient = processed_gradients[0]

        # 讀取 public.json 中的值
        public_values = read_public_json(public_json_path)

        # 初始化變數以儲存分配過程
        public_values_idx = 0

        # 初始化聚合梯度
        aggregated_big_ints = {}

        # 分配 public.json 中的值到梯度，並轉換為整數
        for param_name, param_value in first_gradient.items():
            param_size = param_value.size
            if public_values_idx + param_size <= len(public_values):
                aggregated_big_ints[param_name] = [int(x) for x in public_values[public_values_idx:public_values_idx + param_size]]
                public_values_idx += param_size
            else:
                print(f"錯誤: 無法為參數 {param_name} 分配足夠的值。需要 {param_size}，但剩餘 {len(public_values) - public_values_idx}")

        # 解码和反量化聚合后的梯度
        aggregated_gradients = {}
        for name, big_ints in aggregated_big_ints.items():
            original_shape = self.global_model.state_dict()[name].shape
            # 批量解码大整数
            decoded_quantized_twos_complement = batch_decode_from_big_int(big_ints, original_shape, batch_size, bit_width, pad_zero)
            # 转换前先确保创建一个数组副本
            decoded_quantized_twos_complement = decoded_quantized_twos_complement.copy()
            if r_maxs is not None and name in r_maxs:
                r_max = r_maxs[name] 
            else:
                r_max = 50  # 或者設定為預設值
            # 反量化
            dequantized_gradients = dequantize_from_twos_complement(torch.tensor(decoded_quantized_twos_complement), bit_width, r_max)
            aggregated_gradients[name] = dequantized_gradients.clone().detach().to(device).float()
            
            
        # 更新全球模型的权重
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_gradients:
                    param += aggregated_gradients[name] / num_clients

    def generate_zero_knowledge_proof(self, round, processed_gradients, snarkjs_path, wasm_path, zkey_path):
        if self.role != 'primary':
            print(f"Node {self.node_id} is not a primary node and cannot generate zero-knowledge proof.")
            return None

        json_file_path, round_folder_path = save_gradients_to_json(round, processed_gradients, snarkjs_path, wasm_path, zkey_path)
        if not json_file_path:
            raise ValueError("Failed to save gradients to JSON.")

        output_witness_path = calculate_witness(round, json_file_path, round_folder_path, snarkjs_path, wasm_path)
        if not output_witness_path:
            raise ValueError("Failed to calculate witness.")

        result = create_proof_and_public(round, output_witness_path, round_folder_path, snarkjs_path, zkey_path)
        if not result:
            raise ValueError("Failed to create proof and public.")

        proof_file_path, public_file_path = result

        return public_file_path

    def get_global_model(self):
        if self.global_model is None:
            raise ValueError("Global model is not set or aggregated yet.")
        return self.global_model

    def become_primary(self):
        self.role = 'primary'
        # 可以在這裡添加狀態同步或其他必要操作

    def become_replica(self):
        self.role = 'replica'
        # data_loader 將在主程式中分配
        # 可以在這裡添加狀態清理或其他必要操作
