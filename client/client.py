import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import subprocess
import re

def remove_ansi_escape_sequences(text):
    ansi_escape = re.compile(r'\x1b\[.*?m')
    return ansi_escape.sub('', text)

class Client:
    def __init__(self, client_id, data_loader, model, optimizer, criterion):
        self.client_id = client_id
        self.data_loader = data_loader      
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, epochs=1):
        self.model.to(self.device)  # 确保模型在正确的设备上
        gradients = {name: torch.zeros_like(param).to(self.device) for name, param in self.model.named_parameters()}
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            # 初始化梯度累加器
            for batch_idx, (data, target) in enumerate(tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                # 累加當前batch的梯度
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad
                        
                self.optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(self.data_loader)

            print(f"Client {self.client_id} - Epoch {epoch+1} - Train Loss: {train_loss}")#, Validation Loss: {val_loss}, Accuracy: {accuracy}%

        return gradients, train_loss
        


    def train_exdp(self, epochs=1, N=10, epsilon=1.0):
        """
        使用指數機制的方案進行訓練，實現差分隱私的梯度擾動。

        參數：
            epochs (int): 訓練的輪數。
            N (int): 取樣範圍內的候選值數量。
            epsilon (float): 隱私預算。

        返回：
            gradients (dict): 訓練後的擾動梯度。
            train_loss (float): 訓練損失。
        """
        self.model.to(self.device)  # 確保模型在正確的設備上
        gradients = {name: torch.zeros_like(param).to(self.device) for name, param in self.model.named_parameters()}
        
        # 初始化 previous_gradients 為零張量
        previous_gradients = {name: torch.zeros_like(param).to(self.device) for name, param in self.model.named_parameters()}

        # 獲取學習率
        learning_rate = self.optimizer.param_groups[0]['lr']

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

                    # 計算擾動概率
                    eta = learning_rate  # η 為學習率
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

                    # 累加梯度
                    gradients[name] += perturbed_grad

                    # 更新參數的梯度為擾動後的梯度
                    param.grad = perturbed_grad

                    # 更新 previous_gradients
                    previous_gradients[name] = current_grad.clone().detach()

                self.optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(self.data_loader)
            print(f"Client {self.client_id} - Epoch {epoch+1} - Train Loss: {train_loss}")
        return gradients, train_loss

    def verify_proof(self, snarkjs_path, zkey_path, public_file_path, proof_file_path):
        try:
            result = subprocess.run(
                [snarkjs_path, "groth16", "verify", zkey_path, public_file_path, proof_file_path],
                check=True, text=True, capture_output=True
            )
            clean_output = remove_ansi_escape_sequences(result.stdout.strip())
            #print(f"Cleaned output: {clean_output}")
            if "[INFO]  snarkJS: OK!" in clean_output:
                return self.client_id, True, clean_output
            else:
                return self.client_id, False, clean_output
        except subprocess.CalledProcessError as e:
            return self.client_id, False, e.stderr

    def update_model(self, global_model):
        self.model.load_state_dict(global_model.state_dict())
            
        
