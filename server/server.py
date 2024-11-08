import torch
import json
import os
import numpy as np
from batch import quantize_and_to_twos_complement,batch_encode_to_big_int,batch_decode_from_big_int,dequantize_from_twos_complement
import json
import subprocess

# 讀取 public.json 文件
def read_public_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data
    
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
        return False
    
    return json_file_path, round_folder_path

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
    
class Server:
    def __init__(self, initial_model, aggregator):
        self.server_id = None
        self.global_model = initial_model
        self.aggregator = aggregator
    
    def aggregate(self, client_gradients, lr=0.001):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_clients = len(client_gradients)
        if client_gradients:
            aggregated_gradients = {name: torch.zeros_like(gradient).to(device) for name, gradient in client_gradients[0].items()}
        else:
            print("警告：client_gradients 列表是空的！")
            return
        # 初始化聚合梯度
        
        aggregated_gradients = {name: torch.zeros_like(gradient).to(device) for name, gradient in client_gradients[0].items()}

        # 对选中的客户端梯度进行聚合
        for gradients in client_gradients:
            for name, gradient in gradients.items():
                aggregated_gradients[name] += gradient / num_clients

        # 更新全局模型的权重
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_gradients:
                    param += aggregated_gradients[name]

    def aggregate_with_aggrator(self, client_gradients, lr=0.001):

        aggregated_gradients = self.aggregator(client_gradients)
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_gradients:
                    param += aggregated_gradients[name]


    def batch_aggregate(self, processed_gradients, public_json_path, lr=0.001, bit_width=8, r_maxs=None, batch_size=50, pad_zero=3):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 檢查 processed_gradients 列表中第一個客戶的梯度形狀與各自參數數量
        '''if processed_gradients:
            first_gradient = processed_gradients[0]
            print("第一個客戶的梯度形狀: ")
            for param_name, param_value in first_gradient.items():
                print(f"參數名: {param_name}, 形狀: {param_value.shape}, 數量: {param_value.size}")'''
        
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
                    param -= lr * aggregated_gradients[name]


    def generate_zero_knowledge_proof(self, round, processed_gradients, snarkjs_path, wasm_path, zkey_path):
        
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
        """
        获取经过聚合的全局模型。
        """
        if self.global_model is None:
            raise ValueError("Global model is not set or aggregated yet.")
        return self.global_model
