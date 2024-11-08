import torch
import numpy as np

def quantize_and_to_twos_complement(gradients, bit_width=8, r_max=1):
    # 確保梯度在 GPU 上，如果可用的話
    gradients = gradients.to('cuda') if torch.cuda.is_available() else gradients
    # 獲取梯度的符號並取絕對值
    og_sign = torch.sign(gradients)
    uns_gradients = torch.abs(gradients)
    # 計算量化因子
    scale_factor = (2 ** (bit_width - 1)) - 1.0
    # 進行量化
    quantized_value = torch.round(uns_gradients / r_max * scale_factor)
    # 恢復符號
    quantized_value_signed = quantized_value * og_sign
    # 轉換為二的補碼表示
    twos_complement = to_twos_complement(quantized_value_signed, bit_width)
    return twos_complement

def to_twos_complement(quantized_gradients, bit_width):
    # 計算 2 的 bit_width 次方
    two_power_bit_width = 2 ** bit_width
    # 建立負數的掩碼
    mask_negative = quantized_gradients < 0
    # 對負數進行二的補碼轉換，正數保持不變
    twos_complement = torch.where(
        mask_negative, 
        two_power_bit_width + quantized_gradients, 
        quantized_gradients
    )
    return twos_complement

def batch_encode_to_big_int(quantized_gradients, batch_size, bit_width, pad_zero):
    total_bit_width = bit_width + pad_zero
    # 將梯度展平成一維數組
    quantized_gradients_flattened = quantized_gradients.flatten()
    num_batches = int(np.ceil(len(quantized_gradients_flattened) / batch_size))
    batched_nums = np.zeros(num_batches, dtype=object)
    # 逐批處理梯度
    for i in range(num_batches):
        # 獲取當前批次的數值
        current_batch = quantized_gradients_flattened[i * batch_size: (i + 1) * batch_size]
        big_int = 0
        for value in current_batch:
            # 確保 value 是單個數值，然後進行轉換和編碼
            value_int = int(value)
            big_int = (big_int << total_bit_width) | value_int
        batched_nums[i] = big_int
    return batched_nums

def batch_decode_from_big_int(encoded_big_ints, original_shape, batch_size, bit_width, pad_zero):
    total_bit_width = bit_width + pad_zero  # 總位寬，包括填充的零位
    total_elements = np.prod(original_shape)  # 需要解碼的總元素數
    decoded_data = np.zeros(total_elements, dtype=int)  # 創建解碼後的數組
    current_index = 0
    for big_int in encoded_big_ints:
        # 計算當前批次需要處理的元素數量
        elements_to_process = batch_size if (current_index + batch_size < total_elements) else (total_elements - current_index)
        current_batch = []
        for i in range(elements_to_process):
            shift_amount = i * total_bit_width
            mask = (1 << total_bit_width) - 1
            value = (big_int >> shift_amount) & mask
            current_batch.append(value)
        # 反轉當前批次的數據
        current_batch.reverse()
        # 將解碼的數據添加到結果中
        for value in current_batch:
            if current_index < total_elements:
                decoded_data[current_index] = value
                current_index += 1
    try:
        reshaped_array = np.reshape(decoded_data, original_shape)
    except ValueError as e:
        print(f"重塑時出錯: {e}. 預期形狀為 {original_shape}，但得到 {decoded_data.shape}")
        return None
    return reshaped_array

def from_twos_complement(twos_complement_gradients, bit_width, pad_zero=3):
    # 建立負數的掩碼
    mask_negative = twos_complement_gradients >= 2 ** (bit_width - 1)
    # 計算偏移量
    offset = 2 ** bit_width
    # 恢復原始值
    original_values = torch.where(
        mask_negative, 
        twos_complement_gradients - offset, 
        twos_complement_gradients
    )
    # 計算符號位
    sign = (twos_complement_gradients >> bit_width) & ((1 << pad_zero) - 1)
    # 根據 bit_width 生成掩碼
    mask = (1 << bit_width) - 1
    literal = twos_complement_gradients & mask
    # 判斷是否需要轉換為負值
    max_positive = 2 ** (bit_width - 1) - 1
    literal = torch.where(literal > max_positive, literal - (1 << bit_width), literal)
    return literal

def dequantize_from_twos_complement(twos_complement_gradients, bit_width=8, r_max=1):
    # 將二的補碼轉換回有符號整數
    quantized_value_signed = from_twos_complement(twos_complement_gradients, bit_width)
    # 計算反量化因子
    scale_factor = (2 ** (bit_width - 1)) - 1.0
    # 進行反量化
    uns_result = quantized_value_signed / scale_factor * r_max
    return uns_result

def batch_process_gradients(client_gradients, bit_width, r_maxs, batch_size, pad_zero):

    processed_gradients_list = []
    num_clients = len(client_gradients)
    for gradients in client_gradients:
        processed_gradients = {}
        for name, gradient in gradients.items():
            # 量化並轉換為二的補碼
            if r_maxs is not None and name in r_maxs:
                r_max = r_maxs[name]
            else:
                r_max = 50  # 或者設定為預設值
            twos_complement = quantize_and_to_twos_complement(gradient, bit_width, r_max)
            # 編碼為大整數
            processed_big_int = batch_encode_to_big_int(
                twos_complement.cpu().numpy(), batch_size, bit_width, pad_zero
            )
            # 保存處理後的梯度
            processed_gradients[name] = processed_big_int
        # 添加到處理後的梯度列表
        processed_gradients_list.append(processed_gradients)
    return processed_gradients_list

def compute_R_MAXS(client_gradients, bit_width=8):
    """
    根據 dACIQ 方法計算每一層的 R_MAXS（適合的剪枝閾值）
    
    參數：
    - client_gradients: 客戶端梯度的列表，每個元素都是一個字典，鍵為層的名稱，值為對應的梯度張量
    - bit_width: 量化的位元寬度，預設為 8
    
    返回：
    - R_MAXS: 字典，鍵為層的名稱，值為該層計算出的 R_MAXS 值
    """
    
    # 步驟 1：初始化資料結構以保存每一層的最大值、最小值和尺寸
    layer_names = client_gradients[0].keys()
    layer_stats = {name: {'max_values': [], 'min_values': [], 'size': 0} for name in layer_names}
    
    # 步驟 2：收集每一層的最大值、最小值和尺寸
    num_clients = len(client_gradients)
    for gradients in client_gradients:
        for name, gradient in gradients.items():
            # 檢查梯度是否為空
            if gradient is None:
                print(f"警告：層 {name} 的梯度為 None，將跳過此層的 R_MAXS 計算。")
                continue
            # 將梯度展平成一維數組，並轉換為 NumPy ndarray
            gradient_flat = gradient.flatten().detach().cpu().numpy()
            # 檢查梯度中是否存在 NaN 或 Inf
            if not np.all(np.isfinite(gradient_flat)):
                print(f"警告：層 {name} 的梯度包含 NaN 或 Inf 值，將跳過此層的 R_MAXS 計算。")
                continue
            # 更新最大值和最小值列表
            layer_stats[name]['max_values'].append(np.max(gradient_flat))
            layer_stats[name]['min_values'].append(np.min(gradient_flat))
            # 記錄每一層的尺寸（僅需一次）
            if layer_stats[name]['size'] == 0:
                layer_stats[name]['size'] = gradient_flat.size * num_clients  # 乘以客戶端數量
    
    # 步驟 3：計算每一層的全域最大值和最小值
    R_MAXS = {}
    for name in layer_names:
        # 檢查是否有有效的最大值和最小值
        if not layer_stats[name]['max_values'] or not layer_stats[name]['min_values']:
            print(f"警告：層 {name} 沒有有效的最大值或最小值，將跳過此層的 R_MAXS 計算。")
            continue
        max_value = max(layer_stats[name]['max_values'])
        min_value = min(layer_stats[name]['min_values'])
        layer_size = layer_stats[name]['size']
    
        # 步驟 4：計算每一層的剪枝閾值（R_MAXS）
        # 使用 dACIQ 方法計算最佳剪枝值
        clipping_threshold = _calculate_clipping_threshold(
            max_value, min_value, layer_size, bit_width
        )
        # 考慮到聚合時梯度會累加，乘以客戶端數量
        R_MAXS[name] = clipping_threshold * num_clients
    
    return R_MAXS

def _calculate_clipping_threshold(max_value, min_value, values_size, num_bits):
    """
    使用 ACIQ 方法計算最佳剪枝閾值（適合的 alpha 值）
    
    參數：
    - max_value: 梯度的最大值
    - min_value: 梯度的最小值
    - values_size: 梯度的總元素數量
    - num_bits: 量化的位元寬度
    
    返回：
    - clipping_threshold: 計算出的最佳剪枝閾值
    """
    import numpy as np
    
    # 標準正態分布（N(0,1）下的最佳剪枝值對照表
    alpha_gaus = {
        2: 1.71063516, 3: 2.02612148, 4: 2.39851063, 5: 2.76873681,
        6: 3.12262004, 7: 3.45733738, 8: 3.77355322, 9: 4.07294252,
        10: 4.35732563, 11: 4.62841243, 12: 4.88765043, 13: 5.1363822,
        14: 5.37557768, 15: 5.60671468, 16: 5.82964388, 17: 6.04501354,
        18: 6.25385785, 19: 6.45657762, 20: 6.66251328, 21: 6.86053901,
        22: 7.04555454, 23: 7.26136857, 24: 7.32861916, 25: 7.56127906,
        26: 7.93151212, 27: 7.79833847, 28: 7.79833847, 29: 7.9253003,
        30: 8.37438905, 31: 8.37438899, 32: 8.37438896
    }
    
    # 檢查位元寬度是否在對照表中
    if num_bits not in alpha_gaus:
        raise ValueError(f"位元寬度 {num_bits} 不在支持的範圍內。")
    
    # 根據數值範圍計算 sigma（效率高但精確度略低）
    gaussian_const = (0.5 * 0.35) * (1 + (np.pi * np.log(4)) ** 0.5)
    sigma = ((max_value - min_value) * gaussian_const) / ((2 * np.log(values_size)) ** 0.5)
    # 獲取對應位元寬度的 alpha 值，計算剪枝閾值
    clipping_threshold = alpha_gaus[num_bits] * sigma
    
    return clipping_threshold





if __name__ == "__main__":
    import torch
    import numpy as np
    import pandas as pd
    import pickle
    import os


    bit_width = 16  # 量化位寬
    pad_zero = 3   # 填充零位數
    batch_size = 25  # 批次大小
    r_max = 1
    # 2. 量化並轉換為二進補碼
    bit_width = 8
    r_max = 1
    # 1. 產生原始數據
    torch.manual_seed(42)
    original_data = torch.randn(10, 10)  # 10個樣本，每個樣本10個特徵
    original_data = original_data/10
    quantized_twos_complement = quantize_and_to_twos_complement(original_data, bit_width, r_max)
    print(original_data)
    print(quantized_twos_complement.shape)
    # 3. 批次編碼為大整數
    batch_size = 10  # 一個批次包含全部樣本
    pad_zero = 3
    encoded_big_ints = batch_encode_to_big_int(quantized_twos_complement, batch_size, bit_width, pad_zero)
    print(encoded_big_ints)
    # 4. 對編碼後的數據進行加總
    sum_encoded_big_ints = np.sum(encoded_big_ints)
    print(sum_encoded_big_ints)
    # 5. 解碼加總後的編碼數據
    # 將加總後的大整數視為一個包含單一元素的數組
    sum_encoded_big_ints_array = np.array([sum_encoded_big_ints], dtype=object)
    # 解碼加總後的數據
    decoded_sum_data = batch_decode_from_big_int(sum_encoded_big_ints_array, (1, 10), batch_size, bit_width, pad_zero)

    # 轉換前先確保創建一個數組副本
    decoded_sum_data = decoded_sum_data.copy()

    # 現在可以安全地轉換為 PyTorch 張量
    dequantized_sum_data = dequantize_from_twos_complement(torch.tensor(decoded_sum_data), bit_width, r_max)
    print(dequantized_sum_data)
    # 6. 對原始數據進行批次加總
    # 將Tensor轉換為NumPy數組以進行操作
    original_data_np = original_data.numpy()

    # 原始數據加總
    batch_sum_original = original_data_np.sum(axis=0)  # 對所有樣本的每個特徵進行加總，形狀應為 (10,)
    print(batch_sum_original)
    # 將加總後的數據轉換為 DataFrame
    df_batch_sum_original = pd.DataFrame([batch_sum_original], columns=[f'Feature_{i}' for i in range(batch_sum_original.shape[0])])


    # 將原始數據和加總後的數據儲存到CSV檔案中進行檢查
    df_original = pd.DataFrame(original_data_np)
    # 比較解碼後加總的數據和原始數據的批次加總是否相近
    dequantized_sum_data_np = dequantized_sum_data.numpy().reshape(1, -1)  # 確保形狀為 (1, 10)
    comparison = np.isclose(dequantized_sum_data_np, batch_sum_original, atol=1e-5)

    # 將原始數據、批次加總的數據和解碼後的加總數據儲存到CSV檔案中進行檢查
    df_original = pd.DataFrame(original_data_np)
    df_dequantized_sum_data = pd.DataFrame(dequantized_sum_data_np)  # 解碼後加總的數據可能需要調整形狀


    df_batch_sum_original.to_csv('batch_sum_original.csv', index=False)
    df_dequantized_sum_data.to_csv('dequantized_sum_data.csv', index=False)

