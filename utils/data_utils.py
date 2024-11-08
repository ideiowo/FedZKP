import torch
import random
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_MNIST(batch_size=32, num_clients=10):
    # 數據的預處理和轉換
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 設置 MNIST 資料集的資料夾路徑
    mnist_train_path = './data'
    mnist_test_path = './data'
    
    # 從指定路徑加載訓練和測試資料集
    train_dataset = datasets.MNIST(root=mnist_train_path, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=mnist_test_path, train=False, transform=transform, download=True)
    
    # 創建每個類別的索引列表
    class_indices = {i: [] for i in range(10)}  # MNIST 有 10 個類別
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)
    
    # 確保每個客戶端的數據分布均勻
    client_indices = [[] for _ in range(num_clients)]
    for label, indices in class_indices.items():
        per_client = len(indices) // num_clients
        remainder = len(indices) % num_clients
        start = 0
        for i in range(num_clients):
            end = start + per_client + (1 if i < remainder else 0)
            client_indices[i].extend(indices[start:end])
            start = end
    
    # 創建每個客戶端的訓練數據加載器
    client_loaders = []
    for indices in client_indices:
        dataset_per_client = Subset(train_dataset, indices)
        train_loader = DataLoader(dataset_per_client, batch_size=batch_size, shuffle=True)
        client_loaders.append(train_loader)
    
    # 直接使用整個測試集作為每個客戶端的驗證加載器
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return client_loaders, test_loader

def load_FMNIST(batch_size=64, num_clients=10):
    # 數據的預處理和轉換
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 設置 Fashion-MNIST 資料集的資料夾路徑
    fmnist_path = './data'
    
    # 從指定路徑加載訓練和測試資料集
    train_dataset = datasets.FashionMNIST(root=fmnist_path, train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root=fmnist_path, train=False, transform=transform, download=True)
    
    # 創建每個類別的索引列表
    class_indices = {i: [] for i in range(10)}  # Fashion-MNIST 有 10 個類別
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)
    
    # 確保每個客戶端的數據分布均勻
    client_indices = [[] for _ in range(num_clients)]
    for label, indices in class_indices.items():
        per_client = len(indices) // num_clients
        for i in range(num_clients):
            client_indices[i].extend(indices[i*per_client:(i+1)*per_client])
    
    # 創建每個客戶端的訓練數據加載器
    client_loaders = []
    for indices in client_indices:
        dataset_per_client = Subset(train_dataset, indices)
        train_loader = DataLoader(dataset_per_client, batch_size=batch_size, shuffle=True)
        client_loaders.append(train_loader)
    
    # 直接使用整個測試集作為每個客戶端的驗證加載器
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return client_loaders, test_loader


def load_CIFAR10(batch_size=64, num_clients=10):
    # 數據的預處理和轉換
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 設置 CIFAR-10 資料集的資料夾路徑
    cifar10_path = './data'
    
    # 從指定路徑加載訓練和測試資料集
    train_dataset = datasets.CIFAR10(root=cifar10_path, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=cifar10_path, train=False, transform=transform, download=True)
    
    # 創建每個類別的索引列表
    class_indices = {i: [] for i in range(10)}  # CIFAR-10 有 10 個類別
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)
    
    # 確保每個客戶端的數據分布均勻
    client_indices = [[] for _ in range(num_clients)]
    for label, indices in class_indices.items():
        per_client = len(indices) // num_clients
        remainder = len(indices) % num_clients
        start = 0
        for i in range(num_clients):
            end = start + per_client + (1 if i < remainder else 0)
            client_indices[i].extend(indices[start:end])
            start = end
    
    # 創建每個客戶端的訓練數據加載器
    client_loaders = []
    for indices in client_indices:
        dataset_per_client = Subset(train_dataset, indices)
        train_loader = DataLoader(dataset_per_client, batch_size=batch_size, shuffle=True, drop_last=True)
        client_loaders.append(train_loader)
    
    # 直接使用整個測試集作為每個客戶端的驗證加載器
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return client_loaders, test_loader


def print_loader_details(loader):
    label_count = {}
    for _, labels in loader:
        for label in labels.numpy():
            label_count[label] = label_count.get(label, 0) + 1
    # 對字典按照鍵進行排序
    sorted_label_count = {k: label_count[k] for k in sorted(label_count)}
    return sorted_label_count


if __name__ == "__main__":
    sns.set()
   
    client_loaders, client_val_loaders = load_MNIST(num_clients=10)
    # 使用函數

    client_train_label_counts = []

    for train_loader in client_loaders:
        train_label_count = print_loader_details(train_loader)
        client_train_label_counts.append(train_label_count)

    # 將數據整理成一個二維數組
    num_labels = 10  # 假設有 10 個不同的類別
    data_matrix = np.zeros((len(client_loaders), num_labels), dtype=int)  # 將 dtype 設置為 int
    
    for i, counts in enumerate(client_train_label_counts):
        for label, count in counts.items():
            data_matrix[i, label] = count

    # 繪製熱圖
    plt.figure(figsize=(10, 5))
    sns.heatmap(data_matrix, annot=True, fmt="g", yticklabels=[f"Client {i+1}" for i in range(len(client_loaders))], xticklabels=range(num_labels))
    plt.xlabel("Data Category")
    plt.ylabel("Client")
    plt.title("Training Data Distribution Across Clients")
    plt.show()
