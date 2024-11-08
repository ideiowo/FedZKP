import hashlib
import json
from datetime import datetime

class Block:
    def __init__(self, index, round_num, transactions, timestamp, previous_hash, votes=None, nonce=0):
        self.index = index
        self.round_num = round_num  # 新增輪數屬性
        self.transactions = transactions  # FL相關數據，如梯度
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.votes = votes if votes is not None else {}
        self.nonce = nonce
        self.hash = self.compute_hash()
    
    def compute_hash(self):
        """
        計算區塊的 SHA-256 哈希值，包含所有必要的字段
        """
        block_string = json.dumps({
            'index': self.index,
            'round_num': self.round_num,  # 包含輪數
            'transactions': self.transactions,
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'votes': self.votes,
            'nonce': self.nonce
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    def to_dict(self):
        """
        將區塊轉換為字典格式
        """
        return {
            'index': self.index,
            'round_num': self.round_num,  # 包含輪數
            'transactions': self.transactions,
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'votes': self.votes,
            'nonce': self.nonce,
            'hash': self.hash
        }

class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()
        self.current_round = 0  # 初始化輪數
    
    def create_genesis_block(self):
        """
        創建創世區塊（第一個區塊）
        """
        genesis_block = Block(
            index=0,
            round_num=0,  # 創世區塊的輪數為0
            transactions=[],  # 創世區塊不包含任何FL數據
            timestamp=str(datetime.now()),
            previous_hash="0"
        )
        self.chain.append(genesis_block)
    
    @property
    def last_block(self):
        return self.chain[-1]
    
    def add_block(self, block):
        """
        將新區塊添加到區塊鏈中，並驗證其有效性
        """
        previous_hash = self.last_block.hash
        if previous_hash != block.previous_hash:
            print(f"區塊 {block.index} 的 previous_hash 不匹配。")
            return False
        
        if not self.is_valid_block(block):
            print(f"區塊 {block.index} 不符合驗證要求。")
            return False
        
        self.chain.append(block)
        self.current_round += 1  # 更新輪數
        return True
    
    def is_valid_block(self, block):
        """
        驗證區塊的有效性，包括哈希值和投票結果
        """
        # 檢查區塊哈希是否正確
        if block.hash != block.compute_hash():
            print(f"區塊 {block.index} 的哈希不正確。")
            return False
        
        # 檢查區塊的輪數是否正確
        expected_round = self.last_block.round_num + 1
        if block.round_num != expected_round:
            print(f"區塊 {block.index} 的輪數不正確。預期: {expected_round}, 實際: {block.round_num}")
            return False
        
        # 檢查投票結果是否達到共識要求（此處可根據具體需求實現）
        if not self.has_consensus(block.votes):
            print(f"區塊 {block.index} 未達到共識。")
            return False
        
        return True
    
    def has_consensus(self, votes, required_majority=1/2):
        """
        檢查投票結果是否達到所需的多數票（如 1/2）
        """
        total_votes = len(votes)
        if total_votes == 0:
            return False
        
        positive_votes = sum(1 for vote in votes.values() if vote)
        return (positive_votes / total_votes) >= required_majority
    
    def to_json(self):
        """
        將區塊鏈轉換為 JSON 格式
        """
        chain_data = [block.to_dict() for block in self.chain]
        return json.dumps(chain_data, indent=4)
    
    def from_json(self, json_data):
        """
        從 JSON 數據中載入區塊鏈
        """
        chain_data = json.loads(json_data)
        self.chain = []
        for block_data in chain_data:
            block = Block(
                index=block_data['index'],
                round_num=block_data['round_num'],
                transactions=block_data['transactions'],
                timestamp=block_data['timestamp'],
                previous_hash=block_data['previous_hash'],
                votes=block_data.get('votes', {}),
                nonce=block_data.get('nonce', 0)
            )
            block.hash = block_data['hash']
            self.chain.append(block)
    
    def is_chain_valid(self, chain=None):
        """
        驗證整個區塊鏈的有效性
        """
        if chain is None:
            chain = self.chain
        
        for i in range(1, len(chain)):
            current = chain[i]
            previous = chain[i - 1]
            
            # 檢查當前區塊的哈希是否正確
            if current.hash != current.compute_hash():
                print(f"區塊 {current.index} 的哈希不正確。")
                return False
            
            # 檢查當前區塊的前一個哈希是否匹配上一個區塊的哈希
            if current.previous_hash != previous.hash:
                print(f"區塊 {current.index} 的 previous_hash 不匹配區塊 {previous.index} 的哈希。")
                return False
            
            # 檢查區塊輪數是否正確
            expected_round = previous.round_num + 1
            if current.round_num != expected_round:
                print(f"區塊 {current.index} 的輪數不正確。預期: {expected_round}, 實際: {current.round_num}")
                return False
            
            # 檢查投票結果是否達到共識
            if not self.has_consensus(current.votes):
                print(f"區塊 {current.index} 未達到共識。")
                return False
        
        return True
