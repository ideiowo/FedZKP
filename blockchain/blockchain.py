import hashlib
import json
from datetime import datetime

class Block:
    def __init__(self, index, round_num, aggregated_gradients_hash, verification_proof_hash, consensus_votes, timestamp, previous_hash, nonce=0):
        self.index = index
        self.round_num = round_num
        self.aggregated_gradients_hash = aggregated_gradients_hash
        self.verification_proof_hash = verification_proof_hash
        self.consensus_votes = consensus_votes
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = json.dumps({
            'index': self.index,
            'round_num': self.round_num,
            'aggregated_gradients_hash': self.aggregated_gradients_hash,
            'verification_proof_hash': self.verification_proof_hash,
            'consensus_votes': self.consensus_votes,
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    def to_dict(self):
        return {
            'index': self.index,
            'round_num': self.round_num,
            'aggregated_gradients_hash': self.aggregated_gradients_hash,
            'verification_proof_hash': self.verification_proof_hash,
            'consensus_votes': self.consensus_votes,
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
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
            round_num=0,
            aggregated_gradients_hash="0",  # 初始值
            verification_proof_hash="0",  # 初始值
            consensus_votes={},  # 創世區塊無投票
            timestamp=str(datetime.now()),
            previous_hash="0"
        )
        self.chain.append(genesis_block)
    
    @property
    def last_block(self):
        return self.chain[-1]
    
    def add_block(self, aggregated_gradients_hash, verification_proof_hash, consensus_votes):
        """
        在共識達成後創建並添加新區塊
        :param aggregated_gradients_hash: 經過聚合的梯度或模型參數的哈希值
        :param verification_proof_hash: 驗證證明或文件摘要
        :param consensus_votes: 副本節點的驗證結果
        :param training_metrics: 訓練指標
        """
        new_block = Block(
            index=len(self.chain),
            round_num=self.current_round + 1,
            aggregated_gradients_hash=aggregated_gradients_hash,
            verification_proof_hash=verification_proof_hash,
            consensus_votes=consensus_votes,
            timestamp=str(datetime.now()),
            previous_hash=self.last_block.hash
        )

        # 驗證並添加區塊
        if self.is_valid_block(new_block):
            self.chain.append(new_block)
            self.current_round += 1  # 更新輪數
            print(f"區塊 {new_block.index} 添加成功。")
            return True
        else:
            print(f"區塊 {new_block.index} 添加失敗。")
            return False
    
    def is_valid_block(self, block):
        """
        驗證區塊的有效性，包括哈希值和共識結果
        """
        if block.hash != block.compute_hash():
            print(f"區塊 {block.index} 的哈希不正確。")
            return False
        
        expected_round = self.last_block.round_num + 1
        if block.round_num != expected_round:
            print(f"區塊 {block.index} 的輪數不正確。預期: {expected_round}, 實際: {block.round_num}")
            return False
        
        if not self.has_consensus(block.consensus_votes):
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
