#!/usr/bin/env python3
"""
自定义游戏AI训练器
使用X11截屏 + 行为克隆训练
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import json
import time
from typing import List, Dict, Tuple
import logging
from x11_game_env import X11GameEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameDataset(Dataset):
    """游戏数据集"""
    
    def __init__(self, data_dir: str = "game_data"):
        self.data_dir = data_dir
        self.samples = []
        
        # 加载数据索引
        index_file = os.path.join(data_dir, "index.json")
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                self.samples = json.load(f)
    
    def add_sample(self, frame: np.ndarray, action: Dict) -> None:
        """添加样本"""
        sample_id = len(self.samples)
        frame_path = os.path.join(self.data_dir, f"frame_{sample_id:06d}.npy")
        action_path = os.path.join(self.data_dir, f"action_{sample_id:06d}.json")
        
        # 保存数据
        np.save(frame_path, frame)
        with open(action_path, 'w') as f:
            json.dump(action, f)
        
        # 更新索引
        self.samples.append({
            'frame': frame_path,
            'action': action_path
        })
        
        # 保存索引
        with open(os.path.join(self.data_dir, "index.json"), 'w') as f:
            json.dump(self.samples, f)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame = np.load(sample['frame'])
        with open(sample['action'], 'r') as f:
            action = json.load(f)
        
        # 预处理
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
        
        # 动作向量化
        action_vec = self._action_to_vector(action)
        
        return torch.FloatTensor(frame), torch.FloatTensor(action_vec)
    
    def _action_to_vector(self, action: Dict) -> np.ndarray:
        """将动作转换为向量"""
        vec = []
        
        # 鼠标移动
        vec.extend([
            action.get('mouse_dx', 0.0),
            action.get('mouse_dy', 0.0)
        ])
        
        # 鼠标点击
        vec.append(float(action.get('mouse_click', 0.0)))
        
        # 按键状态
        keys = action.get('keys', [])
        for key in ['w', 'a', 's', 'd', 'space', 'shift']:
            vec.append(1.0 if key in keys else 0.0)
        
        return np.array(vec, dtype=np.float32)


class GameAI(nn.Module):
    """游戏AI模型"""
    
    def __init__(self, input_shape: Tuple[int, int, int], action_dim: int):
        super().__init__()
        self.input_shape = input_shape
        self.action_dim = action_dim
        
        # CNN特征提取
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 计算CNN输出维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            cnn_output_dim = self.conv_layers(dummy_input).shape[1]
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(cnn_output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, x):
        features = self.conv_layers(x)
        return self.fc_layers(features)


class GameAITrainer:
    """游戏AI训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.env: X11GameEnvironment
        self.model: GameAI
        self.dataset: GameDataset
        
        # 配置
        self.window_name = config.get('window_name')
        self.screen_width = config.get('screen_width', 640)
        self.screen_height = config.get('screen_height', 480)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.epochs = config.get('epochs', 100)
        
        # 创建目录
        os.makedirs("game_data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # 初始化
        self._setup_environment()
        self._setup_model()
    
    def _setup_environment(self):
        """设置环境"""
        self.env = X11GameEnvironment(
            window_name=self.window_name,
            width=self.screen_width,
            height=self.screen_height
        )
        logger.info("环境设置完成")
    
    def _setup_model(self):
        """设置模型"""
        input_shape = self.env.get_observation_space()
        action_dim = 9  # 2(mouse) + 1(click) + 6(keys)
        
        self.model = GameAI(input_shape, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        logger.info(f"模型设置完成，输入形状: {input_shape}, 动作维度: {action_dim}")
    
    def collect_data(self, num_samples: int = 1000):
        """收集训练数据"""
        logger.info(f"开始收集 {num_samples} 个样本...")
        
        # 这里需要用户实现数据收集逻辑
        # 可以是人工操作记录，也可以是其他方式
        
        # 示例：随机动作收集
        for i in range(num_samples):
            frame = self.env.get_screen_frame()
            
            # 生成随机动作（实际应用中应该是人工操作）
            action = {
                'mouse_dx': np.random.uniform(-1, 1),
                'mouse_dy': np.random.uniform(-1, 1),
                'mouse_click': np.random.choice([0, 1]),
                'keys': np.random.choice(['w', 'a', 's', 'd', 'space', 'shift'], 
                                       size=np.random.randint(0, 3), 
                                       replace=False).tolist()
            }
            
            # 保存样本
            if not hasattr(self, 'dataset') or self.dataset is None:
                self.dataset = GameDataset()
            
            self.dataset.add_sample(frame, action)
            
            if i % 100 == 0:
                logger.info(f"已收集 {i}/{num_samples} 个样本")
    
    def train(self):
        """训练模型"""
        if not self.dataset or len(self.dataset) == 0:
            logger.error("没有训练数据，请先收集数据")
            return
        
        logger.info("开始训练...")
        
        dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0  # 避免X11多进程问题
        )
        
        self.model.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            batch_count = 0
            
            for frames, actions in dataloader:
                self.optimizer.zero_grad()
                
                predictions = self.model(frames)
                loss = self.criterion(predictions, actions)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / batch_count
            logger.info(f"Epoch {epoch+1}/{self.epochs}, 平均损失: {avg_loss:.6f}")
            
            # 保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_model(f"models/game_ai_epoch_{epoch+1}.pth")
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"模型已保存: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"模型已加载: {path}")
    
    def play(self, duration: int = 60):
        """让AI玩游戏"""
        logger.info("AI开始玩游戏...")
        
        self.model.eval()
        start_time = time.time()
        
        with torch.no_grad():
            while time.time() - start_time < duration:
                # 获取当前屏幕
                frame = self.env.get_screen_frame()
                
                # 预处理
                frame_tensor = torch.FloatTensor(frame).unsqueeze(0)
                frame_tensor = frame_tensor.permute(0, 3, 1, 2) / 255.0
                
                # 预测动作
                action_pred = self.model(frame_tensor)
                action = self._vector_to_action(action_pred[0].numpy())
                
                # 执行动作
                self.env.step(action)
                
                # 控制频率
                time.sleep(0.1)
    
    def _vector_to_action(self, vector: np.ndarray) -> Dict:
        """将向量转换为动作"""
        return {
            'mouse_dx': vector[0],
            'mouse_dy': vector[1],
            'mouse_click': vector[2] > 0.5,
            'keys': [key for i, key in enumerate(['w', 'a', 's', 'd', 'space', 'shift']) 
                    if vector[3+i] > 0.5]
        }


def main():
    """主函数"""
    config = {
        'window_name': None,  # 捕获整个屏幕
        'screen_width': 320,
        'screen_height': 240,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'epochs': 50
    }
    
    trainer = GameAITrainer(config)
    
    # 收集数据
    trainer.collect_data(num_samples=100)
    
    # 训练
    trainer.train()
    
    # 保存最终模型
    trainer.save_model("models/game_ai_final.pth")
    
    # 测试
    trainer.load_model("models/game_ai_final.pth")
    trainer.play(duration=30)


if __name__ == "__main__":
    main()