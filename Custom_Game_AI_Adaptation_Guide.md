# 自定义游戏世界AI适配指南

## 🎯 项目概述

基于MineRL BASALT基线项目，将其适配到**自定义游戏世界**的完整指南。保留核心的行为克隆学习机制，替换Minecraft特定接口。

## 🔍 当前项目结构分析

### 核心依赖分析
- **MineRL接口**：`minerl.data.make()` 和 `gym.make()` - 需要替换
- **观察空间**：Minecraft特定POV画面 - 需要抽象化
- **动作空间**：Minecraft动作（移动、攻击、使用等）- 需要泛化
- **数据格式**：MineRL数据集格式 - 需要适配

### 可复用组件
- ✅ **行为克隆算法**：`BC`类实现
- ✅ **神经网络架构**：CNN+LSTM结构
- ✅ **训练框架**：Sacred实验管理
- ✅ **评估系统**：指标计算和可视化
- ✅ **数据处理**：批处理和迭代器

## 🏗️ 适配架构设计

### 抽象接口层
```python
# custom_game_interface.py
class CustomGameEnvironment:
    """自定义游戏环境接口"""
    
    def __init__(self, game_config):
        self.game_config = game_config
        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space()
    
    def reset(self):
        """重置游戏状态"""
        pass
    
    def step(self, action):
        """执行动作并返回新状态"""
        pass
    
    def get_observation(self):
        """获取当前观察"""
        pass
    
    def is_done(self):
        """检查游戏是否结束"""
        pass
```

### 观察空间适配
```python
# observation_adapter.py
class GameObservationAdapter:
    """将自定义游戏观察转换为标准格式"""
    
    def __init__(self, game_type):
        self.game_type = game_type
        
    def adapt_observation(self, raw_obs):
        """适配原始观察到标准格式"""
        return {
            'visual': self._process_visual(raw_obs.get('screen', [])),
            'state': self._process_game_state(raw_obs),
            'inventory': self._process_inventory(raw_obs.get('items', {}))
        }
    
    def _process_visual(self, screen_data):
        """处理视觉信息"""
        # 统一为64x64x3格式
        return np.array(screen_data).reshape(64, 64, 3)
    
    def _process_game_state(self, obs):
        """处理游戏状态"""
        return np.array([
            obs.get('health', 100),
            obs.get('energy', 100),
            obs.get('position_x', 0),
            obs.get('position_y', 0),
            obs.get('time', 0)
        ])
```

### 动作空间适配
```python
# action_adapter.py
class GameActionAdapter:
    """将标准动作映射到自定义游戏动作"""
    
    def __init__(self, game_type):
        self.action_mapping = self._create_action_mapping(game_type)
    
    def map_action(self, standard_action):
        """将标准动作映射到游戏特定动作"""
        return self.action_mapping[standard_action]
    
    def _create_action_mapping(self, game_type):
        """根据游戏类型创建动作映射"""
        if game_type == "survival":
            return {
                0: {"move": "forward"},
                1: {"move": "backward"},
                2: {"move": "left"},
                3: {"move": "right"},
                4: {"action": "interact"},
                5: {"action": "attack"},
                6: {"action": "craft", "item": "wood"},
                7: {"action": "craft", "item": "stone"}
            }
        # 可以扩展其他游戏类型
```

## 🔄 迁移步骤

### 第1步：环境接口替换
```python
# 替换前（MineRL）
import minerl
env = gym.make('MineRLBasaltFindCave-v0')

# 替换后（自定义游戏）
from custom_game_env import CustomGameEnv
env = CustomGameEnv(game_config)
```

### 第2步：数据收集适配
```python
# 自定义数据收集器
class CustomDataCollector:
    def __init__(self, game_env):
        self.game_env = game_env
        
    def collect_demonstrations(self, num_episodes=100):
        """收集人类演示数据"""
        demonstrations = []
        for episode in range(num_episodes):
            trajectory = self._record_episode()
            demonstrations.append(trajectory)
        return demonstrations
    
    def _record_episode(self):
        """记录单个游戏episode"""
        states, actions, rewards = [], [], []
        obs = self.game_env.reset()
        
        while not self.game_env.is_done():
            # 记录人类操作
            human_action = self._get_human_input()
            next_obs, reward = self.game_env.step(human_action)
            
            states.append(obs)
            actions.append(human_action)
            rewards.append(reward)
            
            obs = next_obs
            
        return {"states": states, "actions": actions, "rewards": rewards}
```

### 第3步：训练管道适配
```python
# 适配后的训练脚本
class CustomGameTrainer:
    def __init__(self, game_env, model_config):
        self.game_env = game_env
        self.model = self._build_model(model_config)
        
    def train(self, demonstrations):
        """使用自定义游戏数据训练"""
        # 复用原有的BC训练逻辑
        bc_trainer = BC(
            observation_space=self.game_env.observation_space,
            action_space=self.game_env.action_space,
            policy_class=self.model,
            expert_data=demonstrations
        )
        bc_trainer.train()
```

## 🎮 游戏类型适配示例

### 示例1：2D生存游戏
```python
class Survival2DGame(CustomGameEnvironment):
    def __init__(self):
        super().__init__({
            "screen_size": (64, 64),
            "actions": ["move", "collect", "craft", "build"],
            "resources": ["wood", "stone", "food"]
        })
```

### 示例2：3D探索游戏
```python
class Exploration3DGame(CustomGameEnvironment):
    def __init__(self):
        super().__init__({
            "screen_size": (128, 128),
            "actions": ["move", "rotate", "interact", "inventory"],
            "features": ["health", "energy", "position", "inventory"]
        })
```

### 示例3：策略建造游戏
```python
class BuildingStrategyGame(CustomGameEnvironment):
    def __init__(self):
        super().__init__({
            "screen_size": (256, 256),
            "actions": ["select", "place", "remove", "upgrade"],
            "resources": ["materials", "population", "gold"]
        })
```

## 📊 兼容性检查清单

### ✅ 无需修改的组件
- [ ] 神经网络架构（CNN+LSTM）
- [ ] 训练算法（行为克隆）
- [ ] 评估指标系统
- [ ] 日志和可视化
- [ ] 模型保存和加载

### ⚠️ 需要适配的组件
- [ ] 环境接口（gym.Env）
- [ ] 观察空间定义
- [ ] 动作空间定义
- [ ] 数据加载器
- [ ] 奖励函数

### 🔧 新增组件
- [ ] 自定义游戏接口
- [ ] 游戏特定观察适配器
- [ ] 游戏特定动作映射器
- [ ] 数据收集工具
- [ ] 游戏状态管理器

## 🚀 快速开始（自定义游戏）

### 第1步：创建游戏环境
```python
# 创建你的游戏环境
from my_game import MyGameEnv
game_env = MyGameEnv()

# 适配到标准接口
from adapters import GameAdapter
adapter = GameAdapter(game_env)
```

### 第2步：收集训练数据
```python
# 收集人类演示
collector = CustomDataCollector(adapter)
demonstrations = collector.collect_demonstrations(num_episodes=50)
```

### 第3步：训练AI
```python
# 使用现有训练框架
trainer = CustomGameTrainer(adapter)
trainer.train(demonstrations)
```

### 第4步：评估AI
```python
# 评估训练结果
evaluator = GameEvaluator(adapter)
results = evaluator.evaluate(trained_model)
```

## 💡 设计建议

### 观察空间设计原则
1. **视觉信息**：统一为固定大小的RGB图像
2. **状态信息**：标准化为数值向量
3. **游戏特定**：通过适配器处理差异

### 动作空间设计原则
1. **离散动作**：便于行为克隆学习
2. **有限范围**：控制在8-16个动作内
3. **语义清晰**：每个动作有明确含义

### 数据格式标准
```json
{
    "observation": {
        "visual": [64, 64, 3],
        "state": [10],
        "game_info": {}
    },
    "action": 0-15,
    "reward": float,
    "done": bool
}
```

## 🔗 下一步行动

1. **选择游戏类型**：确定你的自定义游戏类型
2. **设计观察空间**：定义游戏画面和状态
3. **设计动作空间**：定义AI可以执行的动作
4. **实现环境接口**：创建gym.Env兼容的环境
5. **测试数据收集**：验证人类演示收集
6. **训练AI模型**：使用现有框架训练

这个框架让你能够将强大的行为克隆学习算法应用到任何自定义游戏中，而不仅仅是Minecraft！