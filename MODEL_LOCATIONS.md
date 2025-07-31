# 📁 模型文件位置指南

## 🎯 训练后模型存放位置

### 默认路径结构
```
basalt_competition_baseline_submissions/
├── models/                    # 所有训练好的模型
│   ├── game_ai_epoch_10.pth   # 每10个epoch的检查点
│   ├── game_ai_epoch_20.pth
│   ├── game_ai_epoch_30.pth
│   ├── game_ai_epoch_40.pth
│   └── game_ai_final.pth      # 最终完整模型
├── game_data/                 # 训练数据
│   ├── index.json            # 数据索引
│   ├── frame_000001.npy      # 屏幕截图
│   ├── action_000001.json    # 对应动作
│   └── ...                   # 更多数据
├── configs/                   # 配置文件
│   └── game_config.json      # 训练配置
└── logs/                      # 训练日志
    └── training.log
```

## 📋 模型文件说明

### 1. 检查点模型
- **路径**: `models/game_ai_epoch_{N}.pth`
- **内容**: 第N个epoch的完整模型状态
- **用途**: 恢复训练、选择最佳epoch

### 2. 最终模型
- **路径**: `models/game_ai_final.pth`
- **内容**: 训练完成后的最终模型
- **用途**: 部署到游戏环境

### 3. 模型文件内容
```python
{
    'model_state_dict': {...},    # 模型权重
    'config': {...},              # 训练配置
    'epoch': 50,                  # 训练轮数
    'loss': 0.00123,              # 最终损失
    'timestamp': '2025-07-31...'  # 训练时间
}
```

## 🚀 使用训练好的模型

### 加载模型
```python
from game_ai_trainer import GameAITrainer
import torch

# 加载配置
config = {
    'window_name': 'Your Game',
    'screen_width': 320,
    'screen_height': 240
}

# 创建训练器
trainer = GameAITrainer(config)

# 加载模型
trainer.load_model('models/game_ai_final.pth')

# 开始AI玩游戏
trainer.play(duration=300)  # 玩5分钟
```

### 命令行使用
```bash
# 直接运行训练好的AI
python game_ai_trainer.py --model models/game_ai_final.pth --play

# 指定模型文件
python play_game.py --model models/game_ai_epoch_50.pth --duration 600
```

## 🔄 模型管理

### 自动保存
- 每10个epoch自动保存检查点
- 训练完成后保存最终模型
- 覆盖旧文件前自动备份

### 手动保存
```python
# 在代码中手动保存
trainer.save_model('models/my_custom_model.pth')
```

### 模型选择
```bash
# 比较不同epoch的模型
ls -la models/
# 选择损失最小的模型
```

## 📊 模型大小
- **完整模型**: ~120MB (包含优化器状态)
- **仅权重**: ~45MB (仅模型参数)
- **压缩后**: ~35MB (zip格式)

## 🎯 最佳实践

### 1. 定期备份
```bash
# 备份重要模型
cp models/game_ai_final.pth models/game_ai_final_backup_$(date +%Y%m%d).pth
```

### 2. 版本管理
```bash
# 为不同游戏创建子目录
mkdir -p models/minecraft/
mkdir -p models/chess/
```

### 3. 清理旧模型
```bash
# 删除旧的检查点，保留最近3个
ls -t models/game_ai_epoch_*.pth | tail -n +4 | xargs rm
```

## 🔍 故障排除

### 找不到模型文件
```bash
# 检查模型目录
ls -la models/

# 重新训练
python game_ai_trainer.py --epochs 50
```

### 模型加载失败
```python
# 检查模型文件完整性
import torch
checkpoint = torch.load('models/game_ai_final.pth', map_location='cpu')
print(checkpoint.keys())
```

## 📋 文件权限
- **读取**: 所有用户可读
- **写入**: 仅当前用户可写
- **执行**: 无需执行权限

## 🎮 快速开始命令
```bash
# 1. 查看所有模型
ls models/

# 2. 使用最新模型
python game_ai_trainer.py --model models/game_ai_final.pth --play

# 3. 使用特定epoch模型
python game_ai_trainer.py --model models/game_ai_epoch_30.pth --play