# 🎮 自定义游戏AI训练系统

基于X11的实时游戏AI训练框架，支持任何Linux游戏通过屏幕捕获和行为克隆进行学习。

## 🚀 特性

- ✅ **Python 3.12** 完全兼容
- 🖥️ **X11截屏** 使用Xlib直接捕获游戏画面
- 🎯 **行为克隆** 模仿学习算法
- ⚡ **实时训练** 边玩边学
- 🔧 **零侵入** 无需修改游戏代码
- 📊 **可视化监控** TensorBoard集成

## 📋 快速开始

### 1. 一键安装

```bash
# 克隆项目
git clone <your-repo>
cd game-ai-trainer

# 一键设置环境
python setup_game_ai.py
```

### 2. 配置游戏

编辑 `configs/game_config.json`：

```json
{
  "window_name": "Your Game Window Name",
  "screen_width": 320,
  "screen_height": 240,
  "batch_size": 16,
  "learning_rate": 0.0001,
  "epochs": 50
}
```

### 3. 收集训练数据

```bash
# 启动数据收集（人工操作）
python collect_data.py --duration 300 --output game_data/

# 或使用录制工具
python record_gameplay.py --window "Game Window" --duration 600
```

### 4. 训练AI

```bash
# 开始训练
python game_ai_trainer.py

# 使用TensorBoard监控
tensorboard --logdir logs/
```

### 5. 让AI玩游戏

```bash
# 测试训练好的模型
python play_game.py --model models/game_ai_final.pth --duration 300
```

## 🏗️ 架构

```
game-ai-trainer/
├── x11_game_env.py      # X11游戏环境
├── game_ai_trainer.py   # 训练器主程序
├── setup_game_ai.py     # 环境设置
├── collect_data.py      # 数据收集
├── play_game.py         # AI游戏测试
├── requirements.txt     # Python依赖
├── configs/            # 配置文件
├── game_data/          # 训练数据
├── models/             # 训练好的模型
└── logs/               # 训练日志
```

## 🔧 技术细节

### 观察空间
- **输入**: RGB屏幕截图 (320x240x3)
- **预处理**: 归一化到[0,1]，CHW格式

### 动作空间
- **鼠标**: X/Y移动 (-1.0到1.0)，点击 (0/1)
- **键盘**: WASD、空格、Shift、数字键1-5
- **总维度**: 9维连续动作向量

### 模型架构
```python
CNN特征提取:
- Conv2d(3, 32, 8, stride=4)
- Conv2d(32, 64, 4, stride=2)
- Conv2d(64, 64, 3, stride=1)
- Flatten -> FC(512) -> FC(256) -> FC(9)
```

## 🎯 支持的游戏

理论上支持**任何Linux游戏**，包括：
- 🕹️ Steam游戏
- 🎮 原生Linux游戏
- 🍷 Wine游戏
- 🌐 浏览器游戏
- 📱 Android模拟器游戏

## 📊 性能优化

### 延迟优化
- **XShm**: 共享内存截屏
- **异步处理**: 并行数据加载
- **批处理**: 批量推理

### 内存优化
- **流式数据**: 不加载全部数据到内存
- **缓存策略**: LRU缓存常用帧
- **垃圾回收**: 及时清理不用的数据

## 🛠️ 开发指南

### 添加新游戏
1. 找到游戏窗口名称: `xwininfo -tree -root`
2. 更新配置文件
3. 调整观察/动作空间

### 自定义奖励函数
```python
def custom_reward_function(prev_state, current_state, action):
    # 根据游戏状态计算奖励
    reward = 0.0
    
    # 示例：基于分数变化
    if current_state.score > prev_state.score:
        reward = current_state.score - prev_state.score
    
    return reward
```

### 扩展动作空间
```python
# 在x11_game_env.py中添加新按键
self.key_mapping.update({
    'e': 26,  # 添加E键
    'r': 27   # 添加R键
})
```

## 🔍 故障排除

### 常见问题

#### X11权限错误
```bash
# 解决方案
xhost +local:
export DISPLAY=:0
```

#### 游戏窗口找不到
```bash
# 列出所有窗口
python -c "from x11_game_env import X11GameEnvironment; X11GameEnvironment()._find_window('')"
```

#### 性能问题
- 降低分辨率: `screen_width/height`
- 减少批大小: `batch_size`
- 使用GPU: `CUDA_VISIBLE_DEVICES=0`

## 📈 监控和调试

### TensorBoard指标
- **训练损失**: 行为克隆损失
- **验证损失**: 泛化能力
- **动作分布**: 动作使用频率
- **帧率**: 实时性能

### 日志文件
- `logs/training.log`: 训练日志
- `logs/gameplay.log`: 游戏日志
- `logs/performance.log`: 性能日志

## 🤝 贡献

欢迎提交Issue和PR！

### 开发环境
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/

# 代码格式化
black *.py
```

## 📄 许可证

MIT License - 详见LICENSE文件