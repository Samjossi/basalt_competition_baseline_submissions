# UV虚拟环境管理指南

## 🚀 使用UV管理项目环境

### 安装UV
```bash
# 安装uv（跨平台）
pip install uv

# 或者使用curl安装（Linux/macOS）
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 创建和管理虚拟环境

#### 方法1：使用UV直接管理
```bash
# 创建虚拟环境（自动使用当前目录Python版本）
uv venv

# 激活虚拟环境
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 安装依赖
uv pip install -r requirements.txt

# 升级包
uv pip install --upgrade torch torchvision
```

#### 方法2：使用pyproject.toml（推荐）
创建`pyproject.toml`文件：
```toml
[project]
name = "custom-game-ai"
version = "0.1.0"
description = "自定义游戏AI学习系统"
requires-python = ">=3.8"
dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.24.0",
    "gym>=0.26.0",
    "opencv-python>=4.7.0",
    "matplotlib>=3.6.0",
    "pandas>=1.5.0",
    "pillow>=9.0.0",
    "tensorboard>=2.12.0",
    "tqdm>=4.64.0",
    "rich>=13.0.0",
    "pyyaml>=6.0",
    "click>=8.0.0",
    "joblib>=1.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=5.0.0",
]

[tool.black]
line-length = 88
target-version = ['py38']

[tool.isort]
profile = "black"
```

### 使用UV安装依赖
```bash
# 安装所有依赖
uv sync

# 安装开发依赖
uv sync --extra dev

# 添加新包
uv add torch torchvision
uv add --dev pytest

# 运行脚本
uv run python train.py

# 运行测试
uv run pytest tests/
```

### 环境管理命令
```bash
# 查看已安装包
uv pip list

# 导出环境
uv pip freeze > requirements.txt

# 更新所有包
uv pip install --upgrade-all

# 删除虚拟环境
rm -rf .venv  # Linux/macOS
# 或
rmdir /s .venv  # Windows
```

### 项目结构建议
```
custom-game-ai/
├── pyproject.toml          # UV项目配置
├── requirements.txt        # 传统依赖文件（兼容）
├── uv.lock                # UV锁文件
├── src/
│   ├── __init__.py
│   ├── main.py
│   └── ...
├── .venv/                 # UV虚拟环境
└── scripts/
    ├── setup_env.sh       # 环境设置脚本
    └── run_training.sh    # 训练脚本
```

### 常用脚本
```bash
# setup_env.sh
#!/bin/bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
echo "环境设置完成！"

# run_training.sh
#!/bin/bash
source .venv/bin/activate
uv run python src/train.py "$@"
```

### 与现有工具对比
| 工具 | 速度 | 功能 | 备注 |
|------|------|------|------|
| **UV** | ⚡超快 | 🚀现代化 | 推荐 |
| pip | 中等 | 传统 | 兼容性好 |
| conda | 慢 | 完整生态 | 科学计算 |
| poetry | 中等 | 项目管理 | 依赖解析 |

### 故障排除
```bash
# 如果uv命令未找到
export PATH="$HOME/.local/bin:$PATH"

# 检查Python版本
uv python list

# 指定Python版本
uv venv --python 3.12

# 清理缓存
uv cache clean
```

### 迁移指南（从conda/docker）
```bash
# 从conda迁移
conda activate old_env
pip freeze > old_requirements.txt
uv venv
uv pip install -r old_requirements.txt

# 从docker迁移
# 1. 导出docker镜像包列表
# 2. 创建requirements.txt
# 3. 使用uv安装
```

### 性能优化
```bash
# 使用更快的镜像源
uv pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 并行安装
uv pip install -r requirements.txt --parallel
```

现在你可以使用UV来管理这个AI学习项目的所有依赖！