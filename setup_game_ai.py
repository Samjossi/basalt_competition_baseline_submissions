#!/usr/bin/env python3
"""
游戏AI设置脚本
一键配置Python3.12 + X11 + PyTorch环境
"""

import os
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameAISetup:
    """游戏AI环境设置器"""
    
    def __init__(self):
        self.system = self._detect_system()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
    def _detect_system(self) -> str:
        """检测操作系统"""
        import platform
        system = platform.system().lower()
        if system == "linux":
            # 检测是否为Ubuntu/Debian
            try:
                with open("/etc/os-release", "r") as f:
                    content = f.read()
                    if "ubuntu" in content.lower():
                        return "ubuntu"
                    elif "debian" in content.lower():
                        return "debian"
            except:
                pass
            return "linux"
        elif system == "darwin":
            return "macos"
        else:
            return system
    
    def check_python_version(self) -> bool:
        """检查Python版本"""
        if sys.version_info < (3, 12):
            logger.error(f"需要Python 3.12+，当前版本: {self.python_version}")
            return False
        logger.info(f"Python版本检查通过: {self.python_version}")
        return True
    
    def install_system_dependencies(self):
        """安装系统依赖"""
        logger.info("安装系统依赖...")
        
        if self.system in ["ubuntu", "debian"]:
            cmd = [
                "sudo", "apt", "update",
                "&&", "sudo", "apt", "install", "-y",
                "python3-dev",
                "python3-pip",
                "libx11-dev",
                "libxtst-dev",
                "libxext-dev",
                "libxrandr-dev",
                "libxinerama-dev",
                "libxcursor-dev",
                "libxi-dev",
                "libgl1-mesa-glx",
                "libglib2.0-0"
            ]
        elif self.system == "macos":
            cmd = [
                "brew", "install",
                "python@3.12",
                "libx11",
                "libxtst"
            ]
        else:
            logger.warning(f"不支持的操作系统: {self.system}")
            return
        
        try:
            subprocess.run(" ".join(cmd), shell=True, check=True)
            logger.info("系统依赖安装完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"系统依赖安装失败: {e}")
    
    def install_python_dependencies(self):
        """安装Python依赖"""
        logger.info("安装Python依赖...")
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True)
            logger.info("Python依赖安装完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"Python依赖安装失败: {e}")
    
    def setup_x11_permissions(self):
        """设置X11权限"""
        logger.info("设置X11权限...")
        
        # 检查DISPLAY环境变量
        display = os.environ.get("DISPLAY")
        if not display:
            logger.error("未找到DISPLAY环境变量，请确保在X11环境下运行")
            return
        
        # 设置xhost权限
        try:
            subprocess.run(["xhost", "+local:"], check=True)
            logger.info("X11权限设置完成")
        except subprocess.CalledProcessError:
            logger.warning("xhost命令失败，可能需要手动设置权限")
    
    def test_environment(self):
        """测试环境"""
        logger.info("测试环境...")
        
        try:
            # 测试X11连接
            from Xlib import display
            disp = display.Display()
            disp.close()
            logger.info("X11连接测试通过")
            
            # 测试OpenCV
            import cv2
            logger.info("OpenCV测试通过")
            
            # 测试PyTorch
            import torch
            logger.info(f"PyTorch测试通过，版本: {torch.__version__}")
            
            # 测试Xlib
            from Xlib import display
            logger.info("Xlib测试通过")
            
            return True
            
        except ImportError as e:
            logger.error(f"依赖测试失败: {e}")
            return False
    
    def create_directories(self):
        """创建必要的目录"""
        directories = [
            "game_data",
            "models",
            "logs",
            "configs"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"创建目录: {directory}")
    
    def generate_config(self):
        """生成配置文件"""
        config = {
            "window_name": "Your Game Window Name",
            "screen_width": 320,
            "screen_height": 240,
            "batch_size": 16,
            "learning_rate": 1e-4,
            "epochs": 50,
            "data_collection_duration": 300,  # 秒
            "training_interval": 10  # 每N个样本训练一次
        }
        
        with open("configs/game_config.json", "w") as f:
            import json
            json.dump(config, f, indent=2)
        
        logger.info("配置文件已生成: configs/game_config.json")
    
    def run_setup(self):
        """运行完整设置"""
        logger.info("开始设置游戏AI环境...")
        
        # 检查Python版本
        if not self.check_python_version():
            return False
        
        # 安装系统依赖
        self.install_system_dependencies()
        
        # 安装Python依赖
        self.install_python_dependencies()
        
        # 设置X11权限
        self.setup_x11_permissions()
        
        # 创建目录
        self.create_directories()
        
        # 生成配置
        self.generate_config()
        
        # 测试环境
        if self.test_environment():
            logger.info("✅ 游戏AI环境设置完成！")
            logger.info("下一步:")
            logger.info("1. 编辑 configs/game_config.json 配置你的游戏")
            logger.info("2. 运行 python game_ai_trainer.py 开始训练")
            return True
        else:
            logger.error("❌ 环境测试失败")
            return False


def main():
    """主函数"""
    setup = GameAISetup()
    setup.run_setup()


if __name__ == "__main__":
    main()