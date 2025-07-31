#!/usr/bin/env python3
"""
游戏AI系统测试脚本
测试X11环境、依赖和基本功能
"""

import os
import sys
import time
import logging
import numpy as np
import cv2
from typing import Dict, List

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 测试配置
TEST_CONFIG = {
    'screen_width': 320,
    'screen_height': 240,
    'test_duration': 5,  # 秒
    'window_name': None  # 测试整个屏幕
}


class SystemTester:
    """系统测试器"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_total = 0
        self.results = {}
    
    def run_test(self, test_name: str, test_func):
        """运行单个测试"""
        self.tests_total += 1
        logger.info(f"🧪 开始测试: {test_name}")
        
        try:
            result = test_func()
            if result:
                self.tests_passed += 1
                logger.info(f"✅ {test_name}: 通过")
                self.results[test_name] = "PASS"
            else:
                logger.error(f"❌ {test_name}: 失败")
                self.results[test_name] = "FAIL"
        except Exception as e:
            logger.error(f"❌ {test_name}: 异常 - {e}")
            self.results[test_name] = f"ERROR: {e}"
    
    def test_python_version(self) -> bool:
        """测试Python版本"""
        version = sys.version_info
        logger.info(f"Python版本: {version.major}.{version.minor}.{version.micro}")
        return version >= (3, 12)
    
    def test_dependencies(self) -> bool:
        """测试依赖包"""
        required_packages = [
            'torch', 'numpy', 'cv2', 'Xlib', 'PIL', 'matplotlib'
        ]
        
        failed_packages = []
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package}: 已安装")
            except ImportError:
                logger.error(f"❌ {package}: 未安装")
                failed_packages.append(package)
        
        return len(failed_packages) == 0
    
    def test_x11_connection(self) -> bool:
        """测试X11连接"""
        try:
            from Xlib import display
            
            # 测试基本连接
            disp = display.Display()
            screen = disp.screen()
            logger.info(f"✅ X11连接成功")
            logger.info(f"屏幕尺寸: {screen.width_in_pixels}x{screen.height_in_pixels}")
            
            # 测试根窗口
            root = screen.root
            logger.info(f"根窗口ID: {root.id}")
            
            disp.close()
            return True
            
        except Exception as e:
            logger.error(f"X11连接失败: {e}")
            return False
    
    def test_screen_capture(self) -> bool:
        """测试屏幕捕获"""
        try:
            from x11_game_env import X11GameEnvironment
            
            # 创建环境
            env = X11GameEnvironment(
                width=TEST_CONFIG['screen_width'],
                height=TEST_CONFIG['screen_height']
            )
            
            # 测试截屏
            frame = env.get_screen_frame()
            
            if frame is None:
                logger.error("截屏返回None")
                return False
            
            expected_shape = (TEST_CONFIG['screen_height'], 
                            TEST_CONFIG['screen_width'], 3)
            
            if frame.shape != expected_shape:
                logger.error(f"截屏形状错误: {frame.shape}, 期望: {expected_shape}")
                return False
            
            logger.info(f"✅ 截屏成功，形状: {frame.shape}")
            logger.info(f"数据类型: {frame.dtype}")
            logger.info(f"像素范围: {frame.min()}-{frame.max()}")
            
            # 保存测试图像
            cv2.imwrite("test_capture.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            logger.info("测试图像已保存: test_capture.jpg")
            
            env.close()
            return True
            
        except Exception as e:
            logger.error(f"屏幕捕获测试失败: {e}")
            return False
    
    def test_mouse_control(self) -> bool:
        """测试鼠标控制"""
        try:
            from x11_game_env import X11GameEnvironment
            
            env = X11GameEnvironment(
                width=TEST_CONFIG['screen_width'],
                height=TEST_CONFIG['screen_height']
            )
            
            # 测试鼠标移动
            logger.info("测试鼠标移动...")
            env._move_mouse(10, 10)
            time.sleep(0.1)
            
            # 测试鼠标点击
            logger.info("测试鼠标点击...")
            env._click_mouse()
            time.sleep(0.1)
            
            env.close()
            logger.info("✅ 鼠标控制测试通过")
            return True
            
        except Exception as e:
            logger.error(f"鼠标控制测试失败: {e}")
            return False
    
    def test_keyboard_control(self) -> bool:
        """测试键盘控制"""
        try:
            from x11_game_env import X11GameEnvironment
            
            env = X11GameEnvironment(
                width=TEST_CONFIG['screen_width'],
                height=TEST_CONFIG['screen_height']
            )
            
            # 测试按键
            logger.info("测试键盘控制...")
            env._press_key('w')
            time.sleep(0.1)
            env._press_key('space')
            time.sleep(0.1)
            
            env.close()
            logger.info("✅ 键盘控制测试通过")
            return True
            
        except Exception as e:
            logger.error(f"键盘控制测试失败: {e}")
            return False
    
    def test_model_creation(self) -> bool:
        """测试模型创建"""
        try:
            import torch
            from game_ai_trainer import GameAI
            
            # 创建测试模型
            model = GameAI(
                input_shape=(3, TEST_CONFIG['screen_height'], TEST_CONFIG['screen_width']),
                action_dim=9
            )
            
            # 测试前向传播
            dummy_input = torch.randn(1, 3, TEST_CONFIG['screen_height'], TEST_CONFIG['screen_width'])
            output = model(dummy_input)
            
            expected_shape = (1, 9)
            if output.shape != expected_shape:
                logger.error(f"模型输出形状错误: {output.shape}, 期望: {expected_shape}")
                return False
            
            logger.info(f"✅ 模型创建成功，输出形状: {output.shape}")
            logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
            
            return True
            
        except Exception as e:
            logger.error(f"模型创建测试失败: {e}")
            return False
    
    def test_performance(self) -> bool:
        """测试性能"""
        try:
            from x11_game_env import X11GameEnvironment
            
            env = X11GameEnvironment(
                width=TEST_CONFIG['screen_width'],
                height=TEST_CONFIG['screen_height']
            )
            
            # 测试截屏速度
            start_time = time.time()
            frames = []
            
            for i in range(30):  # 测试30帧
                frame = env.get_screen_frame()
                frames.append(frame)
            
            elapsed = time.time() - start_time
            fps = len(frames) / elapsed
            
            logger.info(f"✅ 性能测试: {fps:.2f} FPS")
            logger.info(f"每帧耗时: {(elapsed/len(frames))*1000:.2f} ms")
            
            env.close()
            
            # 基本性能要求: >10 FPS
            return fps > 10
            
        except Exception as e:
            logger.error(f"性能测试失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("🚀 开始游戏AI系统测试...")
        logger.info("=" * 50)
        
        # 基础测试
        self.run_test("Python版本", self.test_python_version)
        self.run_test("依赖包", self.test_dependencies)
        self.run_test("X11连接", self.test_x11_connection)
        
        # 功能测试
        self.run_test("屏幕捕获", self.test_screen_capture)
        self.run_test("鼠标控制", self.test_mouse_control)
        self.run_test("键盘控制", self.test_keyboard_control)
        self.run_test("模型创建", self.test_model_creation)
        self.run_test("性能测试", self.test_performance)
        
        # 总结
        logger.info("=" * 50)
        logger.info("📊 测试结果总结:")
        
        for test_name, result in self.results.items():
            logger.info(f"  {test_name}: {result}")
        
        logger.info(f"\n🎯 通过率: {self.tests_passed}/{self.tests_total}")
        
        if self.tests_passed == self.tests_total:
            logger.info("🎉 所有测试通过！系统已就绪")
            return True
        else:
            logger.warning("⚠️  部分测试失败，请检查日志")
            return False


def main():
    """主函数"""
    tester = SystemTester()
    success = tester.run_all_tests()
    
    if success:
        logger.info("\n下一步:")
        logger.info("1. 运行: python setup_game_ai.py")
        logger.info("2. 编辑: configs/game_config.json")
        logger.info("3. 开始: python game_ai_trainer.py")
    else:
        logger.error("\n请修复失败的测试后再继续")
        sys.exit(1)


if __name__ == "__main__":
    main()