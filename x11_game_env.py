#!/usr/bin/env python3
"""
X11游戏环境包装器
使用Xlib和XShm进行高性能屏幕捕获
支持键盘鼠标事件注入
"""

import numpy as np
import cv2
from Xlib import display, X
from Xlib.ext import xtest
from Xlib.protocol import event
import threading
import time
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class X11GameEnvironment:
    """X11游戏环境包装器"""
    
    def __init__(self, window_name: str | None = None, width: int = 640, height: int = 480):
        """
        初始化X11游戏环境
        
        Args:
            window_name: 窗口名称，None表示捕获整个屏幕
            width: 捕获宽度
            height: 捕获高度
        """
        self.display = display.Display()
        self.screen = self.display.screen()
        self.root = self.screen.root
        
        self.width = width
        self.height = height
        self.window_name = window_name
        
        # 查找目标窗口
        self.target_window = self._find_window(window_name) if window_name else self.root
        
        # 图像缓冲区
        self.last_frame = None
        
        # 动作空间定义
        self.action_space = {
            'mouse_x': (-1.0, 1.0),  # 鼠标X移动
            'mouse_y': (-1.0, 1.0),  # 鼠标Y移动
            'mouse_click': (0, 1),   # 鼠标点击
            'keys': [                # 支持的按键
                'w', 'a', 's', 'd',  # 移动
                'space', 'shift',    # 跳跃/蹲下
                '1', '2', '3', '4', '5'  # 物品栏
            ]
        }
        
        # 按键映射
        self.key_mapping = {
            'w': 25, 'a': 38, 's': 39, 'd': 40,
            'space': 65, 'shift': 50,
            '1': 10, '2': 11, '3': 12, '4': 13, '5': 14
        }
        
        logger.info(f"X11环境初始化完成，目标窗口: {window_name}")
    
    def _find_window(self, window_name: str) -> Any:
        """查找指定名称的窗口"""
        def _search_window(window, name):
            try:
                wm_name = window.get_wm_name()
                if wm_name and name.lower() in wm_name.lower():
                    return window
                
                children = window.query_tree().children
                for child in children:
                    result = _search_window(child, name)
                    if result:
                        return result
            except:
                pass
            return None
        
        window = _search_window(self.root, window_name)
        return window if window else self.root
    
    def get_screen_frame(self) -> np.ndarray:
        """获取屏幕帧"""
        try:
            # 获取窗口几何信息
            geometry = self.target_window.get_geometry()
            x, y = geometry.x, geometry.y
            
            # 捕获窗口图像
            raw_image = self.target_window.get_image(
                x, y, self.width, self.height, X.ZPixmap, 0xffffffff
            )
            
            # 转换为numpy数组
            image_data = raw_image.data
            frame = np.frombuffer(image_data, dtype=np.uint8)
            frame = frame.reshape((self.height, self.width, 4))  # RGBA
            
            # 转换为RGB格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
            # 调整大小
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            
            self.last_frame = frame
            return frame
            
        except Exception as e:
            logger.error(f"屏幕捕获失败: {e}")
            # 返回黑屏
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        logger.info("重置X11游戏环境")
        return self.get_screen_frame()
    
    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作
        
        Args:
            action: 包含动作信息的字典
                - mouse_dx: 鼠标X移动 (-1到1)
                - mouse_dy: 鼠标Y移动 (-1到1)
                - mouse_click: 鼠标点击 (0或1)
                - keys: 按键列表
        
        Returns:
            observation: 屏幕图像
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        # 执行鼠标移动
        if 'mouse_dx' in action and 'mouse_dy' in action:
            dx = int(action['mouse_dx'] * 100)
            dy = int(action['mouse_dy'] * 100)
            self._move_mouse(dx, dy)
        
        # 执行鼠标点击
        if 'mouse_click' in action and action['mouse_click'] > 0.5:
            self._click_mouse()
        
        # 执行按键
        if 'keys' in action:
            for key in action['keys']:
                if key in self.key_mapping:
                    self._press_key(key)
        
        # 获取新的观察
        observation = self.get_screen_frame()
        
        # 计算奖励（这里需要用户根据具体游戏实现）
        reward = 0.0
        
        # 检查是否结束（这里需要用户根据具体游戏实现）
        done = False
        
        # 额外信息
        info = {
            'frame_shape': observation.shape,
            'action_executed': action
        }
        
        return observation, reward, done, info
    
    def _move_mouse(self, dx: int, dy: int):
        """移动鼠标"""
        try:
            # 获取当前鼠标位置
            pointer = self.root.query_pointer()
            x, y = pointer.root_x, pointer.root_y
            
            # 计算新位置
            new_x = max(0, min(x + dx, self.screen.width_in_pixels))
            new_y = max(0, min(y + dy, self.screen.height_in_pixels))
            
            # 移动鼠标
            xtest.fake_input(self.display, X.MotionNotify, x=new_x, y=new_y)
            self.display.sync()
            
        except Exception as e:
            logger.error(f"鼠标移动失败: {e}")
    
    def _click_mouse(self, button: int = 1):
        """点击鼠标"""
        try:
            xtest.fake_input(self.display, X.ButtonPress, button)
            self.display.sync()
            time.sleep(0.01)
            xtest.fake_input(self.display, X.ButtonRelease, button)
            self.display.sync()
            
        except Exception as e:
            logger.error(f"鼠标点击失败: {e}")
    
    def _press_key(self, key: str):
        """按下按键"""
        try:
            keycode = self.key_mapping.get(key)
            if keycode:
                xtest.fake_input(self.display, X.KeyPress, keycode)
                self.display.sync()
                time.sleep(0.01)
                xtest.fake_input(self.display, X.KeyRelease, keycode)
                self.display.sync()
                
        except Exception as e:
            logger.error(f"按键失败: {e}")
    
    def close(self):
        """关闭环境"""
        self.display.close()
        logger.info("X11环境已关闭")
    
    def get_observation_space(self) -> Tuple[int, int, int]:
        """获取观察空间形状"""
        return (self.height, self.width, 3)
    
    def get_action_space(self) -> Dict:
        """获取动作空间"""
        return self.action_space


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建环境
    env = X11GameEnvironment(width=640, height=480)
    
    # 测试屏幕捕获
    frame = env.reset()
    print(f"捕获帧形状: {frame.shape}")
    
    # 测试动作执行
    action = {
        'mouse_dx': 0.1,
        'mouse_dy': 0.1,
        'mouse_click': 0,
        'keys': ['w']
    }
    
    obs, reward, done, info = env.step(action)
    print(f"动作执行完成，新帧形状: {obs.shape}")
    
    # 显示捕获的图像
    cv2.imshow("X11 Game Capture", frame)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    
    env.close()