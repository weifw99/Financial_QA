"""Baostock连接管理模块

提供统一的连接管理机制，确保Baostock的连接状态得到正确管理。
主要功能：
1. 单例模式管理连接
2. 自动登录登出
3. Context manager支持
"""

import time
import random
from typing import Optional

import baostock as bs

class ConnectionManager:
    _instance: Optional['ConnectionManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._login_result = None
        self._api_delay = lambda: random.uniform(0.1, 0.15)  # API调用间隔时间（秒）
        
    def __enter__(self):
        self.login()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logout()
        
    @property
    def is_logged_in(self) -> bool:
        return (self._login_result is not None and 
                hasattr(self._login_result, 'error_code') and 
                self._login_result.error_code == '0')
    
    def login(self):
        """确保Baostock已登录"""
        if self.is_logged_in:
            # print("Baostock已登录，_login_result 已存在")
            return
            
        try:
            self._login_result = bs.login()
            if self._login_result.error_code != '0':
                raise Exception(f'Baostock登录失败: {self._login_result.error_msg}')
            time.sleep(self._api_delay())  # 登录后休眠
        except Exception as e:
            print(f"登录失败: {e}")
            # 清除登录状态
            self._login_result = None
            raise
    
    def logout(self):
        """确保Baostock已登出"""
        if not self.is_logged_in:
            return
            
        try:
            bs.logout()
            time.sleep(self._api_delay())  # 登出后休眠
        except Exception as e:
            print(f"登出失败: {e}")
        finally:
            self._login_result = None