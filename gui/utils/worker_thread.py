#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作线程：在后台执行长时间运行的任务
"""

import subprocess
import sys
import os
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal


class WorkerThread(QThread):
    """工作线程基类"""
    
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)  # (success, message)
    
    def __init__(self, command, cwd=None, env=None):
        super().__init__()
        self.command = command
        self.cwd = cwd or Path.cwd()
        self.env = env or os.environ.copy()
        self._stop_flag = False
        
    def stop(self):
        """停止任务"""
        self._stop_flag = True
        
    def run(self):
        """执行任务"""
        try:
            self.output_signal.emit(f"执行命令: {' '.join(self.command)}\n")
            self.output_signal.emit(f"工作目录: {self.cwd}\n\n")
            
            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(self.cwd),
                env=self.env,
                universal_newlines=True,
                bufsize=1
            )
            
            # 实时读取输出
            for line in process.stdout:
                if self._stop_flag:
                    process.terminate()
                    self.output_signal.emit("\n任务已停止\n")
                    self.finished_signal.emit(False, "任务已停止")
                    return
                    
                self.output_signal.emit(line)
                
            # 等待进程完成
            return_code = process.wait()
            
            if return_code == 0:
                self.finished_signal.emit(True, "任务完成")
            else:
                self.finished_signal.emit(False, f"任务失败，退出码: {return_code}")
                
        except Exception as e:
            self.output_signal.emit(f"\n错误: {str(e)}\n")
            self.finished_signal.emit(False, f"错误: {str(e)}")

