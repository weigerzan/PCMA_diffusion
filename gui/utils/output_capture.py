#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输出捕获工具：捕获子进程的输出并发送到GUI
"""

import sys
import io
from PyQt5.QtCore import QObject, pyqtSignal


class OutputCapture(QObject):
    """捕获stdout和stderr的输出"""
    
    output_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.stdout_capture = None
        self.stderr_capture = None
        
    def start_capture(self):
        """开始捕获输出"""
        self.stdout_capture = StreamCapture(self.output_signal)
        self.stderr_capture = StreamCapture(self.output_signal)
        sys.stdout = self.stdout_capture
        sys.stderr = self.stderr_capture
        
    def stop_capture(self):
        """停止捕获输出"""
        if self.stdout_capture:
            sys.stdout = self.original_stdout
        if self.stderr_capture:
            sys.stderr = self.original_stderr
            
    def write(self, text):
        """写入输出"""
        self.output_signal.emit(text)
        if self.original_stdout:
            self.original_stdout.write(text)
            self.original_stdout.flush()


class StreamCapture(io.TextIOWrapper):
    """流捕获器"""
    
    def __init__(self, signal):
        self.signal = signal
        self.buffer = io.StringIO()
        
    def write(self, text):
        """写入文本"""
        self.signal.emit(text)
        self.buffer.write(text)
        
    def flush(self):
        """刷新缓冲区"""
        self.buffer.flush()
        
    def read(self, size=-1):
        """读取（未实现）"""
        return self.buffer.read(size)
        
    def readline(self, size=-1):
        """读取一行（未实现）"""
        return self.buffer.readline(size)

