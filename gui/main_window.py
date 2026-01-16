#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCMA扩散模型盲分离任务GUI主窗口
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTabWidget, QPushButton, QTextEdit, QLabel, 
                             QMessageBox, QFileDialog, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QTextCursor

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from gui.tabs.data_generation_tab import DataGenerationTab
from gui.tabs.training_tab import TrainingTab
from gui.tabs.inference_tab import InferenceTab
from gui.tabs.demodulation_tab import DemodulationTab
from gui.tabs.config_tab import ConfigTab
from gui.utils.output_capture import OutputCapture


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCMA扩散模型盲分离任务GUI")
        self.setGeometry(100, 100, 1400, 900)
        
        # 设置工作目录为项目根目录
        self.project_root = Path(__file__).parent.parent
        os.chdir(self.project_root)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建分割器（左侧：标签页，右侧：终端输出）
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧：标签页
        self.tab_widget = QTabWidget()
        splitter.addWidget(self.tab_widget)
        
        # 右侧：终端输出
        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        output_layout.setContentsMargins(5, 5, 5, 5)
        
        output_label = QLabel("终端输出")
        output_label.setFont(QFont("Arial", 10, QFont.Bold))
        output_layout.addWidget(output_label)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Courier", 9))
        self.output_text.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        output_layout.addWidget(self.output_text)
        
        # 清除输出按钮
        clear_btn = QPushButton("清除输出")
        clear_btn.clicked.connect(self.output_text.clear)
        output_layout.addWidget(clear_btn)
        
        splitter.addWidget(output_widget)
        
        # 设置分割器比例（左侧70%，右侧30%）
        splitter.setSizes([1000, 400])
        
        # 创建输出捕获器
        self.output_capture = OutputCapture()
        self.output_capture.output_signal.connect(self.append_output)
        
        # 创建各个标签页
        self.config_tab = ConfigTab(self.output_capture, self.project_root)
        self.data_gen_tab = DataGenerationTab(self.output_capture, self.project_root)
        self.training_tab = TrainingTab(self.output_capture, self.project_root)
        self.inference_tab = InferenceTab(self.output_capture, self.project_root)
        self.demod_tab = DemodulationTab(self.output_capture, self.project_root)
        
        # 添加标签页
        self.tab_widget.addTab(self.config_tab, "配置管理")
        self.tab_widget.addTab(self.data_gen_tab, "1. 生成数据")
        self.tab_widget.addTab(self.training_tab, "2. 训练模型")
        self.tab_widget.addTab(self.inference_tab, "3. 模型推理")
        self.tab_widget.addTab(self.demod_tab, "4. 解调输出")
        
        # 连接信号：配置更新后通知其他标签页
        self.config_tab.config_updated.connect(self.on_config_updated)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
    def append_output(self, text):
        """追加输出到终端显示"""
        self.output_text.moveCursor(QTextCursor.End)
        self.output_text.insertPlainText(text)
        self.output_text.moveCursor(QTextCursor.End)
        
    def on_config_updated(self, config_path):
        """配置更新后的回调"""
        # 通知各个标签页配置已更新
        self.data_gen_tab.load_config(config_path)
        self.training_tab.load_config(config_path)
        self.inference_tab.load_config(config_path)
        self.demod_tab.load_config(config_path)
        self.statusBar().showMessage(f"配置已更新: {config_path}")
        
    def closeEvent(self, event):
        """关闭事件"""
        # 停止所有运行中的任务
        self.data_gen_tab.stop_task()
        self.training_tab.stop_task()
        self.inference_tab.stop_task()
        self.demod_tab.stop_task()
        event.accept()


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

