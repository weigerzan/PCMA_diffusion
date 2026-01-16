#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型推理标签页
"""

import sys
import os
import yaml
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QLineEdit, QGroupBox, QMessageBox,
                             QProgressBar, QFormLayout)
from PyQt5.QtCore import Qt
from gui.utils.worker_thread import WorkerThread


class InferenceTab(QWidget):
    """模型推理标签页"""
    
    def __init__(self, output_capture, project_root):
        super().__init__()
        self.output_capture = output_capture
        self.project_root = project_root
        self.current_config_path = None
        self.worker_thread = None
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 说明
        info_label = QLabel("使用训练好的模型进行推理，生成分离后的信号。测试数据路径会自动从配置文件中读取。")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 配置文件
        config_group = QGroupBox("配置文件")
        config_layout = QHBoxLayout()
        config_layout.addWidget(QLabel("配置文件:"))
        self.config_edit = QLineEdit()
        self.config_edit.setPlaceholderText("从配置管理标签页选择配置文件...")
        self.config_edit.setReadOnly(True)
        config_layout.addWidget(self.config_edit)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # 关键参数显示（只读）
        params_group = QGroupBox("关键参数（只读，从配置文件读取）")
        params_layout = QFormLayout()
        
        self.steps_label = QLabel("-")
        params_layout.addRow("采样步数:", self.steps_label)
        
        self.eta_label = QLabel("-")
        params_layout.addRow("DDIM参数 (eta):", self.eta_label)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # GPU选择
        gpu_group = QGroupBox("GPU设置")
        gpu_layout = QFormLayout()
        
        self.gpu_edit = QLineEdit()
        self.gpu_edit.setPlaceholderText("例如: 0, 1, 2 或留空使用默认GPU")
        gpu_layout.addRow("CUDA_VISIBLE_DEVICES:", self.gpu_edit)
        help_label = QLabel("留空则使用默认GPU，或输入GPU序号（如0,1,2）")
        help_label.setStyleSheet("color: gray; font-size: 10px;")
        gpu_layout.addRow("", help_label)
        
        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)
        
        # 开始推理按钮
        self.start_btn = QPushButton("开始推理")
        self.start_btn.clicked.connect(self.start_inference)
        layout.addWidget(self.start_btn)
        
        # 状态显示
        status_group = QGroupBox("状态")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("就绪")
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_task)
        status_layout.addWidget(self.stop_btn)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        layout.addStretch()
        
    def load_config(self, config_path):
        """加载配置文件"""
        self.current_config_path = config_path
        self.config_edit.setText(config_path)
        
        # 从配置文件读取并显示关键参数
        self.update_params_display()
    
    def update_params_display(self):
        """从配置文件更新参数显示"""
        if not self.current_config_path:
            self.steps_label.setText("-")
            self.eta_label.setText("-")
            return
        
        try:
            config_file = Path(self.current_config_path)
            if not config_file.exists():
                return
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 读取推理参数
            if 'sampling' in config:
                sampling = config['sampling']
                
                steps = sampling.get('num_inference_steps', '-')
                self.steps_label.setText(str(steps))
                
                eta = sampling.get('eta', '-')
                self.eta_label.setText(str(eta))
            else:
                self.steps_label.setText("-")
                self.eta_label.setText("-")
        except Exception as e:
            self.steps_label.setText("读取失败")
            self.eta_label.setText("读取失败")
        
    def start_inference(self):
        """开始推理"""
        if not self.current_config_path:
            QMessageBox.warning(self, "警告", "请先在配置管理标签页选择配置文件")
            return
            
        config_file = Path(self.current_config_path)
        if not config_file.exists():
            QMessageBox.critical(self, "错误", f"配置文件不存在: {config_file}")
            return
            
        # 停止之前的任务
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.warning(self, "警告", "已有任务正在运行，请先停止")
            return
            
        # 构建命令（使用相对于configs目录的文件名）
        # 脚本期望: configs/文件名.yaml
        config_relative = config_file.name  # 只传文件名
        command = [
            sys.executable,
            str(self.project_root / "generate_diffusion_output.py"),
            "--config",
            config_relative
        ]
        
        # 设置环境变量（如果指定了GPU）
        env = os.environ.copy()
        gpu_text = self.gpu_edit.text().strip()
        if gpu_text:
            env['CUDA_VISIBLE_DEVICES'] = gpu_text
        
        # 创建并启动工作线程
        self.worker_thread = WorkerThread(command, cwd=str(self.project_root), env=env)
        self.worker_thread.output_signal.connect(self.on_output)
        self.worker_thread.finished_signal.connect(self.on_finished)
        
        self.status_label.setText("正在推理...")
        self.progress_bar.setVisible(True)
        self.stop_btn.setEnabled(True)
        self.start_btn.setEnabled(False)
        
        self.worker_thread.start()
        
    def on_output(self, text):
        """处理输出"""
        self.output_capture.output_signal.emit(text)
        
    def on_finished(self, success, message):
        """任务完成回调"""
        self.progress_bar.setVisible(False)
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        
        if success:
            self.status_label.setText("推理完成")
            QMessageBox.information(self, "成功", "模型推理完成")
        else:
            self.status_label.setText(f"推理失败: {message}")
            QMessageBox.critical(self, "失败", f"模型推理失败: {message}")
            
    def stop_task(self):
        """停止任务"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.status_label.setText("正在停止...")

