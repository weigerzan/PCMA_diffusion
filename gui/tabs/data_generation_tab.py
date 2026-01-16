#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据生成标签页
"""

import sys
import subprocess
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QLineEdit, QTextEdit, QGroupBox, 
                             QMessageBox, QFileDialog, QProgressBar)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from gui.utils.worker_thread import WorkerThread


class DataGenerationTab(QWidget):
    """数据生成标签页"""
    
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
        info_label = QLabel("生成训练和测试数据。支持仿真数据和实采数据两种模式。")
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
        
        # 数据生成选项
        options_group = QGroupBox("生成选项")
        options_layout = QVBoxLayout()
        
        self.sim_radio = QPushButton("生成仿真数据")
        self.sim_radio.clicked.connect(lambda: self.start_generation("sim"))
        options_layout.addWidget(self.sim_radio)
        
        self.real_radio = QPushButton("生成实采数据")
        self.real_radio.clicked.connect(lambda: self.start_generation("real"))
        options_layout.addWidget(self.real_radio)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # 状态显示
        status_group = QGroupBox("状态")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("就绪")
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 不确定进度
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
        self.current_config_path = str(Path(config_path).absolute())
        self.config_edit.setText(self.current_config_path)
        
    def start_generation(self, data_type):
        """开始生成数据"""
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
            
        # 构建命令
        # 数据生成脚本接受完整路径或相对路径，我们使用绝对路径以确保正确
        config_path = str(config_file.absolute())
        
        if data_type == "sim":
            # 仿真数据生成
            command = [
                sys.executable,
                str(self.project_root / "generate_data" / "generate_sim_dataset.py"),
                "--config",
                config_path
            ]
            self.data_type_for_callback = "sim"
        else:  # real
            # 实采数据生成（第一步：切片）
            command = [
                sys.executable,
                str(self.project_root / "generate_data" / "split_from_raw_data.py"),
                "--config",
                config_path
            ]
            self.data_type_for_callback = "real"
            self.config_file_for_callback = config_file
            
        # 创建并启动工作线程
        self.worker_thread = WorkerThread(command, cwd=str(self.project_root))
        self.worker_thread.output_signal.connect(self.on_output)
        self.worker_thread.finished_signal.connect(self.on_finished)
        
        self.status_label.setText("正在生成数据...")
        self.progress_bar.setVisible(True)
        self.stop_btn.setEnabled(True)
        self.sim_radio.setEnabled(False)
        self.real_radio.setEnabled(False)
        
        self.worker_thread.start()
        
        # 如果是实采数据，需要在第一步完成后执行后续步骤
        if data_type == "real":
            # 这里可以扩展为多步骤执行
            pass
            
    def on_output(self, text):
        """处理输出"""
        self.output_capture.output_signal.emit(text)
        
    def on_finished(self, success, message):
        """任务完成回调"""
        if success:
            # 如果是实采数据，需要执行后续步骤
            if hasattr(self, 'data_type_for_callback') and self.data_type_for_callback == "real":
                # 检查是否已经执行了所有步骤
                if not hasattr(self, 'real_data_step'):
                    # 第一步完成，执行第二步
                    self.real_data_step = 1
                    self.status_label.setText("步骤1完成，开始步骤2...")
                    self.output_capture.output_signal.emit("\n开始执行步骤2: 生成混合信号对...\n")
                    
                    command = [
                        sys.executable,
                        str(self.project_root / "generate_data" / "generate_mixed_from_splits.py"),
                        "--config",
                        str(Path(self.config_file_for_callback).absolute()) if isinstance(self.config_file_for_callback, (str, Path)) else str(self.config_file_for_callback.absolute())
                    ]
                    
                    self.worker_thread = WorkerThread(command, cwd=str(self.project_root))
                    self.worker_thread.output_signal.connect(self.on_output)
                    self.worker_thread.finished_signal.connect(self.on_finished)
                    self.worker_thread.start()
                    return
                elif self.real_data_step == 1:
                    # 第二步完成，执行第三步（更新路径）
                    # 检查是否已经执行过路径更新
                    if hasattr(self, 'real_path_updated'):
                        # 已经更新过，清除状态并完成
                        delattr(self, 'real_data_step')
                        delattr(self, 'data_type_for_callback')
                        delattr(self, 'config_file_for_callback')
                        delattr(self, 'real_path_updated')
                        self.progress_bar.setVisible(False)
                        self.stop_btn.setEnabled(False)
                        self.sim_radio.setEnabled(True)
                        self.real_radio.setEnabled(True)
                        self.status_label.setText("数据生成完成")
                        QMessageBox.information(self, "成功", "数据生成完成")
                        return
                    
                    # 标记为正在更新路径，避免重复执行
                    self.real_path_updated = True
                    self.real_data_step = 2
                    self.status_label.setText("步骤2完成，更新数据路径...")
                    self.output_capture.output_signal.emit("\n开始执行步骤3: 更新数据路径...\n")
                    
                    command = [
                        sys.executable,
                        str(self.project_root / "scripts" / "update_data_paths.py"),
                        "--config",
                        str(Path(self.config_file_for_callback).absolute()) if isinstance(self.config_file_for_callback, (str, Path)) else str(self.config_file_for_callback.absolute()),
                        "--data-type",
                        "real"
                    ]
                    
                    self.worker_thread = WorkerThread(command, cwd=str(self.project_root))
                    self.worker_thread.output_signal.connect(self.on_output)
                    self.worker_thread.finished_signal.connect(self.on_finished)
                    self.worker_thread.start()
                    return
                else:
                    # 所有步骤完成
                    if hasattr(self, 'real_data_step'):
                        delattr(self, 'real_data_step')
                    if hasattr(self, 'data_type_for_callback'):
                        delattr(self, 'data_type_for_callback')
                    if hasattr(self, 'config_file_for_callback'):
                        delattr(self, 'config_file_for_callback')
                    if hasattr(self, 'real_path_updated'):
                        delattr(self, 'real_path_updated')
                    
            # 如果是仿真数据，也需要更新路径（只执行一次）
            if hasattr(self, 'data_type_for_callback') and self.data_type_for_callback == "sim":
                # 检查是否已经执行过路径更新
                if hasattr(self, 'sim_path_updated'):
                    # 已经更新过，清除状态并完成
                    delattr(self, 'data_type_for_callback')
                    delattr(self, 'sim_path_updated')
                    self.progress_bar.setVisible(False)
                    self.stop_btn.setEnabled(False)
                    self.sim_radio.setEnabled(True)
                    self.real_radio.setEnabled(True)
                    self.status_label.setText("数据生成完成")
                    QMessageBox.information(self, "成功", "数据生成完成")
                    return
                
                # 标记为正在更新路径，避免重复执行
                self.sim_path_updated = True
                self.status_label.setText("更新数据路径...")
                self.output_capture.output_signal.emit("\n更新数据路径...\n")
                
                command = [
                    sys.executable,
                    str(self.project_root / "scripts" / "update_data_paths.py"),
                    "--config",
                    str(self.current_config_path),
                    "--data-type",
                    "sim"
                ]
                
                self.worker_thread = WorkerThread(command, cwd=str(self.project_root))
                self.worker_thread.output_signal.connect(self.on_output)
                self.worker_thread.finished_signal.connect(self.on_finished)
                self.worker_thread.start()
                return
                
            # 所有步骤完成
            self.progress_bar.setVisible(False)
            self.stop_btn.setEnabled(False)
            self.sim_radio.setEnabled(True)
            self.real_radio.setEnabled(True)
            self.status_label.setText("数据生成完成")
            QMessageBox.information(self, "成功", "数据生成完成")
        else:
            self.progress_bar.setVisible(False)
            self.stop_btn.setEnabled(False)
            self.sim_radio.setEnabled(True)
            self.real_radio.setEnabled(True)
            self.status_label.setText(f"生成失败: {message}")
            QMessageBox.critical(self, "失败", f"数据生成失败: {message}")
            # 清理状态
            if hasattr(self, 'real_data_step'):
                delattr(self, 'real_data_step')
            if hasattr(self, 'data_type_for_callback'):
                delattr(self, 'data_type_for_callback')
            if hasattr(self, 'config_file_for_callback'):
                delattr(self, 'config_file_for_callback')
            if hasattr(self, 'sim_path_updated'):
                delattr(self, 'sim_path_updated')
            if hasattr(self, 'real_path_updated'):
                delattr(self, 'real_path_updated')
            
    def stop_task(self):
        """停止任务"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.status_label.setText("正在停止...")

