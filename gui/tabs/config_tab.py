#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理标签页 - 重新设计
"""

import yaml
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QLineEdit, QComboBox, QDoubleSpinBox, 
                             QSpinBox, QGroupBox, QFileDialog, QMessageBox,
                             QTabWidget, QFormLayout, QTextEdit, QScrollArea)
from PyQt5.QtCore import pyqtSignal, Qt
import shutil
import os


class ConfigTab(QWidget):
    """配置管理标签页"""
    
    config_updated = pyqtSignal(str)  # 配置更新信号
    
    def __init__(self, output_capture, project_root):
        super().__init__()
        self.output_capture = output_capture
        self.project_root = project_root
        self.current_config_path = None
        self.config = None
        self.is_new_config = False  # 标记是否是新创建的配置
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 配置文件管理
        config_group = QGroupBox("配置文件管理")
        config_layout = QVBoxLayout()
        
        config_path_layout = QHBoxLayout()
        config_path_layout.addWidget(QLabel("配置文件:"))
        self.config_path_edit = QLineEdit()
        self.config_path_edit.setPlaceholderText("选择或创建配置文件...")
        self.config_path_edit.setReadOnly(True)
        config_path_layout.addWidget(self.config_path_edit)
        
        self.load_btn = QPushButton("加载")
        self.load_btn.clicked.connect(self.load_config_file)
        config_path_layout.addWidget(self.load_btn)
        
        self.new_btn = QPushButton("新建")
        self.new_btn.clicked.connect(self.create_new_config)
        config_path_layout.addWidget(self.new_btn)
        
        self.delete_btn = QPushButton("删除")
        self.delete_btn.clicked.connect(self.delete_config_file)
        self.delete_btn.setEnabled(False)
        config_path_layout.addWidget(self.delete_btn)
        
        config_layout.addLayout(config_path_layout)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # 重要参数显示（只读）
        self.important_params_group = QGroupBox("重要参数（只读，不可修改）")
        self.important_params_layout = QFormLayout()
        self.important_params_group.setLayout(self.important_params_layout)
        self.important_params_group.setVisible(False)
        layout.addWidget(self.important_params_group)
        
        # 配置标签页（新建时显示）
        self.config_tabs = QTabWidget()
        self.config_tabs.setVisible(False)
        layout.addWidget(self.config_tabs)
        
        # 创建各个配置标签页
        self.create_simple_config_tab()
        self.create_path_config_tab()
        self.create_advanced_config_tab()
        
        # 初始化时根据数据类型显示/隐藏参数组
        # 默认显示仿真参数组（因为默认选择的是"仿真"）
        if hasattr(self, 'sim_data_gen_group'):
            self.sim_data_gen_group.setVisible(True)
        if hasattr(self, 'real_data_gen_group'):
            self.real_data_gen_group.setVisible(False)
        
        layout.addStretch()
        
    def create_simple_config_tab(self):
        """创建简易配置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 数据类型选择（新建时）
        self.data_type_group = QGroupBox("数据类型")
        data_type_layout = QHBoxLayout()
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems(["仿真", "实采"])
        self.data_type_combo.currentTextChanged.connect(self.on_data_type_changed)
        data_type_layout.addWidget(QLabel("数据类型:"))
        data_type_layout.addWidget(self.data_type_combo)
        data_type_layout.addStretch()
        self.data_type_group.setLayout(data_type_layout)
        scroll_layout.addWidget(self.data_type_group)
        
        # 重要参数配置
        self.simple_params_group = QGroupBox("重要参数")
        simple_layout = QFormLayout()
        
        # 调制方式
        self.modulation_combo = QComboBox()
        self.modulation_combo.addItems(["QPSK", "8PSK", "16QAM"])
        simple_layout.addRow("调制方式:", self.modulation_combo)
        
        # 幅度比
        self.amp_ratio_spin = QDoubleSpinBox()
        self.amp_ratio_spin.setRange(0.1, 1.0)
        self.amp_ratio_spin.setSingleStep(0.1)
        self.amp_ratio_spin.setValue(0.7)
        self.amp_ratio_spin.setDecimals(1)
        simple_layout.addRow("幅度比:", self.amp_ratio_spin)
        
        # 信噪比（范围值）
        snr_range_layout = QHBoxLayout()
        self.snr_min_spin = QDoubleSpinBox()
        self.snr_min_spin.setRange(0, 30)
        self.snr_min_spin.setSingleStep(1)
        self.snr_min_spin.setValue(15)
        self.snr_min_spin.setDecimals(1)
        self.snr_max_spin = QDoubleSpinBox()
        self.snr_max_spin.setRange(0, 30)
        self.snr_max_spin.setSingleStep(1)
        self.snr_max_spin.setValue(15)
        self.snr_max_spin.setDecimals(1)
        # 连接信号，确保最小值不超过最大值
        self.snr_min_spin.valueChanged.connect(lambda v: self.snr_max_spin.setMinimum(v))
        self.snr_max_spin.valueChanged.connect(lambda v: self.snr_min_spin.setMaximum(v))
        # 创建Es/N0标签（动态更新）
        self.es_no_label = QLabel("")
        self.es_no_label.setStyleSheet("color: gray; font-size: 10px;")
        # 更新Es/N0显示的函数
        def update_es_no():
            snr_min = self.snr_min_spin.value()
            snr_max = self.snr_max_spin.value()
            if snr_min == snr_max:
                es_no = snr_min - 9
                self.es_no_label.setText(f"(Es/N0: {es_no:.1f} dB)")
            else:
                es_no_min = snr_min - 9
                es_no_max = snr_max - 9
                self.es_no_label.setText(f"(Es/N0: {es_no_min:.1f}-{es_no_max:.1f} dB)")
        # 连接信号以更新Es/N0显示
        self.snr_min_spin.valueChanged.connect(lambda v: update_es_no())
        self.snr_max_spin.valueChanged.connect(lambda v: update_es_no())
        # 初始更新
        update_es_no()
        snr_range_layout.addWidget(QLabel("信噪比 (dB):"))
        snr_range_layout.addWidget(QLabel("最小值:"))
        snr_range_layout.addWidget(self.snr_min_spin)
        snr_range_layout.addWidget(QLabel("最大值:"))
        snr_range_layout.addWidget(self.snr_max_spin)
        snr_range_layout.addWidget(self.es_no_label)
        snr_range_layout.addStretch()
        simple_layout.addRow(snr_range_layout)
        
        # 仿真数据专用参数
        self.sim_params_group = QGroupBox("仿真数据参数")
        sim_layout = QFormLayout()
        
        # 频偏1（范围值，支持负数）
        freq1_range_layout = QHBoxLayout()
        self.freq_offset1_min_spin = QDoubleSpinBox()
        self.freq_offset1_min_spin.setRange(-500, 500)
        self.freq_offset1_min_spin.setValue(0)
        self.freq_offset1_max_spin = QDoubleSpinBox()
        self.freq_offset1_max_spin.setRange(-500, 500)
        self.freq_offset1_max_spin.setValue(0)
        self.freq_offset1_min_spin.valueChanged.connect(lambda v: self.freq_offset1_max_spin.setMinimum(v))
        self.freq_offset1_max_spin.valueChanged.connect(lambda v: self.freq_offset1_min_spin.setMaximum(v))
        freq1_range_layout.addWidget(QLabel("最小值:"))
        freq1_range_layout.addWidget(self.freq_offset1_min_spin)
        freq1_range_layout.addWidget(QLabel("最大值:"))
        freq1_range_layout.addWidget(self.freq_offset1_max_spin)
        freq1_range_layout.addStretch()
        sim_layout.addRow("频偏1 (Hz):", freq1_range_layout)
        
        # 频偏2（范围值，支持负数）
        freq2_range_layout = QHBoxLayout()
        self.freq_offset2_min_spin = QDoubleSpinBox()
        self.freq_offset2_min_spin.setRange(-500, 500)
        self.freq_offset2_min_spin.setValue(0)
        self.freq_offset2_max_spin = QDoubleSpinBox()
        self.freq_offset2_max_spin.setRange(-500, 500)
        self.freq_offset2_max_spin.setValue(0)
        self.freq_offset2_min_spin.valueChanged.connect(lambda v: self.freq_offset2_max_spin.setMinimum(v))
        self.freq_offset2_max_spin.valueChanged.connect(lambda v: self.freq_offset2_min_spin.setMaximum(v))
        freq2_range_layout.addWidget(QLabel("最小值:"))
        freq2_range_layout.addWidget(self.freq_offset2_min_spin)
        freq2_range_layout.addWidget(QLabel("最大值:"))
        freq2_range_layout.addWidget(self.freq_offset2_max_spin)
        freq2_range_layout.addStretch()
        sim_layout.addRow("频偏2 (Hz):", freq2_range_layout)
        
        # 相偏1（范围值）
        phase1_range_layout = QHBoxLayout()
        self.phase1_min_spin = QDoubleSpinBox()
        self.phase1_min_spin.setRange(0, 2)
        self.phase1_min_spin.setValue(0)
        self.phase1_min_spin.setDecimals(3)
        self.phase1_max_spin = QDoubleSpinBox()
        self.phase1_max_spin.setRange(0, 2)
        self.phase1_max_spin.setValue(0)
        self.phase1_max_spin.setDecimals(3)
        self.phase1_min_spin.valueChanged.connect(lambda v: self.phase1_max_spin.setMinimum(v))
        self.phase1_max_spin.valueChanged.connect(lambda v: self.phase1_min_spin.setMaximum(v))
        phase1_range_layout.addWidget(QLabel("最小值:"))
        phase1_range_layout.addWidget(self.phase1_min_spin)
        phase1_range_layout.addWidget(QLabel("最大值:"))
        phase1_range_layout.addWidget(self.phase1_max_spin)
        phase1_range_layout.addStretch()
        sim_layout.addRow("相偏1 (π):", phase1_range_layout)
        
        # 相偏2（范围值）
        phase2_range_layout = QHBoxLayout()
        self.phase2_min_spin = QDoubleSpinBox()
        self.phase2_min_spin.setRange(0, 2)
        self.phase2_min_spin.setValue(0)
        self.phase2_min_spin.setDecimals(3)
        self.phase2_max_spin = QDoubleSpinBox()
        self.phase2_max_spin.setRange(0, 2)
        self.phase2_max_spin.setValue(0)
        self.phase2_max_spin.setDecimals(3)
        self.phase2_min_spin.valueChanged.connect(lambda v: self.phase2_max_spin.setMinimum(v))
        self.phase2_max_spin.valueChanged.connect(lambda v: self.phase2_min_spin.setMaximum(v))
        phase2_range_layout.addWidget(QLabel("最小值:"))
        phase2_range_layout.addWidget(self.phase2_min_spin)
        phase2_range_layout.addWidget(QLabel("最大值:"))
        phase2_range_layout.addWidget(self.phase2_max_spin)
        phase2_range_layout.addStretch()
        sim_layout.addRow("相偏2 (π):", phase2_range_layout)
        
        # 时延差1（范围值）
        delay1_range_layout = QHBoxLayout()
        self.delay1_min_spin = QSpinBox()
        self.delay1_min_spin.setRange(0, 20)
        self.delay1_min_spin.setValue(0)
        self.delay1_max_spin = QSpinBox()
        self.delay1_max_spin.setRange(0, 20)
        self.delay1_max_spin.setValue(0)
        self.delay1_min_spin.valueChanged.connect(lambda v: self.delay1_max_spin.setMinimum(v))
        self.delay1_max_spin.valueChanged.connect(lambda v: self.delay1_min_spin.setMaximum(v))
        delay1_range_layout.addWidget(QLabel("最小值:"))
        delay1_range_layout.addWidget(self.delay1_min_spin)
        delay1_range_layout.addWidget(QLabel("最大值:"))
        delay1_range_layout.addWidget(self.delay1_max_spin)
        delay1_range_layout.addStretch()
        sim_layout.addRow("时延差1 (采样点):", delay1_range_layout)
        
        # 时延差2（范围值）
        delay2_range_layout = QHBoxLayout()
        self.delay2_min_spin = QSpinBox()
        self.delay2_min_spin.setRange(0, 20)
        self.delay2_min_spin.setValue(0)
        self.delay2_max_spin = QSpinBox()
        self.delay2_max_spin.setRange(0, 20)
        self.delay2_max_spin.setValue(0)
        self.delay2_min_spin.valueChanged.connect(lambda v: self.delay2_max_spin.setMinimum(v))
        self.delay2_max_spin.valueChanged.connect(lambda v: self.delay2_min_spin.setMaximum(v))
        delay2_range_layout.addWidget(QLabel("最小值:"))
        delay2_range_layout.addWidget(self.delay2_min_spin)
        delay2_range_layout.addWidget(QLabel("最大值:"))
        delay2_range_layout.addWidget(self.delay2_max_spin)
        delay2_range_layout.addStretch()
        sim_layout.addRow("时延差2 (采样点):", delay2_range_layout)
        
        self.sim_params_group.setLayout(sim_layout)
        simple_layout.addRow(self.sim_params_group)
        
        self.simple_params_group.setLayout(simple_layout)
        scroll_layout.addWidget(self.simple_params_group)
        
        # 训练参数
        training_group = QGroupBox("训练参数")
        training_layout = QFormLayout()
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(500)
        training_layout.addRow("训练轮数:", self.epochs_spin)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 512)
        self.batch_size_spin.setValue(64)
        training_layout.addRow("批次大小:", self.batch_size_spin)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1e-2)
        self.lr_spin.setValue(2e-4)
        self.lr_spin.setDecimals(6)
        training_layout.addRow("学习率:", self.lr_spin)
        
        # 预训练模型路径
        pretrained_layout = QHBoxLayout()
        self.pretrained_edit = QLineEdit()
        self.pretrained_edit.setPlaceholderText("留空表示null，格式: PATH/unet/")
        self.pretrained_btn = QPushButton("浏览")
        self.pretrained_btn.clicked.connect(lambda: self.browse_directory(self.pretrained_edit))
        pretrained_layout.addWidget(self.pretrained_edit)
        pretrained_layout.addWidget(self.pretrained_btn)
        training_layout.addRow("预训练模型路径:", pretrained_layout)
        pretrained_help = QLabel("留空表示null，格式一般为 PATH/unet/")
        pretrained_help.setStyleSheet("color: gray; font-size: 10px;")
        training_layout.addRow("", pretrained_help)
        
        training_group.setLayout(training_layout)
        scroll_layout.addWidget(training_group)
        
        # 推理参数
        inference_group = QGroupBox("推理参数")
        inference_layout = QFormLayout()
        
        self.inference_steps_spin = QSpinBox()
        self.inference_steps_spin.setRange(10, 1000)
        self.inference_steps_spin.setValue(100)
        inference_layout.addRow("采样步数:", self.inference_steps_spin)
        
        self.eta_spin = QDoubleSpinBox()
        self.eta_spin.setRange(0, 1)
        self.eta_spin.setValue(0.15)
        self.eta_spin.setDecimals(3)
        inference_layout.addRow("DDIM参数 (eta):", self.eta_spin)
        
        inference_group.setLayout(inference_layout)
        scroll_layout.addWidget(inference_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        self.config_tabs.addTab(tab, "简易配置")
        
    def create_path_config_tab(self):
        """创建路径配置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        path_layout = QFormLayout()
        
        # 数据生成保存路径
        self.data_save_dir_layout = QHBoxLayout()
        self.data_save_dir_edit = QLineEdit()
        self.data_save_dir_edit.setPlaceholderText("数据生成保存路径...")
        self.data_save_dir_btn = QPushButton("浏览")
        self.data_save_dir_btn.clicked.connect(lambda: self.browse_directory(self.data_save_dir_edit))
        self.data_save_dir_layout.addWidget(self.data_save_dir_edit)
        self.data_save_dir_layout.addWidget(self.data_save_dir_btn)
        path_layout.addRow("数据生成保存路径:", self.data_save_dir_layout)
        
        # 实采数据原始数据路径（三种调制方式）
        self.raw_data_group = QGroupBox("实采数据原始数据路径")
        raw_data_layout = QFormLayout()
        
        self.raw_qpsk_edit = QLineEdit()
        self.raw_qpsk_btn = QPushButton("浏览")
        self.raw_qpsk_btn.clicked.connect(lambda: self.browse_file(self.raw_qpsk_edit, "*.dat"))
        raw_qpsk_layout = QHBoxLayout()
        raw_qpsk_layout.addWidget(self.raw_qpsk_edit)
        raw_qpsk_layout.addWidget(self.raw_qpsk_btn)
        raw_data_layout.addRow("QPSK原始数据:", raw_qpsk_layout)
        
        self.raw_8psk_edit = QLineEdit()
        self.raw_8psk_btn = QPushButton("浏览")
        self.raw_8psk_btn.clicked.connect(lambda: self.browse_file(self.raw_8psk_edit, "*.dat"))
        raw_8psk_layout = QHBoxLayout()
        raw_8psk_layout.addWidget(self.raw_8psk_edit)
        raw_8psk_layout.addWidget(self.raw_8psk_btn)
        raw_data_layout.addRow("8PSK原始数据:", raw_8psk_layout)
        
        self.raw_16qam_edit = QLineEdit()
        self.raw_16qam_btn = QPushButton("浏览")
        self.raw_16qam_btn.clicked.connect(lambda: self.browse_file(self.raw_16qam_edit, "*.dat"))
        raw_16qam_layout = QHBoxLayout()
        raw_16qam_layout.addWidget(self.raw_16qam_edit)
        raw_16qam_layout.addWidget(self.raw_16qam_btn)
        raw_data_layout.addRow("16QAM原始数据:", raw_16qam_layout)
        
        self.raw_data_group.setLayout(raw_data_layout)
        path_layout.addRow(self.raw_data_group)
        
        # 模型保存路径
        self.model_save_dir_layout = QHBoxLayout()
        self.model_save_dir_edit = QLineEdit()
        self.model_save_dir_edit.setPlaceholderText("模型保存路径...")
        self.model_save_dir_btn = QPushButton("浏览")
        self.model_save_dir_btn.clicked.connect(lambda: self.browse_directory(self.model_save_dir_edit))
        self.model_save_dir_layout.addWidget(self.model_save_dir_edit)
        self.model_save_dir_layout.addWidget(self.model_save_dir_btn)
        path_layout.addRow("模型保存路径:", self.model_save_dir_layout)
        
        # 推理结果保存路径
        self.inference_save_dir_layout = QHBoxLayout()
        self.inference_save_dir_edit = QLineEdit()
        self.inference_save_dir_edit.setPlaceholderText("推理结果保存路径...")
        self.inference_save_dir_btn = QPushButton("浏览")
        self.inference_save_dir_btn.clicked.connect(lambda: self.browse_directory(self.inference_save_dir_edit))
        self.inference_save_dir_layout.addWidget(self.inference_save_dir_edit)
        self.inference_save_dir_layout.addWidget(self.inference_save_dir_btn)
        path_layout.addRow("推理结果保存路径:", self.inference_save_dir_layout)
        
        scroll_layout.addLayout(path_layout)
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        self.config_tabs.addTab(tab, "路径配置")
        
    def create_advanced_config_tab(self):
        """创建高级配置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        advanced_layout = QFormLayout()
        
        # 仿真数据生成参数
        sim_data_gen_group = QGroupBox("仿真数据生成参数")
        sim_data_gen_layout = QFormLayout()
        
        # num_samples (仿真数据总样本数)
        self.num_samples_spin = QSpinBox()
        self.num_samples_spin.setRange(1000, 10000000)
        self.num_samples_spin.setValue(100000)
        sim_data_gen_layout.addRow("总样本数 (num_samples):", self.num_samples_spin)
        
        # shard_size (每个分片的样本数)
        self.shard_size_spin = QSpinBox()
        self.shard_size_spin.setRange(100, 100000)
        self.shard_size_spin.setValue(10000)
        sim_data_gen_layout.addRow("每个分片的样本数 (shard_size):", self.shard_size_spin)
        
        sim_data_gen_group.setLayout(sim_data_gen_layout)
        advanced_layout.addRow(sim_data_gen_group)
        self.sim_data_gen_group = sim_data_gen_group  # 保存引用以便控制显示/隐藏
        
        # 实采数据生成参数
        real_data_gen_group = QGroupBox("实采数据生成参数")
        real_data_gen_layout = QFormLayout()
        
        # max_samples (实采数据每种调制方式最多生成的样本数)
        self.max_samples_spin = QSpinBox()
        self.max_samples_spin.setRange(1000, 10000000)
        self.max_samples_spin.setValue(1000000)
        real_data_gen_layout.addRow("每种调制方式最多生成的样本数 (max_samples):", self.max_samples_spin)
        
        # max_slices_per_file (每个文件最多处理的切片数)
        self.max_slices_spin = QSpinBox()
        self.max_slices_spin.setRange(1, 100000)
        self.max_slices_spin.setValue(10000)
        real_data_gen_layout.addRow("每个文件最多处理的切片数 (max_slices_per_file):", self.max_slices_spin)
        
        # threshold (聚类分数阈值)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0, 20)
        self.threshold_spin.setValue(6.0)
        self.threshold_spin.setDecimals(1)
        real_data_gen_layout.addRow("聚类分数阈值 (threshold):", self.threshold_spin)
        
        # train_ratio (训练集比例)
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0, 1)
        self.train_ratio_spin.setValue(0.9)
        self.train_ratio_spin.setDecimals(2)
        real_data_gen_layout.addRow("训练集比例 (train_ratio):", self.train_ratio_spin)
        
        # target_pairs (训练集目标生成的混合信号对数量)
        self.target_pairs_spin = QSpinBox()
        self.target_pairs_spin.setRange(100, 1000000)
        self.target_pairs_spin.setValue(100000)
        real_data_gen_layout.addRow("训练集目标生成的混合信号对数量 (target_pairs):", self.target_pairs_spin)
        
        # test_target_pairs (测试集目标生成的混合信号对数量，null表示使用target_pairs的10%)
        self.test_target_pairs_spin = QSpinBox()
        self.test_target_pairs_spin.setRange(0, 1000000)
        self.test_target_pairs_spin.setValue(0)  # 0表示null，使用target_pairs的10%
        self.test_target_pairs_spin.setSpecialValueText("自动 (target_pairs的10%)")
        real_data_gen_layout.addRow("测试集目标生成的混合信号对数量 (test_target_pairs):", self.test_target_pairs_spin)
        
        # sps (每符号采样数)
        self.sps_spin = QSpinBox()
        self.sps_spin.setRange(1, 32)
        self.sps_spin.setValue(8)
        real_data_gen_layout.addRow("每符号采样数 (sps):", self.sps_spin)
        
        # samples_per_file (每个文件的样本数)
        self.samples_per_file_spin = QSpinBox()
        self.samples_per_file_spin.setRange(1, 1000)
        self.samples_per_file_spin.setValue(30)
        real_data_gen_layout.addRow("每个文件的样本数 (samples_per_file):", self.samples_per_file_spin)
        
        real_data_gen_group.setLayout(real_data_gen_layout)
        advanced_layout.addRow(real_data_gen_group)
        self.real_data_gen_group = real_data_gen_group  # 保存引用以便控制显示/隐藏
        
        # 训练高级参数
        training_advanced_group = QGroupBox("训练高级参数")
        training_advanced_layout = QFormLayout()
        
        # test_batch_size
        self.test_batch_size_spin = QSpinBox()
        self.test_batch_size_spin.setRange(1, 512)
        self.test_batch_size_spin.setValue(64)
        training_advanced_layout.addRow("测试批次大小 (test_batch_size):", self.test_batch_size_spin)
        
        # gradient_accumulation_steps
        self.gradient_accumulation_steps_spin = QSpinBox()
        self.gradient_accumulation_steps_spin.setRange(1, 100)
        self.gradient_accumulation_steps_spin.setValue(1)
        training_advanced_layout.addRow("梯度累积步数 (gradient_accumulation_steps):", self.gradient_accumulation_steps_spin)
        
        # lr_warmup_steps
        self.lr_warmup_steps_spin = QSpinBox()
        self.lr_warmup_steps_spin.setRange(0, 10000)
        self.lr_warmup_steps_spin.setValue(200)
        training_advanced_layout.addRow("学习率预热步数 (lr_warmup_steps):", self.lr_warmup_steps_spin)
        
        # save_data_epochs
        self.save_data_epochs_spin = QSpinBox()
        self.save_data_epochs_spin.setRange(1, 100)
        self.save_data_epochs_spin.setValue(1)
        training_advanced_layout.addRow("多久做一次测试 (save_data_epochs):", self.save_data_epochs_spin)
        
        # save_model_epochs
        self.save_model_epochs_spin = QSpinBox()
        self.save_model_epochs_spin.setRange(1, 100)
        self.save_model_epochs_spin.setValue(1)
        training_advanced_layout.addRow("多久保存一次模型 (save_model_epochs):", self.save_model_epochs_spin)
        
        training_advanced_group.setLayout(training_advanced_layout)
        advanced_layout.addRow(training_advanced_group)
        
        # 模型参数
        model_group = QGroupBox("模型参数")
        model_layout = QFormLayout()
        
        # predict_target固定为x0，不在UI中显示
        # 在保存配置时自动设置为"x0"
        
        # num_diffusion_timesteps
        self.num_diffusion_timesteps_spin = QSpinBox()
        self.num_diffusion_timesteps_spin.setRange(100, 2000)
        self.num_diffusion_timesteps_spin.setValue(1000)
        model_layout.addRow("扩散时间步数 (num_diffusion_timesteps):", self.num_diffusion_timesteps_spin)
        
        # signal_len
        self.signal_len_spin = QSpinBox()
        self.signal_len_spin.setRange(512, 10000)
        self.signal_len_spin.setValue(3072)
        model_layout.addRow("信号长度 (signal_len):", self.signal_len_spin)
        
        model_group.setLayout(model_layout)
        advanced_layout.addRow(model_group)
        
        # 系统配置
        system_group = QGroupBox("系统配置")
        system_layout = QFormLayout()
        
        # num_workers (CPU核心数，同时用于generate_mixed和split)
        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 128)
        self.num_workers_spin.setValue(0)  # 0表示null，使用所有可用核心
        self.num_workers_spin.setSpecialValueText("自动 (使用所有可用核心)")
        system_layout.addRow("多线程处理CPU核心数 (num_workers):", self.num_workers_spin)
        help_label = QLabel("此参数将同时设置到 generate_mixed.num_workers 和 split.num_workers")
        help_label.setStyleSheet("color: gray; font-size: 10px;")
        system_layout.addRow("", help_label)
        
        system_group.setLayout(system_layout)
        advanced_layout.addRow(system_group)
        
        # 输出目录命名规则说明
        naming_info = QLabel(
            "输出目录命名规则：\n"
            "模型保存路径和推理结果保存路径会根据参数自动生成子目录，避免覆盖。\n"
            "格式：{基础路径}/{调制方式}_{数据类型}_{信噪比}dB_{幅度比}"
        )
        naming_info.setWordWrap(True)
        naming_info.setStyleSheet("color: blue;")
        advanced_layout.addRow(naming_info)
        
        scroll_layout.addLayout(advanced_layout)
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        # 保存按钮
        save_btn = QPushButton("保存配置")
        save_btn.clicked.connect(self.save_config)
        layout.addWidget(save_btn)
        
        self.config_tabs.addTab(tab, "高级配置")
        
    def browse_directory(self, line_edit):
        """浏览目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择目录")
        if dir_path:
            line_edit.setText(dir_path)
            
    def browse_file(self, line_edit, file_filter="*"):
        """浏览文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", f"Files ({file_filter})")
        if file_path:
            line_edit.setText(file_path)
            
    def on_data_type_changed(self, data_type_text):
        """数据类型改变时的回调"""
        is_sim = (data_type_text == "仿真")
        self.sim_params_group.setEnabled(is_sim)
        self.raw_data_group.setEnabled(not is_sim)
        
        # 控制高级配置中仿真/实采参数组的显示
        if hasattr(self, 'sim_data_gen_group'):
            self.sim_data_gen_group.setVisible(is_sim)
        if hasattr(self, 'real_data_gen_group'):
            self.real_data_gen_group.setVisible(not is_sim)
        
    def load_config_file(self):
        """加载配置文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择配置文件", str(self.project_root / "configs"),
            "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            self.load_config(file_path)
            
    def load_config(self, config_path):
        """加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            self.current_config_path = config_path
            self.config_path_edit.setText(config_path)
            self.is_new_config = False
            
            # 隐藏配置标签页，显示重要参数（只读）
            self.config_tabs.setVisible(False)
            self.important_params_group.setVisible(True)
            self.delete_btn.setEnabled(True)
            
            # 提取并显示重要参数（只读）
            self.display_important_params()
            
            QMessageBox.information(self, "成功", f"配置已加载: {config_path}")
            self.config_updated.emit(config_path)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载配置失败: {str(e)}")
            
    def display_important_params(self):
        """显示重要参数（只读）"""
        # 清除之前的显示
        while self.important_params_layout.count():
            child = self.important_params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        if not self.config:
            return
            
        # 提取参数
        data_type = "未知"
        modulation = "未知"
        amp_ratio = "未知"
        snr = "未知"
        sim_params = {}
        
        if 'data_generation' in self.config:
            if 'generate_sim' in self.config['data_generation']:
                data_type = "仿真"
                sim_config = self.config['data_generation']['generate_sim']
                modulation = sim_config.get('modulation1', '未知')
                snr_val = sim_config.get('snr_db', 0)
                if isinstance(snr_val, list):
                    snr = f"{snr_val[0]}-{snr_val[1]}"
                else:
                    snr = str(snr_val)
                amp_val = sim_config.get('amp_ratio', 0.7)
                if isinstance(amp_val, list):
                    amp_ratio = f"{amp_val[0]}-{amp_val[1]}"
                else:
                    amp_ratio = str(amp_val)
                # 提取仿真参数（支持范围值）
                freq1 = sim_config.get('freq_offset1', 0)
                freq1_str = f"{freq1[0]}-{freq1[1]}" if isinstance(freq1, list) and len(freq1) == 2 else str(freq1)
                
                freq2 = sim_config.get('freq_offset2', 0)
                freq2_str = f"{freq2[0]}-{freq2[1]}" if isinstance(freq2, list) and len(freq2) == 2 else str(freq2)
                
                phase1 = sim_config.get('phase1', 0)
                phase1_str = f"{phase1[0]}-{phase1[1]}" if isinstance(phase1, list) and len(phase1) == 2 else str(phase1)
                
                phase2 = sim_config.get('phase2', 0)
                phase2_str = f"{phase2[0]}-{phase2[1]}" if isinstance(phase2, list) and len(phase2) == 2 else str(phase2)
                
                delay1 = sim_config.get('delay1_samp', 0)
                delay1_str = f"{delay1[0]}-{delay1[1]}" if isinstance(delay1, list) and len(delay1) == 2 else str(delay1)
                
                delay2 = sim_config.get('delay2_samp', 0)
                delay2_str = f"{delay2[0]}-{delay2[1]}" if isinstance(delay2, list) and len(delay2) == 2 else str(delay2)
                
                sim_params = {
                    'freq_offset1': freq1_str,
                    'freq_offset2': freq2_str,
                    'phase1': phase1_str,
                    'phase2': phase2_str,
                    'delay1': delay1_str,
                    'delay2': delay2_str,
                }
            elif 'generate_mixed' in self.config['data_generation']:
                data_type = "实采"
                mixed_config = self.config['data_generation']['generate_mixed']
                modulation = mixed_config.get('modulation', '未知')
                snr = str(mixed_config.get('target_snr_db', 0))
                amp_range = mixed_config.get('amp_range', [0.7, 0.7])
                amp_ratio = f"{amp_range[0]}-{amp_range[1]}"
        
        # 显示参数
        self.important_params_layout.addRow("数据类型:", QLabel(data_type))
        self.important_params_layout.addRow("调制方式:", QLabel(str(modulation)))
        self.important_params_layout.addRow("幅度比:", QLabel(str(amp_ratio)))
        self.important_params_layout.addRow("信噪比 (dB):", QLabel(str(snr)))
        
        if data_type == "仿真":
            self.important_params_layout.addRow("频偏1 (Hz):", QLabel(str(sim_params.get('freq_offset1', '-'))))
            self.important_params_layout.addRow("频偏2 (Hz):", QLabel(str(sim_params.get('freq_offset2', '-'))))
            self.important_params_layout.addRow("相偏1 (π):", QLabel(str(sim_params.get('phase1', '-'))))
            self.important_params_layout.addRow("相偏2 (π):", QLabel(str(sim_params.get('phase2', '-'))))
            self.important_params_layout.addRow("时延差1:", QLabel(str(sim_params.get('delay1', '-'))))
            self.important_params_layout.addRow("时延差2:", QLabel(str(sim_params.get('delay2', '-'))))
            
    def create_new_config(self):
        """创建新配置文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存配置文件", str(self.project_root / "configs"),
            "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            # 确保文件有.yaml扩展名
            file_path_obj = Path(file_path)
            if file_path_obj.suffix.lower() not in ['.yaml', '.yml']:
                file_path = str(file_path_obj.with_suffix('.yaml'))
            
            # 从基础配置创建
            base_config = self.project_root / "configs" / "base_config.yaml"
            if base_config.exists():
                shutil.copy(base_config, file_path)
                self.current_config_path = file_path
                self.config_path_edit.setText(file_path)
                self.is_new_config = True
                
                # 加载基础配置
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                
                # 显示配置标签页，隐藏重要参数显示
                self.config_tabs.setVisible(True)
                self.important_params_group.setVisible(False)
                self.delete_btn.setEnabled(False)
                
                # 加载默认值到UI
                self.load_config_to_ui()
                
                QMessageBox.information(self, "成功", f"新配置文件已创建，请填写参数后保存")
            else:
                QMessageBox.warning(self, "警告", "未找到基础配置文件 base_config.yaml")
                
    def load_config_to_ui(self):
        """将配置加载到UI（用于新建配置时）"""
        if not self.config:
            return
            
        # 判断数据类型并加载参数
        if 'data_generation' in self.config:
            if 'generate_sim' in self.config['data_generation']:
                self.data_type_combo.setCurrentText("仿真")
                sim_config = self.config['data_generation']['generate_sim']
                
                # 加载调制方式
                mod = sim_config.get('modulation1', '8PSK')
                idx = self.modulation_combo.findText(mod)
                if idx >= 0:
                    self.modulation_combo.setCurrentIndex(idx)
                
                # 加载信噪比（范围值）
                snr_val = sim_config.get('snr_db', 15.0)
                if isinstance(snr_val, list) and len(snr_val) == 2:
                    self.snr_min_spin.setValue(float(snr_val[0]))
                    self.snr_max_spin.setValue(float(snr_val[1]))
                elif isinstance(snr_val, str):
                    try:
                        val = float(snr_val)
                        self.snr_min_spin.setValue(val)
                        self.snr_max_spin.setValue(val)
                    except ValueError:
                        self.snr_min_spin.setValue(15.0)
                        self.snr_max_spin.setValue(15.0)
                else:
                    val = float(snr_val)
                    self.snr_min_spin.setValue(val)
                    self.snr_max_spin.setValue(val)
                
                # 加载幅度比（如果是列表，取平均值）
                amp_val = sim_config.get('amp_ratio', 0.7)
                if isinstance(amp_val, list) and len(amp_val) == 2:
                    self.amp_ratio_spin.setValue(float((amp_val[0] + amp_val[1]) / 2))
                elif isinstance(amp_val, str):
                    try:
                        self.amp_ratio_spin.setValue(float(amp_val))
                    except ValueError:
                        self.amp_ratio_spin.setValue(0.7)
                else:
                    self.amp_ratio_spin.setValue(float(amp_val))
                
                # 加载仿真参数（范围值）
                # 频偏1
                freq1 = sim_config.get('freq_offset1', 0.0)
                if isinstance(freq1, list) and len(freq1) == 2:
                    self.freq_offset1_min_spin.setValue(float(freq1[0]))
                    self.freq_offset1_max_spin.setValue(float(freq1[1]))
                elif isinstance(freq1, str):
                    try:
                        val = float(freq1)
                        self.freq_offset1_min_spin.setValue(val)
                        self.freq_offset1_max_spin.setValue(val)
                    except ValueError:
                        self.freq_offset1_min_spin.setValue(0.0)
                        self.freq_offset1_max_spin.setValue(0.0)
                else:
                    val = float(freq1)
                    self.freq_offset1_min_spin.setValue(val)
                    self.freq_offset1_max_spin.setValue(val)
                    
                # 频偏2
                freq2 = sim_config.get('freq_offset2', 0.0)
                if isinstance(freq2, list) and len(freq2) == 2:
                    self.freq_offset2_min_spin.setValue(float(freq2[0]))
                    self.freq_offset2_max_spin.setValue(float(freq2[1]))
                elif isinstance(freq2, str):
                    try:
                        val = float(freq2)
                        self.freq_offset2_min_spin.setValue(val)
                        self.freq_offset2_max_spin.setValue(val)
                    except ValueError:
                        self.freq_offset2_min_spin.setValue(0.0)
                        self.freq_offset2_max_spin.setValue(0.0)
                else:
                    val = float(freq2)
                    self.freq_offset2_min_spin.setValue(val)
                    self.freq_offset2_max_spin.setValue(val)
                    
                # 相偏1
                phase1 = sim_config.get('phase1', 0.0)
                if isinstance(phase1, list) and len(phase1) == 2:
                    self.phase1_min_spin.setValue(float(phase1[0]))
                    self.phase1_max_spin.setValue(float(phase1[1]))
                elif isinstance(phase1, str):
                    try:
                        val = float(phase1)
                        self.phase1_min_spin.setValue(val)
                        self.phase1_max_spin.setValue(val)
                    except ValueError:
                        self.phase1_min_spin.setValue(0.0)
                        self.phase1_max_spin.setValue(0.0)
                else:
                    val = float(phase1)
                    self.phase1_min_spin.setValue(val)
                    self.phase1_max_spin.setValue(val)
                    
                # 相偏2
                phase2 = sim_config.get('phase2', 0.0)
                if isinstance(phase2, list) and len(phase2) == 2:
                    self.phase2_min_spin.setValue(float(phase2[0]))
                    self.phase2_max_spin.setValue(float(phase2[1]))
                elif isinstance(phase2, str):
                    try:
                        val = float(phase2)
                        self.phase2_min_spin.setValue(val)
                        self.phase2_max_spin.setValue(val)
                    except ValueError:
                        self.phase2_min_spin.setValue(0.0)
                        self.phase2_max_spin.setValue(0.0)
                else:
                    val = float(phase2)
                    self.phase2_min_spin.setValue(val)
                    self.phase2_max_spin.setValue(val)
                    
                # 时延差1
                delay1 = sim_config.get('delay1_samp', 0)
                if isinstance(delay1, list) and len(delay1) == 2:
                    self.delay1_min_spin.setValue(int(delay1[0]))
                    self.delay1_max_spin.setValue(int(delay1[1]))
                elif isinstance(delay1, str):
                    try:
                        val = int(float(delay1))
                        self.delay1_min_spin.setValue(val)
                        self.delay1_max_spin.setValue(val)
                    except ValueError:
                        self.delay1_min_spin.setValue(0)
                        self.delay1_max_spin.setValue(0)
                else:
                    val = int(delay1)
                    self.delay1_min_spin.setValue(val)
                    self.delay1_max_spin.setValue(val)
                    
                # 时延差2
                delay2 = sim_config.get('delay2_samp', 0)
                if isinstance(delay2, list) and len(delay2) == 2:
                    self.delay2_min_spin.setValue(int(delay2[0]))
                    self.delay2_max_spin.setValue(int(delay2[1]))
                elif isinstance(delay2, str):
                    try:
                        val = int(float(delay2))
                        self.delay2_min_spin.setValue(val)
                        self.delay2_max_spin.setValue(val)
                    except ValueError:
                        self.delay2_min_spin.setValue(0)
                        self.delay2_max_spin.setValue(0)
                else:
                    val = int(delay2)
                    self.delay2_min_spin.setValue(val)
                    self.delay2_max_spin.setValue(val)
                
                # 加载保存路径
                self.data_save_dir_edit.setText(sim_config.get('save_dir', ''))
                
            elif 'generate_mixed' in self.config['data_generation']:
                self.data_type_combo.setCurrentText("实采")
                mixed_config = self.config['data_generation']['generate_mixed']
                
                # 加载调制方式
                mod = mixed_config.get('modulation', '8PSK')
                idx = self.modulation_combo.findText(mod.upper())
                if idx >= 0:
                    self.modulation_combo.setCurrentIndex(idx)
                
                # 加载信噪比（实采数据是固定值）
                snr_val = mixed_config.get('target_snr_db', 15.0)
                if isinstance(snr_val, str):
                    try:
                        val = float(snr_val)
                        self.snr_min_spin.setValue(val)
                        self.snr_max_spin.setValue(val)
                    except ValueError:
                        self.snr_min_spin.setValue(15.0)
                        self.snr_max_spin.setValue(15.0)
                else:
                    val = float(snr_val)
                    self.snr_min_spin.setValue(val)
                    self.snr_max_spin.setValue(val)
                
                # 加载幅度比
                amp_range = mixed_config.get('amp_range', [0.7, 0.7])
                if isinstance(amp_range, list) and len(amp_range) == 2:
                    self.amp_ratio_spin.setValue(float((amp_range[0] + amp_range[1]) / 2))
                else:
                    self.amp_ratio_spin.setValue(0.7)
                
                # 加载输出路径（提取基础路径，去掉参数后缀）
                output_dir = mixed_config.get('output_dir', '')
                if output_dir:
                    # 尝试提取基础路径（去掉参数后缀）
                    import re
                    # 移除参数后缀：/{MODULATION}_{sim|real}_{SNR}dB_{AMP}
                    base_dir = re.sub(r'/[A-Z0-9]+_(sim|real)_\d+dB_\d+\.\d+$', '', output_dir)
                    self.data_save_dir_edit.setText(base_dir if base_dir != output_dir else output_dir)
        
        # 加载原始数据路径
        if 'data_generation' in self.config and 'raw_data' in self.config['data_generation']:
            raw_paths = self.config['data_generation']['raw_data'].get('paths', {})
            self.raw_qpsk_edit.setText(raw_paths.get('QPSK', ''))
            self.raw_8psk_edit.setText(raw_paths.get('8PSK', ''))
            self.raw_16qam_edit.setText(raw_paths.get('16QAM', ''))
        
        # 加载模型保存路径（提取基础路径，去掉可能的后缀）
        if 'training' in self.config:
            model_dir = self.config['training'].get('output_dir', '')
            if model_dir:
                # 移除常见的后缀模式
                import re
                base_dir = re.sub(r'-[A-Z0-9]+_(sim|real)_\d+dB_\d+\.\d+$', '', model_dir)
                self.model_save_dir_edit.setText(base_dir if base_dir != model_dir else model_dir)
        
        # 加载推理结果保存路径（提取基础路径）
        if 'sampling' in self.config:
            inference_dir = self.config['sampling'].get('output_dir', '')
            if inference_dir:
                # 提取基础路径（最后一个/之前的部分）
                parts = inference_dir.rsplit('/', 1)
                if len(parts) == 2:
                    # 检查最后一部分是否是参数后缀
                    import re
                    if re.match(r'^[A-Z0-9]+_(sim|real)_\d+dB_\d+\.\d+$', parts[1]):
                        self.inference_save_dir_edit.setText(parts[0])
                    else:
                        self.inference_save_dir_edit.setText(inference_dir)
                else:
                    self.inference_save_dir_edit.setText(inference_dir)
        
        # 加载高级配置
        if 'training' in self.config:
            epochs = self.config['training'].get('num_epochs', 500)
            self.epochs_spin.setValue(int(epochs) if isinstance(epochs, (int, float, str)) else 500)
            
            batch_size = self.config['training'].get('train_batch_size', 64)
            self.batch_size_spin.setValue(int(batch_size) if isinstance(batch_size, (int, float, str)) else 64)
            
            lr = self.config['training'].get('learning_rate', 2e-4)
            # 处理字符串形式的科学计数法，如 "2e-4"
            if isinstance(lr, str):
                try:
                    lr = float(lr)
                except ValueError:
                    lr = 2e-4
            self.lr_spin.setValue(float(lr))
        
        if 'sampling' in self.config:
            steps = self.config['sampling'].get('num_inference_steps', 100)
            self.inference_steps_spin.setValue(int(steps) if isinstance(steps, (int, float, str)) else 100)
            
            eta = self.config['sampling'].get('eta', 0.15)
            self.eta_spin.setValue(float(eta) if isinstance(eta, (int, float, str)) else 0.15)
        
        # 加载max_slices_per_file
        if 'data_generation' in self.config and 'split' in self.config['data_generation']:
            max_slices = self.config['data_generation']['split'].get('max_slices_per_file', 10000)
            if max_slices is None:
                self.max_slices_spin.setValue(10000)  # null表示全部，但UI中显示10000
            else:
                self.max_slices_spin.setValue(int(max_slices) if isinstance(max_slices, (int, float, str)) else 10000)
        else:
            self.max_slices_spin.setValue(10000)
        
        # 加载高级配置参数
        if 'data_generation' in self.config:
            # num_samples (仿真数据)
            if 'generate_sim' in self.config['data_generation']:
                num_samples = self.config['data_generation']['generate_sim'].get('num_samples', 100000)
                self.num_samples_spin.setValue(int(num_samples) if isinstance(num_samples, (int, float, str)) else 100000)
                
                shard_size = self.config['data_generation']['generate_sim'].get('shard_size', 10000)
                self.shard_size_spin.setValue(int(shard_size) if isinstance(shard_size, (int, float, str)) else 10000)
            
            # max_samples (split配置)
            if 'split' in self.config['data_generation']:
                max_samples = self.config['data_generation']['split'].get('max_samples', 1000000)
                if max_samples is None:
                    self.max_samples_spin.setValue(1000000)  # null表示全部，但UI中显示1000000
                else:
                    self.max_samples_spin.setValue(int(max_samples) if isinstance(max_samples, (int, float, str)) else 1000000)
                
                threshold = self.config['data_generation']['split'].get('threshold', 6.0)
                self.threshold_spin.setValue(float(threshold) if isinstance(threshold, (int, float, str)) else 6.0)
                
                train_ratio = self.config['data_generation']['split'].get('train_ratio', 0.9)
                self.train_ratio_spin.setValue(float(train_ratio) if isinstance(train_ratio, (int, float, str)) else 0.9)
            
            # target_pairs, test_target_pairs, sps, samples_per_file (generate_mixed配置)
            if 'generate_mixed' in self.config['data_generation']:
                target_pairs = self.config['data_generation']['generate_mixed'].get('target_pairs', 100000)
                self.target_pairs_spin.setValue(int(target_pairs) if isinstance(target_pairs, (int, float, str)) else 100000)
                
                test_target_pairs = self.config['data_generation']['generate_mixed'].get('test_target_pairs', None)
                if test_target_pairs is None:
                    self.test_target_pairs_spin.setValue(0)  # 0表示null
                else:
                    self.test_target_pairs_spin.setValue(int(test_target_pairs) if isinstance(test_target_pairs, (int, float, str)) else 0)
                
                sps = self.config['data_generation']['generate_mixed'].get('sps', 8)
                self.sps_spin.setValue(int(sps) if isinstance(sps, (int, float, str)) else 8)
                
                samples_per_file = self.config['data_generation']['generate_mixed'].get('samples_per_file', 30)
                self.samples_per_file_spin.setValue(int(samples_per_file) if isinstance(samples_per_file, (int, float, str)) else 30)
        
        # 加载预训练模型路径
        if 'training' in self.config:
            pretrained = self.config['training'].get('pretrained', None)
            if pretrained is None or pretrained == 'null' or pretrained == '':
                self.pretrained_edit.setText('')
            else:
                self.pretrained_edit.setText(str(pretrained))
        
        # 加载训练高级参数
        if 'training' in self.config:
            test_batch_size = self.config['training'].get('test_batch_size', 64)
            self.test_batch_size_spin.setValue(int(test_batch_size) if isinstance(test_batch_size, (int, float, str)) else 64)
            
            gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
            self.gradient_accumulation_steps_spin.setValue(int(gradient_accumulation_steps) if isinstance(gradient_accumulation_steps, (int, float, str)) else 1)
            
            lr_warmup_steps = self.config['training'].get('lr_warmup_steps', 200)
            self.lr_warmup_steps_spin.setValue(int(lr_warmup_steps) if isinstance(lr_warmup_steps, (int, float, str)) else 200)
            
            save_data_epochs = self.config['training'].get('save_data_epochs', 1)
            self.save_data_epochs_spin.setValue(int(save_data_epochs) if isinstance(save_data_epochs, (int, float, str)) else 1)
            
            save_model_epochs = self.config['training'].get('save_model_epochs', 1)
            self.save_model_epochs_spin.setValue(int(save_model_epochs) if isinstance(save_model_epochs, (int, float, str)) else 1)
        
        # 加载模型参数
        if 'model' in self.config:
            # predict_target固定为x0，不需要加载
            num_diffusion_timesteps = self.config['model'].get('num_diffusion_timesteps', 1000)
            self.num_diffusion_timesteps_spin.setValue(int(num_diffusion_timesteps) if isinstance(num_diffusion_timesteps, (int, float, str)) else 1000)
        
        if 'data' in self.config:
            signal_len = self.config['data'].get('signal_len', 3072)
            self.signal_len_spin.setValue(int(signal_len) if isinstance(signal_len, (int, float, str)) else 3072)
        
        # 加载num_workers配置（从generate_mixed或split中读取，优先使用generate_mixed）
        num_workers = None
        if 'data_generation' in self.config:
            if 'generate_mixed' in self.config['data_generation']:
                num_workers = self.config['data_generation']['generate_mixed'].get('num_workers', None)
            if num_workers is None and 'split' in self.config['data_generation']:
                num_workers = self.config['data_generation']['split'].get('num_workers', None)
        
        if num_workers is None:
            self.num_workers_spin.setValue(0)  # 0表示null
        else:
            self.num_workers_spin.setValue(int(num_workers) if isinstance(num_workers, (int, float, str)) else 0)
        
        # 根据数据类型显示/隐藏相应的参数组
        data_type_text = self.data_type_combo.currentText()
        is_sim = (data_type_text == "仿真")
        if hasattr(self, 'sim_data_gen_group'):
            self.sim_data_gen_group.setVisible(is_sim)
        if hasattr(self, 'real_data_gen_group'):
            self.real_data_gen_group.setVisible(not is_sim)
            
    def save_config(self):
        """保存配置"""
        if not self.config or not self.current_config_path:
            QMessageBox.warning(self, "警告", "请先创建配置文件")
            return
            
        if not self.is_new_config:
            QMessageBox.warning(self, "警告", "已加载的配置文件不允许修改，请新建配置文件")
            return
            
        try:
            # 获取参数
            data_type_text = self.data_type_combo.currentText()
            is_sim = (data_type_text == "仿真")
            modulation = self.modulation_combo.currentText()
            amp_ratio = self.amp_ratio_spin.value()
            # 对于输出目录名称，使用信噪比的平均值
            snr_avg = (self.snr_min_spin.value() + self.snr_max_spin.value()) / 2
            snr = snr_avg
            
            # 生成输出目录名称（根据参数）
            output_suffix = f"{modulation}_{'sim' if is_sim else 'real'}_{snr:.0f}dB_{amp_ratio:.1f}"
            
            # 更新配置
            if is_sim:
                # 仿真数据配置
                if 'generate_sim' not in self.config.get('data_generation', {}):
                    if 'data_generation' not in self.config:
                        self.config['data_generation'] = {}
                    self.config['data_generation']['generate_sim'] = {}
                    
                sim_config = self.config['data_generation']['generate_sim']
                sim_config['modulation1'] = modulation
                sim_config['modulation2'] = modulation
                
                # 保存信噪比（范围值，如果相等则保存为固定值）
                snr_min = self.snr_min_spin.value()
                snr_max = self.snr_max_spin.value()
                if snr_min == snr_max:
                    sim_config['snr_db'] = snr_min
                else:
                    sim_config['snr_db'] = [snr_min, snr_max]
                
                sim_config['amp_ratio'] = amp_ratio
                
                # 保存频偏1（范围值）
                freq1_min = self.freq_offset1_min_spin.value()
                freq1_max = self.freq_offset1_max_spin.value()
                if freq1_min == freq1_max:
                    sim_config['freq_offset1'] = freq1_min
                else:
                    sim_config['freq_offset1'] = [freq1_min, freq1_max]
                
                # 保存频偏2（范围值）
                freq2_min = self.freq_offset2_min_spin.value()
                freq2_max = self.freq_offset2_max_spin.value()
                if freq2_min == freq2_max:
                    sim_config['freq_offset2'] = freq2_min
                else:
                    sim_config['freq_offset2'] = [freq2_min, freq2_max]
                
                # 保存相偏1（范围值）
                phase1_min = self.phase1_min_spin.value()
                phase1_max = self.phase1_max_spin.value()
                if phase1_min == phase1_max:
                    sim_config['phase1'] = phase1_min
                else:
                    sim_config['phase1'] = [phase1_min, phase1_max]
                
                # 保存相偏2（范围值）
                phase2_min = self.phase2_min_spin.value()
                phase2_max = self.phase2_max_spin.value()
                if phase2_min == phase2_max:
                    sim_config['phase2'] = phase2_min
                else:
                    sim_config['phase2'] = [phase2_min, phase2_max]
                
                # 保存时延差1（范围值）
                delay1_min = self.delay1_min_spin.value()
                delay1_max = self.delay1_max_spin.value()
                if delay1_min == delay1_max:
                    sim_config['delay1_samp'] = delay1_min
                else:
                    sim_config['delay1_samp'] = [delay1_min, delay1_max]
                
                # 保存时延差2（范围值）
                delay2_min = self.delay2_min_spin.value()
                delay2_max = self.delay2_max_spin.value()
                if delay2_min == delay2_max:
                    sim_config['delay2_samp'] = delay2_min
                else:
                    sim_config['delay2_samp'] = [delay2_min, delay2_max]
                
                # 更新保存路径（在基础目录下创建参数子目录）
                base_save_dir = self.data_save_dir_edit.text() or "/nas/datasets/yixin/PCMA/sim_data"
                # 确保路径以参数子目录结尾
                if not base_save_dir.endswith(output_suffix):
                    sim_config['save_dir'] = f"{base_save_dir}/{output_suffix}"
                else:
                    sim_config['save_dir'] = base_save_dir
                
                # 清除实采数据配置
                if 'generate_mixed' in self.config['data_generation']:
                    del self.config['data_generation']['generate_mixed']
                if 'split' in self.config['data_generation']:
                    del self.config['data_generation']['split']
            else:
                # 实采数据配置
                if 'generate_mixed' not in self.config.get('data_generation', {}):
                    if 'data_generation' not in self.config:
                        self.config['data_generation'] = {}
                    self.config['data_generation']['generate_mixed'] = {}
                    if 'split' not in self.config['data_generation']:
                        self.config['data_generation']['split'] = {}
                        
                mixed_config = self.config['data_generation']['generate_mixed']
                mixed_config['modulation'] = modulation
                # 实采数据的信噪比是固定值（使用平均值）
                mixed_config['target_snr_db'] = snr
                mixed_config['amp_range'] = [amp_ratio, amp_ratio]
                
                # 更新输出路径（根据参数自动生成）
                base_output_dir = self.data_save_dir_edit.text() or f"/nas/datasets/yixin/PCMA/real_data/{modulation.lower()}"
                # 如果基础路径已经包含后缀，则直接使用；否则添加后缀
                if output_suffix not in base_output_dir:
                    mixed_config['output_dir'] = f"{base_output_dir}/{output_suffix}"
                else:
                    mixed_config['output_dir'] = base_output_dir
                    
                # 更新split配置的输出目录
                if 'split' in self.config['data_generation']:
                    split_output_dir = base_output_dir.rsplit('/', 1)[0] if '/' in base_output_dir else base_output_dir
                    self.config['data_generation']['split']['output_dir'] = split_output_dir
                
                # 更新原始数据路径
                if 'raw_data' not in self.config['data_generation']:
                    self.config['data_generation']['raw_data'] = {'paths': {}}
                self.config['data_generation']['raw_data']['paths'] = {
                    'QPSK': self.raw_qpsk_edit.text(),
                    '8PSK': self.raw_8psk_edit.text(),
                    '16QAM': self.raw_16qam_edit.text()
                }
                
                # 清除仿真数据配置
                if 'generate_sim' in self.config['data_generation']:
                    del self.config['data_generation']['generate_sim']
            
            # 更新data配置
            if 'data' not in self.config:
                self.config['data'] = {}
            self.config['data']['modulation'] = modulation
            
            # 更新训练/测试数据路径，使其匹配生成的数据路径
            if is_sim:
                # 仿真数据：前9个shard用于训练，最后一个用于测试
                data_save_dir = sim_config.get('save_dir', f"/nas/datasets/yixin/PCMA/sim_data/{output_suffix}")
                # 使用base + shard_list + pattern方式
                self.config['data']['train'] = {
                    'base': data_save_dir,
                    'shard_list': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'pattern': f'{modulation}-{modulation}_*_shard{{idx:02d}}_of*_*.pth'
                }
                self.config['data']['test'] = {
                    'base': data_save_dir,
                    'shard_list': [10],  # 最后一个shard用于测试
                    'pattern': f'{modulation}-{modulation}_*_shard{{idx:02d}}_of*_*.pth'
                }
            else:
                # 实采数据：train和test目录
                data_output_dir = mixed_config.get('output_dir', f"/nas/datasets/yixin/PCMA/real_data/{modulation.lower()}/{output_suffix}")
                # 实采数据文件名格式：real_{modulation}_mixed_amp{amp}_shard{idx:02d}_of{total:02d}_c128.pth
                # 使用通配符pattern，匹配所有shard文件
                amp_str = f"amp{amp_ratio:.1f}"
                pattern = f'real_{modulation.lower()}_mixed_{amp_str}_shard{{idx:02d}}_of*_c128.pth'
                self.config['data']['train'] = {
                    'base': f'{data_output_dir}/train',
                    'shard_list': [],  # 将在数据生成后自动更新（通过update_data_paths.py或手动）
                    'pattern': pattern
                }
                self.config['data']['test'] = {
                    'base': f'{data_output_dir}/test',
                    'shard_list': [],  # 将在数据生成后自动更新（通过update_data_paths.py或手动）
                    'pattern': pattern
                }
            
            # 更新训练输出目录（在基础目录下创建参数子目录）
            base_model_dir = self.model_save_dir_edit.text() or "results/DDPM-PCMA"
            # 确保路径以参数子目录结尾
            if not base_model_dir.endswith(output_suffix):
                self.config['training']['output_dir'] = f"{base_model_dir}/{output_suffix}"
            else:
                self.config['training']['output_dir'] = base_model_dir
            
            # 更新推理输出目录（在基础目录下创建参数子目录）
            base_inference_dir = self.inference_save_dir_edit.text() or "/nas/datasets/yixin/PCMA/diffusion_prediction"
            # 确保路径以参数子目录结尾
            if not base_inference_dir.endswith(output_suffix):
                self.config['sampling']['output_dir'] = f"{base_inference_dir}/{output_suffix}"
            else:
                self.config['sampling']['output_dir'] = base_inference_dir
            
            # 更新训练和推理参数（简易配置）
            self.config['training']['num_epochs'] = self.epochs_spin.value()
            self.config['training']['train_batch_size'] = self.batch_size_spin.value()
            self.config['training']['learning_rate'] = self.lr_spin.value()
            # 更新预训练模型路径
            pretrained_path = self.pretrained_edit.text().strip()
            if pretrained_path:
                # 如果路径不以/unet/结尾，尝试添加（但保持用户输入的格式）
                if not pretrained_path.endswith('/unet/') and not pretrained_path.endswith('/unet'):
                    # 如果路径以/结尾，添加unet/；否则添加/unet/
                    if pretrained_path.endswith('/'):
                        pretrained_path = pretrained_path + 'unet/'
                    else:
                        pretrained_path = pretrained_path + '/unet/'
                elif pretrained_path.endswith('/unet'):
                    # 如果以/unet结尾但没有最后的/，添加/
                    pretrained_path = pretrained_path + '/'
                self.config['training']['pretrained'] = pretrained_path
            else:
                self.config['training']['pretrained'] = None
            self.config['sampling']['num_inference_steps'] = self.inference_steps_spin.value()
            self.config['sampling']['eta'] = self.eta_spin.value()
            
            # 更新高级配置参数
            if is_sim:
                # 仿真数据参数
                if 'generate_sim' not in self.config['data_generation']:
                    self.config['data_generation']['generate_sim'] = {}
                self.config['data_generation']['generate_sim']['num_samples'] = self.num_samples_spin.value()
                self.config['data_generation']['generate_sim']['shard_size'] = self.shard_size_spin.value()
            else:
                # 实采数据参数
                if 'split' not in self.config['data_generation']:
                    self.config['data_generation']['split'] = {}
                max_samples_val = self.max_samples_spin.value()
                self.config['data_generation']['split']['max_samples'] = max_samples_val if max_samples_val > 0 else None
                # max_slices_per_file
                max_slices_val = self.max_slices_spin.value()
                self.config['data_generation']['split']['max_slices_per_file'] = max_slices_val if max_slices_val > 0 else None
                self.config['data_generation']['split']['threshold'] = self.threshold_spin.value()
                self.config['data_generation']['split']['train_ratio'] = self.train_ratio_spin.value()
                
                if 'generate_mixed' not in self.config['data_generation']:
                    self.config['data_generation']['generate_mixed'] = {}
                self.config['data_generation']['generate_mixed']['target_pairs'] = self.target_pairs_spin.value()
                test_target_pairs_val = self.test_target_pairs_spin.value()
                self.config['data_generation']['generate_mixed']['test_target_pairs'] = test_target_pairs_val if test_target_pairs_val > 0 else None
                self.config['data_generation']['generate_mixed']['sps'] = self.sps_spin.value()
                self.config['data_generation']['generate_mixed']['samples_per_file'] = self.samples_per_file_spin.value()
            
            # 更新训练高级参数
            self.config['training']['test_batch_size'] = self.test_batch_size_spin.value()
            self.config['training']['gradient_accumulation_steps'] = self.gradient_accumulation_steps_spin.value()
            self.config['training']['lr_warmup_steps'] = self.lr_warmup_steps_spin.value()
            self.config['training']['save_data_epochs'] = self.save_data_epochs_spin.value()
            self.config['training']['save_model_epochs'] = self.save_model_epochs_spin.value()
            
            # 更新模型参数
            if 'model' not in self.config:
                self.config['model'] = {}
            # predict_target固定为x0
            self.config['model']['predict_target'] = 'x0'
            self.config['model']['num_diffusion_timesteps'] = self.num_diffusion_timesteps_spin.value()
            
            # 更新data参数
            if 'data' not in self.config:
                self.config['data'] = {}
            self.config['data']['signal_len'] = self.signal_len_spin.value()
            
            # 更新num_workers配置（同时设置到generate_mixed和split）
            num_workers_val = self.num_workers_spin.value()
            num_workers_final = num_workers_val if num_workers_val > 0 else None
            
            # 设置到generate_mixed
            if 'data_generation' in self.config and 'generate_mixed' in self.config['data_generation']:
                self.config['data_generation']['generate_mixed']['num_workers'] = num_workers_final
            
            # 设置到split
            if 'data_generation' in self.config and 'split' in self.config['data_generation']:
                self.config['data_generation']['split']['num_workers'] = num_workers_final
                
            # 保存到文件
            with open(self.current_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, 
                         allow_unicode=True, sort_keys=False)
            
            # 保存后，切换为只读模式
            self.is_new_config = False
            self.config_tabs.setVisible(False)
            self.important_params_group.setVisible(True)
            self.delete_btn.setEnabled(True)
            self.display_important_params()
            
            QMessageBox.information(self, "成功", "配置已保存")
            self.config_updated.emit(self.current_config_path)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存配置失败: {str(e)}")
            
    def delete_config_file(self):
        """删除配置文件"""
        if not self.current_config_path:
            return
            
        reply = QMessageBox.question(
            self, "确认删除", 
            f"确定要删除配置文件吗？\n{self.current_config_path}",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                os.remove(self.current_config_path)
                self.current_config_path = None
                self.config = None
                self.config_path_edit.clear()
                self.important_params_group.setVisible(False)
                self.config_tabs.setVisible(False)
                self.delete_btn.setEnabled(False)
                QMessageBox.information(self, "成功", "配置文件已删除")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除配置文件失败: {str(e)}")
            
    def get_config_path(self):
        """获取当前配置文件路径"""
        return self.current_config_path
