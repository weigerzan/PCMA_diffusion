#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解调输出标签页
"""

import sys
import json
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QLineEdit, QGroupBox, QMessageBox,
                             QProgressBar, QTableWidget, QTableWidgetItem,
                             QHeaderView, QFileDialog, QDialog, QFormLayout,
                             QDialogButtonBox, QTextEdit)
from PyQt5.QtCore import Qt
from gui.utils.worker_thread import WorkerThread


class DemodulationTab(QWidget):
    """解调输出标签页"""
    
    def __init__(self, output_capture, project_root):
        super().__init__()
        self.output_capture = output_capture
        self.project_root = project_root
        self.current_config_path = None
        self.worker_thread = None
        self.results_data = []  # 存储所有解调结果
        self.results_file = project_root / "demod_test_results" / "results_history.json"
        self.init_ui()
        self.load_results_history()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 说明
        info_label = QLabel("对模型输出进行解调并计算SER（误符号率）。支持从result文件读取已有结果。")
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
        
        # 解调选项
        options_group = QGroupBox("解调选项")
        options_layout = QVBoxLayout()
        
        self.run_demod_btn = QPushButton("运行解调（从推理结果）")
        self.run_demod_btn.clicked.connect(self.run_demodulation)
        options_layout.addWidget(self.run_demod_btn)
        
        self.load_result_btn = QPushButton("从result文件读取结果")
        self.load_result_btn.clicked.connect(self.load_result_file)
        options_layout.addWidget(self.load_result_btn)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # SER结果显示
        ser_group = QGroupBox("SER结果")
        ser_layout = QVBoxLayout()
        
        ser_info_layout = QHBoxLayout()
        self.ser1_label = QLabel("信号1 SER: -")
        self.ser2_label = QLabel("信号2 SER: -")
        self.avg_ser_label = QLabel("平均SER: -")
        ser_info_layout.addWidget(self.ser1_label)
        ser_info_layout.addWidget(self.ser2_label)
        ser_info_layout.addWidget(self.avg_ser_label)
        ser_layout.addLayout(ser_info_layout)
        
        ser_group.setLayout(ser_layout)
        layout.addWidget(ser_group)
        
        # 结果历史记录表格
        history_group = QGroupBox("结果历史记录")
        history_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(14)
        self.results_table.setHorizontalHeaderLabels([
            "时间", "数据类型", "调制方式", "幅度比", "信噪比", 
            "频偏1(Hz)", "频偏2(Hz)", "相偏1(π)", "相偏2(π)", 
            "时延差1", "时延差2", "信号1 SER", "信号2 SER", "平均SER"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.doubleClicked.connect(self.show_result_details)
        history_layout.addWidget(self.results_table)
        
        # 操作按钮
        btn_layout = QHBoxLayout()
        self.delete_btn = QPushButton("删除选中")
        self.delete_btn.clicked.connect(self.delete_selected_result)
        btn_layout.addWidget(self.delete_btn)
        
        self.clear_btn = QPushButton("清空所有")
        self.clear_btn.clicked.connect(self.clear_all_results)
        btn_layout.addWidget(self.clear_btn)
        
        history_layout.addLayout(btn_layout)
        history_group.setLayout(history_layout)
        layout.addWidget(history_group)
        
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
        
    def load_config(self, config_path):
        """加载配置文件"""
        self.current_config_path = config_path
        self.config_edit.setText(config_path)
        
    def run_demodulation(self):
        """运行解调"""
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
        # 注意：脚本会从配置文件中读取调制方式，不需要传递--modulation参数
        config_relative = config_file.name  # 只传文件名
        command = [
            sys.executable,
            str(self.project_root / "test_decoder_ser_multi_process.py"),
            "--config",
            config_relative
        ]
        
        # 创建并启动工作线程
        self.worker_thread = WorkerThread(command, cwd=str(self.project_root))
        self.worker_thread.output_signal.connect(self.on_output)
        self.worker_thread.finished_signal.connect(self.on_demod_finished)
        
        self.status_label.setText("正在解调...")
        self.progress_bar.setVisible(True)
        self.stop_btn.setEnabled(True)
        self.run_demod_btn.setEnabled(False)
        
        self.worker_thread.start()
        
    def get_modulation_from_config(self):
        """从配置文件获取调制方式"""
        try:
            import yaml
            with open(self.current_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('data', {}).get('modulation', '8PSK')
        except:
            return '8PSK'
            
    def extract_config_params(self):
        """从配置文件提取所有相关参数"""
        params = {
            'data_type': 'unknown',  # 'sim' or 'real'
            'modulation': '8PSK',
            'amp_ratio': 0.7,
            'snr': 15.0,
            'freq_offset1': None,
            'freq_offset2': None,
            'phase1': None,
            'phase2': None,
            'delay1': None,
            'delay2': None,
        }
        
        if not self.current_config_path:
            return params
            
        try:
            import yaml
            with open(self.current_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            params['modulation'] = config.get('data', {}).get('modulation', '8PSK')
            
            if 'data_generation' in config:
                # 检查是仿真数据还是实采数据
                if 'generate_sim' in config['data_generation']:
                    params['data_type'] = 'sim'
                    sim_config = config['data_generation']['generate_sim']
                    
                    # 提取幅度比和信噪比
                    amp_ratio = sim_config.get('amp_ratio', 0.7)
                    if isinstance(amp_ratio, list):
                        params['amp_ratio'] = (amp_ratio[0] + amp_ratio[1]) / 2 if len(amp_ratio) == 2 else amp_ratio[0]
                    else:
                        params['amp_ratio'] = amp_ratio
                    
                    snr = sim_config.get('snr_db', 15.0)
                    if isinstance(snr, list):
                        params['snr'] = (snr[0] + snr[1]) / 2 if len(snr) == 2 else snr[0]
                    else:
                        params['snr'] = snr
                    
                    # 提取仿真参数
                    freq_offset1 = sim_config.get('freq_offset1', 0.0)
                    if isinstance(freq_offset1, list):
                        params['freq_offset1'] = (freq_offset1[0] + freq_offset1[1]) / 2 if len(freq_offset1) == 2 else freq_offset1[0]
                    else:
                        params['freq_offset1'] = freq_offset1
                    
                    freq_offset2 = sim_config.get('freq_offset2', 0.0)
                    if isinstance(freq_offset2, list):
                        params['freq_offset2'] = (freq_offset2[0] + freq_offset2[1]) / 2 if len(freq_offset2) == 2 else freq_offset2[0]
                    else:
                        params['freq_offset2'] = freq_offset2
                    
                    phase1 = sim_config.get('phase1', 0.0)
                    if isinstance(phase1, list):
                        params['phase1'] = (phase1[0] + phase1[1]) / 2 if len(phase1) == 2 else phase1[0]
                    else:
                        params['phase1'] = phase1
                    
                    phase2 = sim_config.get('phase2', 0.0)
                    if isinstance(phase2, list):
                        params['phase2'] = (phase2[0] + phase2[1]) / 2 if len(phase2) == 2 else phase2[0]
                    else:
                        params['phase2'] = phase2
                    
                    delay1 = sim_config.get('delay1_samp', 0)
                    if isinstance(delay1, list):
                        params['delay1'] = (delay1[0] + delay1[1]) / 2 if len(delay1) == 2 else delay1[0]
                    else:
                        params['delay1'] = delay1
                    
                    delay2 = sim_config.get('delay2_samp', 0)
                    if isinstance(delay2, list):
                        params['delay2'] = (delay2[0] + delay2[1]) / 2 if len(delay2) == 2 else delay2[0]
                    else:
                        params['delay2'] = delay2
                        
                elif 'generate_mixed' in config['data_generation']:
                    params['data_type'] = 'real'
                    mixed_config = config['data_generation']['generate_mixed']
                    
                    amp_range = mixed_config.get('amp_range', [0.7, 0.7])
                    params['amp_ratio'] = (amp_range[0] + amp_range[1]) / 2
                    params['snr'] = mixed_config.get('target_snr_db', 15.0)
        except Exception as e:
            self.output_capture.output_signal.emit(f"提取配置参数失败: {str(e)}\n")
        
        return params
            
    def on_demod_finished(self, success, message):
        """解调完成回调"""
        self.progress_bar.setVisible(False)
        self.stop_btn.setEnabled(False)
        self.run_demod_btn.setEnabled(True)
        
        if success:
            self.status_label.setText("解调完成")
            # 尝试从输出目录读取结果
            self.load_results_from_output_dir()
            QMessageBox.information(self, "成功", "解调完成")
        else:
            self.status_label.setText(f"解调失败: {message}")
            QMessageBox.critical(self, "失败", f"解调失败: {message}")
            
    def load_results_from_output_dir(self):
        """从输出目录加载结果"""
        try:
            # 从配置文件读取模型输出目录，结果应该在模型目录下的demod_results子目录
            possible_dirs = []
            
            try:
                import yaml
                with open(self.current_config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # 优先查找模型输出目录下的demod_results子目录
                training_output_dir = config.get('training', {}).get('output_dir', '')
                if training_output_dir:
                    model_demod_dir = Path(training_output_dir) / "demod_results"
                    possible_dirs.append(model_demod_dir)
                    self.output_capture.output_signal.emit(f"查找模型解调结果目录: {model_demod_dir}\n")
                
                # 也尝试采样输出目录（旧版本可能保存在这里）
                sampling_output_dir = config.get('sampling', {}).get('output_dir', '')
                if sampling_output_dir:
                    possible_dirs.append(Path(sampling_output_dir))
            except Exception as e:
                self.output_capture.output_signal.emit(f"读取配置文件失败: {str(e)}\n")
            
            # 添加默认目录作为备选
            possible_dirs.extend([
                self.project_root / "results",  # 脚本默认输出目录
                self.project_root / "demod_test_results",  # 脚本参数默认目录
            ])
            
            # 在所有可能目录中查找结果文件
            result_files = []
            for result_dir in possible_dirs:
                if result_dir.exists():
                    found = list(result_dir.glob("demod_result_*.json"))
                    if found:
                        result_files.extend(found)
                        self.output_capture.output_signal.emit(f"在 {result_dir} 找到 {len(found)} 个结果文件\n")
            
            if not result_files:
                self.output_capture.output_signal.emit("未找到结果文件，请检查输出目录\n")
                # 列出搜索过的目录
                self.output_capture.output_signal.emit(f"搜索过的目录: {[str(d) for d in possible_dirs]}\n")
                return
            
            # 汇总所有结果文件
            self.aggregate_results_from_files(result_files)
                
        except Exception as e:
            import traceback
            error_msg = f"加载结果失败: {str(e)}\n{traceback.format_exc()}\n"
            self.output_capture.output_signal.emit(error_msg)
            QMessageBox.warning(self, "警告", f"加载结果失败: {str(e)}")
            
    def aggregate_results_from_files(self, result_files):
        """汇总多个结果文件，计算平均SER"""
        try:
            ser1_list = []
            ser2_list = []
            all_results = []
            
            for result_file in result_files:
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    result1 = data.get('result1', {})
                    result2 = data.get('result2', {})
                    
                    ser1 = result1.get('ser')
                    ser2 = result2.get('ser')
                    
                    if ser1 is not None:
                        ser1_list.append(ser1)
                    if ser2 is not None:
                        ser2_list.append(ser2)
                    
                    all_results.append(data)
                except Exception as e:
                    self.output_capture.output_signal.emit(f"读取文件 {result_file} 失败: {str(e)}\n")
                    continue
            
            if not ser1_list and not ser2_list:
                self.output_capture.output_signal.emit("所有结果文件中都没有有效的SER数据\n")
                return
            
            # 计算平均SER
            avg_ser1 = np.mean(ser1_list) if ser1_list else None
            avg_ser2 = np.mean(ser2_list) if ser2_list else None
            avg_ser = (avg_ser1 + avg_ser2) / 2 if (avg_ser1 is not None and avg_ser2 is not None) else None
            
            # 更新显示
            self.ser1_label.setText(f"信号1 SER: {avg_ser1:.6f}" if avg_ser1 is not None else "信号1 SER: -")
            self.ser2_label.setText(f"信号2 SER: {avg_ser2:.6f}" if avg_ser2 is not None else "信号2 SER: -")
            self.avg_ser_label.setText(f"平均SER: {avg_ser:.6f}" if avg_ser is not None else "平均SER: -")
            
            # 保存到历史记录
            from datetime import datetime
            
            # 从配置文件获取所有参数
            config_params = self.extract_config_params()
            
            # 使用最新的结果文件路径
            latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
            
            result_entry = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'data_type': config_params['data_type'],
                'modulation': config_params['modulation'],
                'amp_ratio': config_params['amp_ratio'],
                'snr': config_params['snr'],
                'freq_offset1': config_params['freq_offset1'],
                'freq_offset2': config_params['freq_offset2'],
                'phase1': config_params['phase1'],
                'phase2': config_params['phase2'],
                'delay1': config_params['delay1'],
                'delay2': config_params['delay2'],
                'ser1': avg_ser1,
                'ser2': avg_ser2,
                'avg_ser': avg_ser,
                'file_path': str(latest_file),
                'num_files': len(result_files),
                'full_data': {
                    'aggregated': True,
                    'num_files': len(result_files),
                    'ser1_list': ser1_list,
                    'ser2_list': ser2_list,
                    'all_results': all_results[:5]  # 只保存前5个结果作为示例
                }
            }
            
            self.results_data.append(result_entry)
            self.save_results_history()
            self.update_results_table()
            
            self.output_capture.output_signal.emit(f"成功加载并汇总 {len(result_files)} 个结果文件\n")
            self.output_capture.output_signal.emit(f"信号1平均SER: {avg_ser1:.6f}\n")
            self.output_capture.output_signal.emit(f"信号2平均SER: {avg_ser2:.6f}\n")
            self.output_capture.output_signal.emit(f"总体平均SER: {avg_ser:.6f}\n")
            
        except Exception as e:
            import traceback
            error_msg = f"汇总结果失败: {str(e)}\n{traceback.format_exc()}\n"
            self.output_capture.output_signal.emit(error_msg)
            QMessageBox.critical(self, "错误", f"汇总结果失败: {str(e)}")
            
    def load_result_file(self):
        """从result文件或目录读取结果"""
        # 让用户选择文件或目录
        choice = QMessageBox.question(
            self, "选择读取方式", 
            "请选择读取方式：\n\n是 - 选择目录（读取目录下所有JSON文件）\n否 - 选择单个文件",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )
        
        if choice == QMessageBox.Cancel:
            return
        elif choice == QMessageBox.Yes:
            # 选择目录
            dir_path = QFileDialog.getExistingDirectory(
                self, "选择结果目录", str(self.project_root)
            )
            if dir_path:
                result_files = list(Path(dir_path).glob("demod_result_*.json"))
                if not result_files:
                    QMessageBox.warning(self, "警告", f"在目录 {dir_path} 中未找到结果文件（demod_result_*.json）")
                    return
                self.output_capture.output_signal.emit(f"找到 {len(result_files)} 个结果文件，开始汇总...\n")
                self.aggregate_results_from_files(result_files)
        else:
            # 选择单个文件
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择结果文件", str(self.project_root),
                "JSON Files (*.json);;All Files (*)"
            )
            if file_path:
                self.parse_result_file(Path(file_path))
            
    def parse_result_file(self, file_path):
        """解析结果文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 提取SER信息
            result1 = data.get('result1', {})
            result2 = data.get('result2', {})
            
            ser1 = result1.get('ser', 0)
            ser2 = result2.get('ser', 0)
            avg_ser = (ser1 + ser2) / 2 if (ser1 is not None and ser2 is not None) else 0
            
            # 更新显示
            self.ser1_label.setText(f"信号1 SER: {ser1:.6f}" if ser1 is not None else "信号1 SER: -")
            self.ser2_label.setText(f"信号2 SER: {ser2:.6f}" if ser2 is not None else "信号2 SER: -")
            self.avg_ser_label.setText(f"平均SER: {avg_ser:.6f}" if avg_ser > 0 else "平均SER: -")
            
            # 保存到历史记录
            from datetime import datetime
            
            # 从配置文件获取所有参数
            config_params = self.extract_config_params()
                
            result_entry = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'data_type': config_params['data_type'],
                'modulation': config_params['modulation'],
                'amp_ratio': config_params['amp_ratio'],
                'snr': config_params['snr'],
                'freq_offset1': config_params['freq_offset1'],
                'freq_offset2': config_params['freq_offset2'],
                'phase1': config_params['phase1'],
                'phase2': config_params['phase2'],
                'delay1': config_params['delay1'],
                'delay2': config_params['delay2'],
                'ser1': ser1,
                'ser2': ser2,
                'avg_ser': avg_ser,
                'file_path': str(file_path),
                'full_data': data
            }
            
            self.results_data.append(result_entry)
            self.save_results_history()
            self.update_results_table()
            
            QMessageBox.information(self, "成功", "结果已加载并保存到历史记录")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"解析结果文件失败: {str(e)}")
            
    def update_results_table(self):
        """更新结果表格"""
        self.results_table.setRowCount(len(self.results_data))
        
        for i, result in enumerate(self.results_data):
            col = 0
            # 时间
            self.results_table.setItem(i, col, QTableWidgetItem(result.get('timestamp', '-')))
            col += 1
            
            # 数据类型
            data_type = result.get('data_type', 'unknown')
            data_type_display = '仿真' if data_type == 'sim' else ('实采' if data_type == 'real' else '未知')
            self.results_table.setItem(i, col, QTableWidgetItem(data_type_display))
            col += 1
            
            # 调制方式
            self.results_table.setItem(i, col, QTableWidgetItem(str(result.get('modulation', '-'))))
            col += 1
            
            # 幅度比
            self.results_table.setItem(i, col, QTableWidgetItem(f"{result.get('amp_ratio', 0):.2f}"))
            col += 1
            
            # 信噪比
            self.results_table.setItem(i, col, QTableWidgetItem(f"{result.get('snr', 0):.1f}"))
            col += 1
            
            # 仿真参数（如果是实采数据，显示"-"）
            if data_type == 'sim':
                freq_offset1 = result.get('freq_offset1')
                self.results_table.setItem(i, col, QTableWidgetItem(
                    f"{freq_offset1:.1f}" if freq_offset1 is not None else "-"
                ))
                col += 1
                
                freq_offset2 = result.get('freq_offset2')
                self.results_table.setItem(i, col, QTableWidgetItem(
                    f"{freq_offset2:.1f}" if freq_offset2 is not None else "-"
                ))
                col += 1
                
                phase1 = result.get('phase1')
                self.results_table.setItem(i, col, QTableWidgetItem(
                    f"{phase1:.3f}" if phase1 is not None else "-"
                ))
                col += 1
                
                phase2 = result.get('phase2')
                self.results_table.setItem(i, col, QTableWidgetItem(
                    f"{phase2:.3f}" if phase2 is not None else "-"
                ))
                col += 1
                
                delay1 = result.get('delay1')
                self.results_table.setItem(i, col, QTableWidgetItem(
                    f"{delay1:.0f}" if delay1 is not None else "-"
                ))
                col += 1
                
                delay2 = result.get('delay2')
                self.results_table.setItem(i, col, QTableWidgetItem(
                    f"{delay2:.0f}" if delay2 is not None else "-"
                ))
                col += 1
            else:
                # 实采数据，显示"-"
                for _ in range(6):
                    self.results_table.setItem(i, col, QTableWidgetItem("-"))
                    col += 1
            
            # SER结果
            ser1 = result.get('ser1')
            self.results_table.setItem(i, col, QTableWidgetItem(
                f"{ser1:.6f}" if ser1 is not None else "-"
            ))
            col += 1
            
            ser2 = result.get('ser2')
            self.results_table.setItem(i, col, QTableWidgetItem(
                f"{ser2:.6f}" if ser2 is not None else "-"
            ))
            col += 1
            
            avg_ser = result.get('avg_ser')
            self.results_table.setItem(i, col, QTableWidgetItem(
                f"{avg_ser:.6f}" if avg_ser is not None else "-"
            ))
            
    def show_result_details(self, index):
        """显示结果详情"""
        row = index.row()
        if row < len(self.results_data):
            result = self.results_data[row]
            dialog = ResultDetailsDialog(result, self)
            dialog.exec_()
            
    def delete_selected_result(self):
        """删除选中的结果"""
        selected_rows = set(item.row() for item in self.results_table.selectedItems())
        if not selected_rows:
            QMessageBox.warning(self, "警告", "请先选择要删除的结果")
            return
            
        reply = QMessageBox.question(
            self, "确认", f"确定要删除 {len(selected_rows)} 条结果吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 从后往前删除，避免索引变化
            for row in sorted(selected_rows, reverse=True):
                if row < len(self.results_data):
                    self.results_data.pop(row)
            self.save_results_history()
            self.update_results_table()
            QMessageBox.information(self, "成功", "结果已删除")
            
    def clear_all_results(self):
        """清空所有结果"""
        reply = QMessageBox.question(
            self, "确认", "确定要清空所有结果吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.results_data = []
            self.save_results_history()
            self.update_results_table()
            QMessageBox.information(self, "成功", "所有结果已清空")
            
    def save_results_history(self):
        """保存结果历史"""
        try:
            self.results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.output_capture.output_signal.emit(f"保存结果历史失败: {str(e)}\n")
            
    def load_results_history(self):
        """加载结果历史"""
        try:
            if self.results_file.exists():
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    self.results_data = json.load(f)
                self.update_results_table()
        except Exception as e:
            self.output_capture.output_signal.emit(f"加载结果历史失败: {str(e)}\n")
            
    def on_output(self, text):
        """处理输出"""
        self.output_capture.output_signal.emit(text)
        
    def stop_task(self):
        """停止任务"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.status_label.setText("正在停止...")


class ResultDetailsDialog(QDialog):
    """结果详情对话框"""
    
    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.result = result
        self.setWindowTitle("结果详情")
        self.setModal(True)
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 基本信息
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setPlainText(json.dumps(self.result.get('full_data', {}), indent=2, ensure_ascii=False))
        layout.addWidget(info_text)
        
        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

