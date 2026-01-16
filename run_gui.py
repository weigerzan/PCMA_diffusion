#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCMA扩散模型盲分离任务GUI启动脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 切换到项目根目录
os.chdir(project_root)

from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

