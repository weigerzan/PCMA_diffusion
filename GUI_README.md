# PCMA扩散模型盲分离任务GUI使用说明

## 简介

这是一个用于PCMA扩散模型盲分离任务的图形用户界面（GUI）应用程序。它提供了完整的任务流程管理，包括数据生成、模型训练、模型推理和解调输出。

## 功能特性

1. **跨平台支持**：可在Linux和Windows环境下运行
2. **实时终端输出**：显示所有任务的实时输出信息
3. **四个独立模块**：
   - 生成数据（支持仿真和实采数据）
   - 训练模型
   - 模型推理
   - 解调输出（包括SER计算和结果管理）
4. **配置管理**：可以配置YAML中的所有参数，包括：
   - 简易配置：数据类型、调制方式、信噪比、幅度比等关键参数
   - 路径配置：数据保存路径、模型保存路径、推理结果保存路径
   - 高级配置：训练参数、推理参数、数据生成参数、系统配置等
5. **结果管理**：支持保存、查看和删除解调结果
6. **GPU管理**：支持在训练和推理时指定使用的GPU设备

## 系统要求

- Python 3.7 或更高版本
- PyQt5 5.15.0 或更高版本
- PyYAML 6.0 或更高版本
- 其他依赖请参考项目主README.md

## 安装依赖

### 在新机器上首次安装

如果您在另一台没有GUI相关库的机器上运行，需要先安装依赖：

#### 方法1：使用requirements文件（推荐）

```bash
# 进入项目目录
cd /path/to/PCMA_diffusion

# 安装GUI依赖
pip install -r requirements_gui.txt

# 如果还需要安装项目其他依赖，请参考主README.md
```

#### 方法2：单独安装

```bash
pip install PyQt5>=5.15.0 PyYAML>=6.0
```

#### 方法3：使用conda环境（推荐用于隔离环境）

```bash
# 创建新的conda环境
conda create -n pcma_gui python=3.8

# 激活环境
conda activate pcma_gui

# 安装GUI依赖
pip install -r requirements_gui.txt

# 安装其他项目依赖（如果需要）
# 参考主README.md中的依赖安装说明
```

### 验证安装

安装完成后，可以验证是否安装成功：

```bash
python -c "import PyQt5.QtWidgets; print('PyQt5安装成功')"
python -c "import yaml; print('PyYAML安装成功')"
```

### 离线安装方法（适用于无法联网的机器）

如果您需要在离线（无法联网）的Windows/Linux机器上安装GUI依赖，可以按照以下步骤：

#### 步骤1：在有网络的机器上下载wheel文件

**重要**：确保下载机器的Python版本与离线机器相同（例如都是Python 3.8或3.9）。

```bash
# 进入项目目录
cd /path/to/PCMA_diffusion

# 创建离线包目录
mkdir offline_packages

# 下载PyQt5和PyYAML的wheel文件及其依赖
# Windows系统（64位）：
pip download PyQt5 PyYAML -d ./offline_packages --platform win_amd64 --only-binary :all:

# Windows系统（32位，较少见）：
# pip download PyQt5 PyYAML -d ./offline_packages --platform win32 --only-binary :all:

# Linux系统（根据您的系统架构选择）：
# 对于x86_64架构：
pip download PyQt5 PyYAML -d ./offline_packages --platform linux_x86_64 --only-binary :all:
# 对于ARM架构：
# pip download PyQt5 PyYAML -d ./offline_packages --platform linux_aarch64 --only-binary :all:

# 如果上述命令失败，可以尝试不指定平台（会下载当前平台的包）：
# 注意：这要求下载机器和离线机器是相同的操作系统和架构
pip download PyQt5 PyYAML -d ./offline_packages
```

**Windows系统特别说明**：
- 如果离线机器是Windows，建议在Windows机器上下载wheel文件
- 确保Python版本匹配（例如都是Python 3.8）
- 如果无法确定架构，可以先在离线机器上运行：`python -c "import platform; print(platform.machine())"` 查看架构

**注意**：
- 确保下载机器的Python版本和架构与离线机器相同
- Windows系统需要下载Windows版本的wheel文件
- Linux系统需要下载Linux版本的wheel文件
- 如果Python版本不同，wheel文件可能不兼容

#### 步骤2：传输文件到离线机器

将整个`offline_packages`目录复制到离线机器上（可以使用U盘、网络共享等方式）。

#### 步骤3：在离线机器上安装

**Windows系统**：
```cmd
REM 进入项目目录
cd C:\path\to\PCMA_diffusion

REM 从本地目录安装（不连接网络）
pip install --no-index --find-links .\offline_packages PyQt5 PyYAML

REM 或者如果offline_packages在其他位置：
pip install --no-index --find-links C:\path\to\offline_packages PyQt5 PyYAML
```

**Linux系统**：
```bash
# 进入项目目录
cd /path/to/PCMA_diffusion

# 从本地目录安装（不连接网络）
pip install --no-index --find-links ./offline_packages PyQt5 PyYAML

# 或者如果offline_packages在其他位置：
pip install --no-index --find-links /path/to/offline_packages PyQt5 PyYAML
```

**如果安装失败**：
- 检查wheel文件是否完整（文件大小是否正常）
- 检查Python版本是否匹配
- 尝试单独安装每个包：`pip install --no-index --find-links ./offline_packages PyQt5`，然后安装PyYAML

#### 步骤4：验证安装

```bash
# 测试PyQt5
python -c "from PyQt5.QtWidgets import QApplication; print('PyQt5安装成功')"

# 测试PyYAML
python -c "import yaml; print('PyYAML安装成功')"
```

#### 离线安装的替代方案

如果上述方法不可行，您也可以：

1. **使用conda离线包**（如果离线机器有conda）：
   ```bash
   # 在有网络的机器上
   conda create -n pcma_gui python=3.8
   conda activate pcma_gui
   conda install pyqt pyyaml
   conda pack -n pcma_gui -o pcma_gui_env.tar.gz
   
   # 在离线机器上解压并使用
   mkdir pcma_gui_env
   tar -xzf pcma_gui_env.tar.gz -C pcma_gui_env
   source pcma_gui_env/bin/activate  # Linux
   # 或
   pcma_gui_env\Scripts\activate  # Windows
   ```

2. **复制已安装的Python环境**：
   - 如果另一台相同系统的机器已经安装了PyQt5和PyYAML
   - 可以直接复制site-packages目录下的相关包到离线机器

### 常见安装问题

1. **Linux系统缺少Qt库**：
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3-pyqt5
   
   # CentOS/RHEL
   sudo yum install python3-qt5
   ```

2. **Windows系统**：
   - 直接使用pip安装即可，PyQt5会自动下载预编译的二进制包
   - 离线安装时，确保下载的是Windows版本的wheel文件

3. **macOS系统**：
   ```bash
   # 如果pip安装失败，可以尝试使用conda
   conda install pyqt
   ```

4. **离线安装时wheel文件不匹配**：
   - 确保下载机器的Python版本与离线机器相同
   - 确保下载的是对应平台的wheel文件（Windows/Linux/macOS）
   - 如果架构不匹配，尝试下载通用版本的wheel文件

## 使用方法

### 在新机器上运行GUI

1. **确保已安装依赖**（见上面的"安装依赖"部分）

2. **确保项目文件完整**：
   - 确保`gui/`目录存在且包含所有文件
   - 确保`configs/base_config.yaml`存在
   - 确保`run_gui.py`存在

3. **启动GUI**：
   ```bash
   # 进入项目根目录
   cd /path/to/PCMA_diffusion
   
   # 启动GUI
   python run_gui.py
   ```

4. **如果遇到导入错误**：
   - 确保在项目根目录下运行
   - 检查Python路径是否正确
   - 确保所有依赖都已安装

### 启动GUI

```bash
# 在项目根目录下运行
python run_gui.py
```

**注意**：必须在项目根目录下运行，因为GUI需要访问项目中的其他模块和配置文件。

### 使用流程

1. **配置管理**
   - 在"配置管理"标签页中，加载或创建配置文件
   - **简易配置**：设置重要参数
     - 数据类型（实采/仿真）
     - 调制方式（QPSK/8PSK/16QAM）
     - 幅度比
     - 信噪比（可以是固定值或范围）
     - 如果是仿真数据，还需设置频偏、相偏、时延差（可以是固定值或范围）
   - **路径配置**：配置各种输出和数据路径
     - 数据保存路径
     - 模型保存路径
     - 推理结果保存路径
     - 原始数据路径（实采数据需要）
   - **高级配置**：设置其他参数
     - 训练参数：训练轮数、批次大小、学习率等
     - 推理参数：采样步数、DDIM参数等
     - 数据生成参数：根据数据类型显示相应参数
     - 系统配置：多线程CPU核心数（同时设置到generate_mixed和split）
   - 保存配置

2. **生成数据**
   - 切换到"1. 生成数据"标签页
   - 选择"生成仿真数据"或"生成实采数据"
   - 等待数据生成完成
   - 数据路径会自动更新到配置文件中

3. **训练模型**
   - 切换到"2. 训练模型"标签页
   - 确认配置文件正确
   - 查看关键参数（只读，从配置文件读取）
   - **GPU设置**：在"CUDA_VISIBLE_DEVICES"输入框中指定GPU序号（如`0`, `1`, `2`），留空则使用默认GPU
   - 点击"开始训练"
   - 训练数据路径会自动从配置文件中读取

4. **模型推理**
   - 切换到"3. 模型推理"标签页
   - 确认配置文件正确
   - 查看关键参数（只读，从配置文件读取）
   - **GPU设置**：在"CUDA_VISIBLE_DEVICES"输入框中指定GPU序号（如`0`, `1`, `2`），留空则使用默认GPU
   - 点击"开始推理"
   - 测试数据路径会自动从配置文件中读取

5. **解调输出**
   - 切换到"4. 解调输出"标签页
   - 方式1：点击"运行解调（从推理结果）"直接运行解调
   - 方式2：点击"从result文件读取结果"加载已有的解调结果
   - 查看SER结果（信号1 SER、信号2 SER、平均SER）
   - 在结果历史记录中查看、管理和删除所有解调结果

## 重要说明

### 数据路径自动管理

- **训练数据路径**：
  - 实采数据：从`train`目录中取得
  - 仿真数据：取前9个分片
  
- **测试数据路径**：
  - 实采数据：从`test`目录中取得
  - 仿真数据：取最后一个切片

这些路径会在数据生成后自动更新到配置文件中。

### 配置文件

GUI使用YAML配置文件来管理所有参数。建议的工作流程：

1. 从`configs/base_config.yaml`创建新的配置文件
2. 在GUI中设置重要参数
3. 保存配置
4. 执行各个步骤

### 结果管理

所有解调结果会自动保存到`demod_test_results/results_history.json`文件中，包括：
- 时间戳
- 调制方式
- 幅度比
- 信噪比
- 信号1 SER
- 信号2 SER
- 平均SER
- 完整的结果数据

可以在结果历史记录表格中查看、删除这些结果。

## 故障排除

1. **如果GUI无法启动**：
   - 检查是否安装了PyQt5：`pip install PyQt5`
   - 检查Python版本（建议3.7+）
   - 确保在项目根目录下运行
   - Linux系统可能需要安装系统Qt库（见"安装依赖"部分）

2. **如果任务执行失败**：
   - 查看右侧终端输出区域，查看详细错误信息
   - 检查配置文件路径是否正确
   - 检查数据文件是否存在
   - 检查GPU设置是否正确（如果使用GPU）

3. **如果配置无法保存**：
   - 确保配置文件路径有效
   - 检查文件权限
   - 确保配置文件目录存在

4. **在新机器上运行时的常见问题**：
   - **导入错误**：确保在项目根目录下运行，且所有依赖已安装
   - **找不到模块**：检查`gui/`目录是否完整
   - **配置文件错误**：确保`configs/base_config.yaml`存在
   - **路径问题**：确保项目路径正确，避免使用相对路径问题

## 快速部署指南（在新机器上）

如果您需要在另一台机器上运行GUI，请按照以下步骤：

### 1. 获取项目代码

```bash
# 如果使用git
git clone <repository_url>
cd PCMA_diffusion

# 或者直接复制整个项目目录
```

### 2. 检查项目结构

确保以下文件和目录存在：
```
PCMA_diffusion/
├── gui/                    # GUI代码目录
│   ├── main_window.py
│   ├── tabs/
│   └── utils/
├── configs/
│   └── base_config.yaml    # 基础配置文件（必需）
├── run_gui.py              # GUI启动脚本
├── requirements_gui.txt    # GUI依赖文件
└── ...                     # 其他项目文件
```

### 3. 安装依赖

#### 情况A：机器可以联网

```bash
# 方法1：使用pip（推荐）
pip install -r requirements_gui.txt

# 方法2：使用conda（推荐用于隔离环境）
conda create -n pcma_gui python=3.8
conda activate pcma_gui
pip install -r requirements_gui.txt
```

#### 情况B：机器无法联网（离线安装）

请参考上面的"离线安装方法"部分，按照以下步骤：

1. **在有网络的机器上下载wheel文件**：
   ```bash
   # Windows系统
   pip download PyQt5 PyYAML -d ./offline_packages --platform win_amd64 --only-binary :all:
   
   # Linux系统
   pip download PyQt5 PyYAML -d ./offline_packages
   ```

2. **将offline_packages目录复制到离线机器**

3. **在离线机器上安装**：
   ```bash
   pip install --no-index --find-links ./offline_packages PyQt5 PyYAML
   ```

### 4. 验证安装

```bash
# 测试PyQt5
python -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 OK')"

# 测试PyYAML
python -c "import yaml; print('PyYAML OK')"
```

### 5. 运行GUI

```bash
# 确保在项目根目录下
cd /path/to/PCMA_diffusion
python run_gui.py
```

### 6. 如果遇到问题

- **导入错误**：确保在项目根目录下运行
- **找不到模块**：检查`gui/`目录是否完整
- **Qt相关错误**：Linux系统可能需要安装系统Qt库（见"安装依赖"部分）
- **配置文件错误**：确保`configs/base_config.yaml`存在
- **离线安装失败**：确保下载的wheel文件与离线机器的Python版本和系统架构匹配

## 注意事项

- 每个步骤可以单独运行，但建议按顺序执行
- 在执行下一步之前，确保上一步已完成
- 配置文件路径会在数据生成后自动更新，但建议在执行前确认
- 长时间运行的任务（如训练）可以通过"停止"按钮中断
- **GPU设置**：在训练和推理标签页中可以指定使用的GPU，通过`CUDA_VISIBLE_DEVICES`环境变量设置
- **num_workers设置**：在高级配置中设置的CPU核心数会同时应用到`generate_mixed.num_workers`和`split.num_workers`

## 开发说明

GUI代码位于`gui/`目录下：
- `main_window.py`：主窗口
- `tabs/`：各个功能标签页
- `utils/`：工具类（输出捕获、工作线程等）

如需扩展功能，可以修改相应的标签页文件。

