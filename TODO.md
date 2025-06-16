# TODO List

## 🔧 代码修复任务

### 半监督训练日志问题
- [ ] **修复数据增强状态显示不完整问题**
  - 问题：训练日志中只显示一个数据增强选项（`Data augmentation: DISABLED`）
  - 应该显示：标注数据集和无标注数据集的独立增强状态
  - 影响文件：`ijmond-camera-ai/bvm_training/trans_bvm_self_supervised/dataloader.py` `ijmond-camera-ai/bvm_training/trans_bvm_self_supervised/train.py`
  - 预期输出格式：
    ```
    Labeled dataset size: XXX
      - Data augmentation: DISABLED (labeled data)
    
    Unlabeled dataset size: XXX  
      - Data augmentation: ENABLED/DISABLED (based on --aug flag)
    ```
  - 优先级：中等
  - 状态：待修复

### 学习率调度器参数不一致问题
- [ ] **修复三个训练模型的学习率优化器参数配置**
  - 问题：命令行参数与实际使用的调度器参数不匹配，导致参数浪费
  - 具体问题：
    ```python
    # 参数定义但未使用
    --decay_rate=0.9, --decay_epoch=20
    # 实际硬编码使用
    step_size=10, gamma=0.5
    ```
  - 影响文件：
    - `ijmond-camera-ai/bvm_training/trans_bvm/train.py` (监督学习)
    - `ijmond-camera-ai/bvm_training/trans_bvm_self_supervised/train.py` (半监督学习)  
    - `ijmond-camera-ai/bvm_training/trans_bvm_self_supervised_domain_adaptation/train.py` (域适应)
  - 修复内容：
    - 统一使用命令行参数 `opt.decay_epoch` 和 `opt.decay_rate`
    - 添加多种调度器选项支持 (step/cosine/exponential)
    - 确保所有参数都得到有效使用
    - 添加学习率变化日志记录
  - 预期修复代码：
    ```python
    # 使用参数化的调度器
    scheduler = lr_scheduler.StepLR(
        optimizer, 
        step_size=opt.decay_epoch,  # 使用命令行参数
        gamma=opt.decay_rate        # 使用命令行参数
    )
    ```
  - 优先级：**高**
  - 状态：待修复

## 📊 数据处理任务

### 视频数据集下载
- [ ] **下载和准备视频数据集**
  - 任务：下载必要的视频数据用于训练和测试
  - 可能包含：监控视频、烟雾检测视频、工业场景视频等
  - 数据格式：视频文件(.mp4, .avi等)或视频帧序列
  - 存储位置：`data/ijmond-camera/`
  - 后续处理：视频帧提取、标注、预处理
  - 优先级：**高**
  - 状态：待执行

## 🏗️ 项目重构任务

### 代码库统一整理
- [ ] **统一所有非训练、测试脚本外的高趋同文件**
  - 任务：整理和统一项目中重复度高的工具文件
  - 包含范围：
    - 数据加载器（`dataloader.py`）
    - 工具函数（`utils.py`）
    - 模型架构（`ResNet_models.py`、`ResNet.py`）
    - 损失函数（各种loss模块）
    - 配置文件和常量定义
  - 目标：减少代码重复，提高维护性
  - 方法：提取公共模块，建立统一的基础库
  - 优先级：低（项目后期执行）
  - 状态：待规划
