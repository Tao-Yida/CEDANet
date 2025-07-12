# TODO List

## 🔧 代码修复任务



### 半监督训练日志问题
- [x] **修复数据增强状态显示不完整问题** ✅ **已完成**
  - 问题：训练日志中只显示一个数据增强选项（`Data augmentation: DISABLED`）
  - 应该显示：标注数据集和无标注数据集的独立增强状态
  - 影响文件：`src/bvm_training/trans_bvm_self_supervised/dataloader.py` `src/bvm_training/trans_bvm_self_supervised/train.py`
  - 预期输出格式：
    ```
    Labeled dataset size: XXX
      - Data augmentation: DISABLED (labeled data)
    
    Unlabeled dataset size: XXX  
      - Data augmentation: ENABLED/DISABLED (based on --aug flag)
    ```
  - 优先级：中等
  - 状态：✅ **已修复** (2025-06-25)


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
