# 视频下载与标签筛选工具

单一脚本解决方案，支持多种筛选条件。

## 快速使用

```bash
# 下载所有高质量视频（默认）
python download_videos.py

# 只下载高置信度样本（金标准 + 强一致性）
python download_videos.py high_confidence

# 只下载阳性样本（有烟雾）
python download_videos.py positive_only

# 只下载阴性样本（无烟雾）
python download_videos.py negative_only

# 只下载金标准样本（研究人员验证）
python download_videos.py gold_standard
```

## 输出文件

- `data/videos/`: 下载的视频文件
- `data/video_labels.csv`: 简化的标签信息文件（只包含file_name, label_state, label_state_admin三列）

## 筛选效果

| 筛选条件        | 视频数量 | 说明                          |
| --------------- | -------- | ----------------------------- |
| all             | ~878     | 排除劣质视频后的所有视频      |
| high_confidence | ~852     | 高置信度样本（金标准+强一致） |
| gold_standard   | ~97      | 仅金标准样本（研究人员验证）  |
| positive_only   | ~481     | 仅阳性样本（检测到烟雾）      |
| negative_only   | ~371     | 仅阴性样本（未检测到烟雾）    |

## 标签说明

- **47/32**: 金标准（研究人员标注）
- **23/16**: 强一致性（志愿者一致同意）  
- **19/20**: 弱一致性（第三方仲裁）
- **5/4**: 可能性（单一志愿者）
- **3**: 分歧（志愿者不一致）
- **-1**: 无数据
- **-2**: 劣质视频（自动排除）

奇数为阳性（有烟雾），偶数为阴性（无烟雾）。
