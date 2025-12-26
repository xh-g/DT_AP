# KiloGram 使用指南

这份文档旨在帮助你快速上手 KiloGram 仓库，重点介绍数据集格式和模型代码的使用。

## 1. 仓库概览

KiloGram 是一个关于七巧板（Tangrams）的数据集和相关代码库。主要包含以下几个部分：
- `dataset/`: 存放原始数据（JSON格式）和七巧板的 SVG 图片。
- `models/`: 包含模型的预处理、微调（训练）和评估代码。
- `analysis/`: 分析代码。
- `tasks/`: 数据收集任务相关的代码。

## 2. 数据集格式 (`dataset/`)

数据集主要以 JSON 格式存储，核心文件包括：
- `full.json`: 完整数据集，包含 1013 个七巧板，每个至少 10 个标注。
- `dense.json`: 密集标注集，包含 74 个七巧板，每个至少 50 个标注。
- `dense10.json`: 从 `full.json` 中抽取的 74 个七巧板的子集。

### JSON 数据结构

每个 JSON 文件是一个字典，键是七巧板的 ID（例如 `page1-0`），值是该七巧板的详细信息。

```json
{
    "page1-0": {
        "snd": 0.8833,  // 命名一致性分数 (Naming Divergence?)
        "pnd": 0.8885,  // 部分命名一致性分数
        "psa": 4.3555,  // 部分分割一致性分数
        "annotations": [ // 标注列表
            {
                "whole": {
                    "wholeAnnotation": "chair", // 整体描述
                    "timestamp": "..."
                },
                "part": { // 部分描述，键是 SVG 中的多边形 ID
                    "1": "backrest",
                    "2": "base",
                    "3": "base",
                    "4": "backrest",
                    "5": "seat",
                    "6": "base",
                    "7": "backrest"
                },
                "workerId": "...", // 标注者 ID
                "metadata": { // 标注过程的元数据
                    "0": {
                        "final": true, // 是否包含在最终提交中
                        "pieces": [5], // 涉及的拼图块 ID
                        "annotation": "seat", // 当前步骤的标注
                        "timestamp": "..."
                    },
                    // ... 更多步骤
                }
            },
            // ... 更多标注
        ]
    }
}
```

### 图片数据
- `dataset/tangrams-svg/`: 包含所有七巧板的 SVG 格式图片。文件名对应 JSON 中的 ID。

## 3. 模型代码 (`models/`)

模型代码主要分为预处理 (`preprocessing`) 和微调/训练 (`finetuning`)。

### 3.1 微调与训练 (`models/finetuning/`)

这是模型训练的核心目录。支持 CLIP 和 ViLT 模型。

#### 环境准备
1. 进入 `models/finetuning/` 目录。
2. 安装依赖：`pip install -r requirements.txt`。
3. **数据准备**：
   - 需要下载预处理好的数据（链接见原 `README.md` 或 HuggingFace）。
   - 将数据放在 `./data/` 目录下，或者在 `dataloader/data_pathes.py` 和 `evaluate.py` 中配置路径。

#### 运行训练
使用 `main.py` 进行训练。主要参数如下：

| 参数 | 说明 | 示例 |
| :--- | :--- | :--- |
| `--model_type` | 模型类型 | `clip` 或 `vilt` |
| `--dataset_type` | 数据集变体 | `aug_aug` (增强), `whole_black`, `part_color` 等 |
| `--exp_name` | 实验名称 | 用于 Weights & Biases 记录 |
| `--batch_size` | 批次大小 | 默认为 10 |
| `--max_epochs` | 最大轮数 | 默认为 200 |
| `--eval` | 仅评估模式 | 如果设置此标志，则不进行训练 |
| `--load_path` | 加载模型路径 | 用于评估或继续训练 |

**示例命令：**
```bash
# 训练 CLIP 模型
python main.py --model_type clip --dataset_type aug_aug --exp_name my_clip_exp

# 评估模型
python main.py --model_type clip --eval --load_path path/to/saved/model --dataset_type aug_aug
```

### 3.2 预处理 (`models/preprocessing/`)

- `model_inputs/`: 包含生成模型输入的脚本，例如生成黑色/彩色图像、增强数据、生成引用游戏上下文等。
- `process_scans/`: 包含处理原始扫描书籍图像生成 SVG 的代码（通常不需要重新运行，除非你有新的原始扫描图）。

## 4. 其他注意事项

- **重复数据**：在 v0.1.1 版本中，发现并合并了一些重复的七巧板（如 `page5-207` 等），详情请参考 `dataset/README.md`。
- **许可证**：代码使用 MIT 许可证，但七巧板图像来源于书籍扫描，仅供教育和研究使用。
