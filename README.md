<div align="center">

# MiniMind-8GB-Practice

**在一张消费级 8GB 显卡上，从零跑通 64M 小语言模型的完整训练链路**

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-red?logo=pytorch)](https://pytorch.org/)
[![GPU](https://img.shields.io/badge/GPU-RTX%204060%20Laptop%208GB-green)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](./LICENSE)

中文 | 基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 官方项目的轻量化复刻实践

</div>

---

## 这是什么

官方 [MiniMind](https://github.com/jingyaogong/minimind) 是一套优秀的小语言模型全链路开源项目，但其文档默认环境是 **RTX 3090（24GB）**。本项目回答一个现实问题：

> **手上只有一张 RTX 4060 Laptop（8GB）、Windows 系统、没有大显存，还能不能把整条链路亲手跑通？**

答案是能。本项目是一份**面向消费级 8GB 显卡的轻量化复刻实践**：不改模型结构、不降数据规模，通过对训练参数、显存调度和工程环境的系统性适配，在 8GB 显卡上完整复现了官方的全部核心阶段，并留下逐阶段的真实实测数据与踩坑记录。

**本仓库的定位不是 Fork 一个新模型，而是一份"官方项目在受限硬件上的可复现实践手册 + 适配参数"。**

---

## 亮点

- **完整链路**:预训练 → SFT（监督微调）→ LoRA（低秩适配）→ DPO（直接偏好优化）→ GRPO/CISPO（强化学习对齐）→ 网页部署,全部在 8GB 单卡跑通。
- **真实数据**:每个阶段都记录真实 loss（损失）/ reward（奖励）、步数、耗时,不含估算。
- **显存实战**:针对 8GB 显存给出一套"无损优先、有损兜底"的分级优化策略,可直接复用。
- **Windows 排错**:记录页面文件溢出、DLL 冲突、路径与终端差异等 16+ 个真实坑及解法。
- **效果验证**:三轮 A/B 对比实验,用可观察的行为差异量化每个对齐阶段的收益,并诚实指出 64M 模型的能力边界。

---

## 与官方项目的关系

| 维度 | 官方 MiniMind | 本项目（8GB Practice） |
|---|---|---|
| 定位 | 从零实现小模型 + 完整教程 | **官方项目在受限硬件上的轻量化复刻实践** |
| 默认硬件 | RTX 3090 24GB / Linux | **RTX 4060 Laptop 8GB / Windows** |
| 模型结构 | 64M Dense Decoder-Only | 完全一致（未改动） |
| 训练数据 | mini 版约 2.8GB | 完全一致（未缩减） |
| 核心贡献 | 算法与代码 | **8GB 适配参数 + 显存策略 + 实测数据 + 排错记录** |
| 产出 | 模型 + 文档 | 模型 + 可复现实践记录 + A/B 对比证据 |

> 模型架构、训练算法与大部分源码来自官方项目,本仓库在其基础上沉淀"小显存怎么跑通"的工程实践。致谢见文末。

---

## 模型架构（沿用官方,未改动）

- 64M（百万）参数 Dense（稠密）Decoder-Only（仅解码器）Transformer（变换器）
- RMSNorm（均方根归一化）+ Pre-Norm（前置归一化）
- RoPE（旋转位置编码）+ YaRN（长文本外推）
- GQA（分组查询注意力,q8/kv4）,SwiGLU（门控线性单元）激活
- 词表 6400,hidden_size 768,8 层

---

## 实测结果

所有数字来自 RTX 4060 Laptop 8GB 上的真实训练,详见下方"训练记录"。

| 阶段 | 关键参数 | 实测结果 |
|---|---|---|
| 预训练 Pretrain | 127 万条,1 epoch（轮次） | 158,780 步,loss 1.7651,约 4h |
| SFT 监督微调 | mini 指令数据 | 226,430 步,loss 1.7540,约 4h |
| LoRA 低秩适配 | rank=16 | 0.393M（占全量 0.6%）,loss 2.10 → 1.35 |
| DPO 偏好对齐 | chosen/rejected 对 | loss 0.528（理论起点 ln2≈0.693） |
| GRPO/CISPO | num_generations=2, max_gen_len=512 | 19,502 步,reward -3.48 → +2.38,约 6h |

### 对齐效果（A/B 对比）

同一首诗生成任务,GRPO 前后对照:

| 对照项 | SFT（对齐前） | GRPO（对齐后） |
|---|---|---|
| 诗歌完整性 | 结尾"用心去感受"重复逾百次,无法停止 | 正常收尾,结构完整 |
| reward 趋势 | — | -3.48 → +2.38,稳定上升 |
| 回答长度 | 频繁顶满上限 | Avg Len 265,未顶满 |

**诚实结论**:RLHF（人类反馈强化学习）塑造的是**行为格式**（写完整、不啰嗦、带思考链）,而非注入知识——64M 模型在算术推理上依然失败（`max_gen_len` 与思考链格式都能学会,但"1+1 为什么"仍答非所问）。这是模型容量的物理天花板,不是训练缺陷。

---

## 8GB 显存适配策略（核心贡献）

GRPO 阶段需同时驻留 policy（策略）、reference（参考）、reward（奖励,1.8B/3.4GB）三个模型,8GB 显存严重贴边。采用**分级策略**:

```
第 1 级（无损）  关闭 Edge/VSCode 等占用进程 → 释放约 1GB
      ↓ 仍 OOM
第 2 级（无损）  set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
                整理碎片（报错中约 500MB 为碎片）
      ↓ 仍 OOM
第 3 级（有损）  num_generations 4 → 2（组内样本减半）
      但保留 max_gen_len 512 —— 回答完整性优先
```

其他关键适配:
- `--batch_size 1`:三模型共存时的最小可行批量
- `--num_workers 0`:规避 Windows 页面文件溢出（WinError 1455,8 个子进程重复映射 torch DLL）
- 奖励模型 3.4GB 用 HTTP Range 分块断点续传,应对大文件下载中断

---

## 快速开始

### 环境

```bash
conda create -n llp python=3.10
conda activate llp
pip install -r requirements.txt
# PyTorch 请按官网选择 CUDA 12.1 版本安装
```

> 关键版本:transformers 4.57.6 / huggingface-hub 0.34.4（更高版本会把本地模型路径误判为 HF repo id）。

### 数据

从 ModelScope（魔搭）下载 mini 版数据集,放到 `dataset/`。GRPO 另需 `rlaif.jsonl`（约 23.7MB）。

### 训练（均在 trainer/ 目录执行）

```bash
# 预训练
python train_pretrain.py

# SFT
python train_full_sft.py

# LoRA
python train_lora.py

# DPO
python train_dpo.py

# GRPO（8GB 专用参数）
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u train_grpo.py --batch_size 1 --num_generations 2 --max_gen_len 512 --num_workers 0
```

### 验证

```bash
python eval_llm.py --weight grpo --device cuda
python eval_llm.py --weight grpo --device cuda --open_thinking 1  # 查看思考链
```

---

## 目录结构

```
minimind-8gb-practice/
├── model/          # 模型结构（沿用官方）
├── trainer/        # 各阶段训练脚本 + rollout_engine
├── scripts/        # web_demo / 模型转换 / API 服务
├── configs/        # 8GB 专用训练参数（见下）
├── eval_llm.py     # 对话验证脚本
└── requirements.txt
```

> 数据与权重文件体积较大（合计 >7GB）,未纳入本仓库,请按官方渠道下载。

---

## 常见问题

- **Windows 下 `WinError 1455` 页面文件太小?** 训练加 `--num_workers 0`,不要开多进程 DataLoader（数据加载器）。
- **GRPO 报 CUDA out of memory?** 按上文三级策略,先关占用进程 + `expandable_segments`,再降 `num_generations`。
- **`ModuleNotFoundError: trainer.rollout_engine`?** 该文件为官方较新版本引入,已从官方仓库同步放在 `trainer/`。
- **权重"没生成"?** 从 `trainer/` 运行时权重在 `minimind/out/`,不是 `trainer/out/`（脚本内用相对路径 `../out`）。
- **transformers 导入冲突?** 固定到 4.57.6,huggingface-hub 固定到 0.34.4。

---

## 致谢

本项目的模型架构、训练算法与绝大部分源码来自 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)（Apache-2.0）。本仓库的贡献仅在于:**在 RTX 4060 Laptop 8GB / Windows 环境下的轻量化适配参数、显存调度策略、真实实测数据与排错记录**。感谢原作者开源这套优秀的小模型全链路项目。

## License

[Apache-2.0](./LICENSE)（沿用官方项目协议）
