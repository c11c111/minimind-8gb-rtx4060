# 8GB 显卡训练命令速查

所有命令在 `trainer/` 目录下执行。Python 解释器建议用绝对路径（绕开 Windows 环境激活问题）:

```
"D:/Anaconda/envs/llp/python.exe"
```

## 预训练 / SFT / LoRA / DPO

```bash
python train_pretrain.py
python train_full_sft.py
python train_lora.py
python train_dpo.py
```

## GRPO（8GB 专用，已验证可跑通 19,502 步）

```bash
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "D:/Anaconda/envs/llp/python.exe" -u train_grpo.py --batch_size 1 --num_generations 2 --max_gen_len 512 --log_interval 1 --save_interval 100 --num_workers 0
```

参数说明:

| 参数 | 值 | 原因 |
|---|---|---|
| batch_size | 1 | policy/reference/reward 三模型共存，最小可行批量 |
| num_generations | 2 | 从 4 降级，组内样本减半省显存；4 会在 log_softmax 处 OOM |
| max_gen_len | 512 | 保住回答完整性，这是不能牺牲的项 |
| num_workers | 0 | 规避 Windows 页面文件溢出（WinError 1455） |
| expandable_segments | True | 回收约 500MB 显存碎片 |
| -u | — | 强制日志实时刷新，便于定位卡在哪一步 |

## 对话验证

```bash
python eval_llm.py --weight grpo --device cuda
python eval_llm.py --weight grpo --device cuda --open_thinking 1
cd ../scripts && streamlit run web_demo.py   # 网页对话
```

## OOM 分级处理

1. 关闭 Edge / VSCode 等占用显存的桌面程序（释放约 1GB）
2. `set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
3. 仍不够:`num_generations 4 → 2`,但保留 `max_gen_len 512`
