# qwen3-tn

Qwen3-0.6B Span-Edit TN(TTS 文本正则化)。方案见 `docs/PLAN.md`,任务清单 `TODO.md`。

模型发布:https://huggingface.co/leeoxiang/qwen3-0.6b-zh-tn(sft_v6,私有,CC-BY-NC-4.0,
调用/部署说明见其模型卡;训练数据含标贝非商用成分,商用需换数重训)。

## 环境

- `uv venv .venv --python 3.12`,依赖装在 `.venv`(见 `scripts/setup_env.sh`)
- DeepSeek API key 在 `.env`(不进 git),`scripts/launch.sh` 会自动注入

## 目录

```
tn/parser.py        输出解析契约(冻结):parse_and_apply / render_output
tn/anchor.py        数据侧锚点编译:build_edit_script
tn/verbalizer/      gold 读法唯一来源:12 类采样/渲染/反向校验
tn/datagen/         DeepSeek 造数:client(重试)/ specs(prompt)/ run_gen(断点续跑)
tn/train/           SFT:build_dataset(loss mask)/ sft.py
tn/eval/run_eval.py 评测:句准率/Span P·R/数字错读率/解析失败率/分桶/badcase
```

## 常用命令(长任务一律 launch.sh 后台,断开终端不影响)

```bash
# 造数(断点续跑:重跑同一命令自动跳过已完成)
bash scripts/launch.sh datagen -- .venv/bin/python -m tn.datagen.run_gen \
    --out data/train_v0.jsonl --n 5000 --seed 1 --concurrency 12

# 训练(中断续训加 --resume)
bash scripts/launch.sh sft_v0 -- env CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/python -m tn.train.sft --data data/train_v0.jsonl --out runs/sft_v0

# 评测
bash scripts/launch.sh eval_v0 -- env CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/python -m tn.eval.run_eval --model runs/sft_v0/final \
    --data data/blind_v0.jsonl --out runs/sft_v0/eval_blind

# 状态 / 测试
bash scripts/status.sh
.venv/bin/python -m pytest tests/ -q
```

## 数据格式

```jsonl
{"src": "会议定在18号下午3:30。",
 "edits": [["18号","十八号"],["3:30","三点三十分"]],
 "meta": {"classes": ["DATE","TIME"], "ctxs": ["date","time"],
          "domain": "新闻报道", "kind": "positive", "source": "synth-dsv4"}}
```

模型 I/O:`<|fim_prefix|>{src}<|fim_suffix|>` → 每行 `锚点->读法`,无编辑输出空(直接 eos)。
