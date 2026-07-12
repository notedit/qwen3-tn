# 数据目录说明

## 评测集(冻结,不得改动)

| 文件 | 规模 | 说明 |
|---|---|---|
| `blind_v0.jsonl` | 2,386 | 合成盲测集(独立 seed/prompt,与训练 pipeline 隔离) |
| `blind_chunks.jsonl` | 2,389 | 盲测切块版(压测/流式衔接用) |
| `databaker_eval.jsonl` | 5,023 | 标贝 dev+test 转换(`tn.datagen.databaker`),真实分布回归集;**非商用许可** |
| `polynorm_eval.jsonl` | 361 | Apple PolyNorm-Bench zh-CN 转换(`tn.datagen.polynorm`),27 类跨域;gold 为 Apple 原标注(读法约定与本项目有差异,以 acceptable 口径为主) |

## 训练组件(混合训练按配方 cat)

| 文件 | 规模 | 说明 |
|---|---|---|
| `train_all_v1.jsonl` | 51,906 | v1 合成(= 已删除的 train_v0 + train_v1,见 git 历史) |
| `databaker_train.jsonl` | 19,912 | 标贝 train 转换;**非商用许可,仅内部实验,不得用于生产模型** |
| `train_v2_synth.jsonl` | 20,369 | v2 合成:负槽位 + 负样本 7 类(seed 2) |
| `train_v3_synth.jsonl` | 6,281 | v3 合成:code/quantity 最小对 + 历史年份(seed 3) |
| `train_v4_synth.jsonl` | 7,455 | v4 合成:多币种/分数/单位/比分定向(seed 4,--boost) |

训练混合配方(合并文件不入库,现场 cat):

```bash
# sft_v2 = train_all_v1 + databaker_train + train_v2_synth
# sft_v3 = 上者 + train_v3_synth
# sft_v4(当前最优)= 上者 + train_v4_synth
cat data/train_all_v1.jsonl data/databaker_train.jsonl \
    data/train_v2_synth.jsonl data/train_v3_synth.jsonl \
    data/train_v4_synth.jsonl > /tmp/train_v4_all.jsonl
```

## 质检过程文件

- `review_pool.jsonl` — judge 质检拦下的可疑样本池(待人工裁决)
- `review_100.md` — 人工抽检导出

## 原始数据(不入库)

- 标贝 BMES(`*.char.bmes`,repo 根目录,已 gitignore):https://www.data-baker.com/data/index/TNtts/
- PolyNorm-Bench:https://github.com/apple/ml-speech-polynorm-bench
