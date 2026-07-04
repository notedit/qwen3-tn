# TN 一期 TODO(本机 8×H200)

> 方案见 `docs/PLAN.md`(v1.1)。Python 用项目 `.venv`(uv,py3.12)。
> 进度标记截至 2026-07-04。

## Phase 0:环境与底座 ✅

- [x] 目录骨架 / `.gitignore` / `.env`(DeepSeek key,不进 git)
- [x] `Qwen/Qwen3-0.6B-Base` 下载;venv:torch 2.12.1+cu130 / transformers 5.13
- [x] tokenizer 实测:`<|fim_prefix|>`(151659)充当 `<|tn|>`,`<|fim_suffix|>`(151661)充当 `<|sep|>`,eos `<|endoftext|>`(151643),写入 `tn/constants.py`
- [x] ~~vLLM + 本地 32B~~ 改用 **DeepSeek API**(`deepseek-v4-flash`,thinking disabled;实测连通)
- [x] 后台基建:`scripts/launch.sh`(setsid+nohup+pid+log,实测断开存活)/ `status.sh`

## Phase 1:格式契约 ✅(spec 已冻结)

- [x] `tn/parser.py`:宽松第一匹配、行内最后一个 `->` 分隔、失败枚举
- [x] `tn/anchor.py`:最短唯一化(±2 字封顶)
- [x] fuzz:10 万次 round-trip + 5 万次损坏鲁棒性,全绿(顺手抓到 strip 破坏边缘空白锚点的真 bug)

## Phase 2:verbalizer 库 ✅

- [x] 12 类全部实现(采样器 + gold 渲染 + `is_valid` 反向校验 + `split_gold_edit`)
- [x] 单测 105 断言 + 采样/校验自洽性(每类 500 次)
- [ ] 迭代遗留:VERSION 多位段 canonical(十一点二十 vs 一一点二零)待错误分析定夺

## Phase 3:LLM 造数据 pipeline ✅(持续迭代)

- [x] `client.py`(tenacity 退避 ≤8 次 / 并发信号量)、`specs.py`(权重/域/prompt)、`run_gen.py`(断点续跑 + 三道校验 + judge 抽检)
- [x] 5k 冒烟 → train_v0 4320 条;盲测集 blind_v0 2386 条(独立 seed + prompt variant)
- [x] 首轮错误分析驱动的改进:NUMBER 语境冲突守卫(postfilter 清掉 0.8% 噪声)、prompt 允许汉字数字混排、meta 增加 nsw 字段
- [x] `data/review_100.md` 人工抽检文件已导出(**待用户过目**)
- [ ] train_v1 5 万条(后台生成中)→ postfilter → 补造被删 id
- [ ] n-gram 去重脚本(目前仅精确去重;负样本有轻度模板化:"今天天气真好"×7)

## Phase 4:训练 ✅(v0 已跑通,v1 待数据)

- [x] `tn/train/sft.py`(HF Trainer,completion-only loss,bf16+sdpa,可 --resume)
- [x] Stage-1 v0:4.3k × 3 epoch,88 秒,loss 0.274
- [ ] Stage-1 v1:v0+v1 合并 ~54k 重训
- [ ] Stage-2 难例强化(rejection sampling)

## Phase 5:评测框架 ✅

- [x] `tn/eval/run_eval.py`:五项指标 + acceptable 变体口径 + class×domain 分桶 + badcase 导出
- [x] v0 首份报告(blind_v0,2396 句):句准率 92.2%(acceptable 92.8%)、parse_fail 0.33%、数字错读 10.7%(acceptable 口径)、负样本零误触、生成均值 12.9 token / p99 46
- [ ] 生成后校验(verbalizer 拦截)作为线上模拟指标接入

## Phase 6:迭代闭环(进行中)

- [ ] v1 模型分桶报告;出口指标:句准率 ≥98.5%、数字错读 ≤0.05%、解析失败 ≤0.1%
- [ ] 难例定向合成(120 电话 vs 数量、比分 vs 时间、百分比 range)
- [ ] 人工抽检 500 句校准
- [ ] (可选)1.7B-Base 对照
- [ ] token/decode 步数分布落盘(供 L20 延迟推算)

## 二期(不在本机):L20/TurboMind 压测、量化、EAGLE3、Stage A/PUA、VoxFlow 集成、影子模式、5k 人工标注集
