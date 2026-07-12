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

## Phase 7:数据 v2(2026-07-11,负监督 + 真实分布)✅ 第一步

- [x] verbalizer `render()`(canonical 渲染)+ SCORE 连字符/RANGE en-dash/3位年份扩覆盖
- [x] 标贝 BMES 转换器 `tn/datagen/databaker.py`;eval 5023 句(`data/databaker_eval.jsonl`,固定 OOD 回归集)/ train 19,912 句(非商用,仅内部实验);覆盖率报告 `reports/databaker_coverage.md`
- [x] 负槽位(正样本内嵌不展开型号)+ 负样本 7 类;v2 合成 20,369 条
- [x] sft_v2(92k 混合):标贝集句准率 67.3%→**96.6%**,过度触发 0.73→**0.046**/句,负样本 100%;合成盲测 99.75% 无回退
- [x] v3(2026-07-11):最小对(code/quantity 成对)+ 历史年份 + SCORE/RANGE 分隔符 → sft_v3;标贝集 96.7%/97.4%(与 v2 持平),真歧义区来回翻转(368路/470级)
- [x] 数字错读分解(v3,标贝集):同 span 差异 = 风格翻转 113(两读均合法)+ 真读错 22(多为标注噪声/FRACTION 超采样范围)+ 漏读 66;**真模型幻觉 ≈0.2% 且线上 postcheck 可拦**(非法读法不在 valid 集合)
- [x] v4(2026-07-12,PolyNorm 缺口定向):多币种词表 19 种 + Unicode/带分数 + 斜杠月日 +
  电学/前置模板单位(mph)+ ± 前缀 + 比分小数/大比分 + 尾零省读变体 + --boost 定向权重;
  PolyNorm 57.3→61.5%(acceptable 62.6→65.9%,数字错读 16.5→7.5%),
  定向桶 Currency .25→.55 / Fractions .65→.90 / Unit .61→.83 / Decimal→1.00;
  databaker 96.6/97.4 持平、盲测 99.79% 新高。**runs/sft_v4 为当前最优**
- [ ] 逐位 vs 数值需要「读法政策」而非更多数据:按词法 cue(路/线/号/级)定 canonical 并在最小对中一致编码;databaker strict 已近约定差异天花板,以 acceptable 为主指标
- [ ] PolyNorm 剩余:Phone 幺/一 canonical 差异(政策)、Version v前缀读"版本"(政策)、
  nbsp 币种代码进下批造数、范围外 15 类按需扩类(数学表达式/坐标/罗马数字)
- [ ] 归因消融(可选):v2 数据不含标贝 train 重训,分离"负监督"与"真实域适应"贡献

## 二期(不在本机):L20/TurboMind 压测、量化、EAGLE3、Stage A/PUA、VoxFlow 集成、影子模式、5k 人工标注集
