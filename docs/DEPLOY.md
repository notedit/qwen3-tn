# TN 推理部署方案与压测报告

> 状态:H200 实测完成(2026-07-04),L20 数字为外推估算,待 L20 实机验证。

## 1. 结论速览

- **引擎选型:vLLM**(实测比 LMDeploy TurboMind/PyTorch 后端快 ~30%;TurboMind 对 0.6B 未吃到 graph 优化,2.05ms/token vs vLLM 1.51)
- **量化:FP8 直接可用**(`--quantization fp8`,一个 flag,H200/L20 都原生支持):精度无回退(盲测 1200 句 99.58% vs bf16 99.62%),bs=1 P50 从 27.1→20.6ms(-24%)
- **H200 实测(FP8,bs=1 全链路):P50 20.6ms / P99 63ms**;P50 已远优于 50ms,P99 超标
- **P99 的瓶颈是 decode 步数,不是算力**:P99 请求 = 数字密集句输出 ~44 token;实测按标点切块(≤40 字)**无法**压低步数(密集句切不散,13.1/49 vs 整句 12.8/44)——P99 达标的正解是 **EAGLE3/MTP 投机解码**(等效步数 ÷1.5-2.5),FP8 与 host 开销优化为辅
- **0.6B bs=1 的延迟本质**:每步 ~1.2ms 是 serving host 开销(调度/detok),GPU 读权重只占 0.25ms(H200);这也是 TurboMind/各种 graph 开关无收益的原因

## 2. 推荐部署配置(L20 单卡)

```bash
vllm serve <model_path> \
  --port 8103 \
  --quantization fp8 \
  --gpu-memory-utilization 0.3 \
  --max-model-len 1024 \
  --max-num-seqs 8 \
  --enable-prefix-caching
```

业务侧(参考实现 `tn/serve/server.py`,或集成进 VoxFlow Python Worker):

- prompt:`<|fim_prefix|>{原文块}<|fim_suffix|>`,greedy(temperature=0),stop 于 `<|endoftext|>`(id 151643),max_tokens 128
- 输出 → `tn/parser.parse_edits` → **生成后校验** `tn/serve/postcheck.filter_edits`(逐 span 拦幻觉,非法 span 回退原文,不整句作废)→ `apply_edits`
- ParseError / 超时 35ms → `fallback=true`,上游走 WFST 全链路
- 监控四件套:解析失败率、postcheck 拦截数、超时 fallback 率、P50/P99(`/metrics`)
- batch:`max_num_seqs 8` 上限即可,TN 短输出下 continuous batching 排队穿透很小(H200 实测并发 8 时 P50 仅 +7ms)

## 3. H200 实测(vLLM,含 HTTP + 解析 + 生成后校验全链路)

单步 decode 速率(256 token 强制生成,bs=1):

| 配置 | ms/token |
|---|---|
| LMDeploy TurboMind bf16 | 2.05 |
| LMDeploy PyTorch 后端 bf16 | 2.20 |
| vLLM bf16 | **1.51** |
| vLLM bf16 + FULL_DECODE_ONLY cudagraph | 1.65(无收益) |
| vLLM bf16 + async-scheduling | 1.64(无收益) |
| vLLM FP8 | **1.36** |

端到端延迟(盲测整句,500 请求,含 HTTP + tokenize + 生成 + 解析 + 生成后校验):

| 配置 | 并发 | P50 | P90 | P99 | 均值 | QPS |
|---|---|---|---|---|---|---|
| bf16 | 1 | 27.1 | 56.1 | 81.3 | 30.0 | 33 |
| bf16 | 4 | 35.6 | 74.4 | 109.2 | 39.0 | 102 |
| bf16 | 8 | 53.6 | 109.9 | 189.7 | 57.9 | 137 |
| **FP8** | **1** | **20.6** | **42.8** | **63.1** | **22.7** | 44 |
| FP8 | 4 | 35.1 | 73.2 | 110.2 | 38.4 | 103 |
| FP8 | 8 | 55.1 | 117.5 | 190.8 | 61.0 | 130 |

要点:

- 输出 token:均值 12.8,P99 44。**延迟 ≈ 固定开销(HTTP+prefill+解析 ≈5ms)+ 步数 × 单步耗时**;FP8 bs=1 反推单步 ≈1.36ms,与探针一致
- **切块(≤40 字)实测不降 P99 步数**(13.1/49 vs 12.8/44):P99 请求是数字密集句(一句 2-3 个 NSW),密度切不散。切块的价值在长输入的 prefill 和流式衔接,不在 decode
- 并发是闭环打满口径(压测语义),线上 L0 低并发 + 35ms 熔断不会出现 c8 的排队尾部;容量规划看 QPS 列
- 无编辑句 = prefill + 1 step(百分位曲线左半段全是这类),P50 由此受益

## 4. L20 延迟估算(待实机验证)

模型:`单步耗时 ≈ host 开销 + 权重字节数 / 显存带宽`

- H200 反推 host 开销:1.51 − 1.2GB/4.8TBps ≈ **1.26ms/步**(FP8 口径 1.23,一致);L20 主机侧按 1.1× 计 ≈ 1.4ms
- L20 带宽 864GB/s:权重读取 bf16 1.2GB → 1.39ms;FP8 0.6GB → 0.69ms

基线(H200 FP8 bs=1 实测):固定开销 ≈5.3ms,单步 1.36ms,P50 步数 ≈11,P99 步数 ≈44。
L20 侧:固定开销按 prefill 放大取 ≈9ms;host 开销按 1.1× 取 ≈1.4ms/步。

| L20 估算(bs=1) | 单步 | P50(11 步) | P99(44 步) |
|---|---|---|---|
| bf16(1.4 + 1.2GB/864GBps) | ≈2.8ms | ≈40ms ⚠ | ≈132ms ❌ |
| FP8(1.4 + 0.6GB/864GBps) | ≈2.1ms | ≈32ms ✅ | ≈101ms ❌ |
| FP8 + EAGLE3(步数 ÷2) | ≈2.1ms | ≈21ms ✅ | ≈55ms ⚠ |
| FP8 + EAGLE3 + 进程内引擎(host −0.8ms)* | ≈1.3ms | ≈16ms ✅ | ≈38ms ✅ |

结论:**L20 上 P50 达标只需 FP8;P99 < 50ms 需要 EAGLE3 + host 开销优化两条都做**。
排序建议:先 FP8(零成本)→ EAGLE3(你有 L20 + Qwen3-1.7B 的 EAGLE3 经验,且改写内容高度可预测,draft 接受率会很好)→ 进程内引擎。

*host 开销压缩手段:vLLM `AsyncLLM` 嵌入 VoxFlow Python Worker(砍掉 api_server 进程一跳)、关闭逐步增量 detokenize;更激进可评估 TensorRT-LLM(小模型 host 开销最低,工程成本高)。

过渡期兜底(EAGLE3 落地前):`max_tokens` 截断 + 截断即 fallback WFST——把尾部延迟换成小比例 fallback 率(cap=32 时 fallback ≈2%,P99 界 ≈9+32×2.1≈76ms;是缓解不是达标)。35ms 超时熔断保底始终在。

**L20 验证清单**(把本仓库拷过去即可复现):

```bash
# 1. 环境:uv venv + vllm(参考 .venv-vllm)
# 2. 单步速率:先看 ms/token 是否落在估算区间(bf16 ~2.5-3.0 / fp8 ~1.9-2.3)
CUDA_VISIBLE_DEVICES=0 vllm serve runs/sft_v1/final --port 8103 --quantization fp8 ...
python -m tn.serve.openai_bench --url http://127.0.0.1:8103 --n 500                     # 整句
python -m tn.serve.openai_bench --url http://127.0.0.1:8103 --data data/blind_chunks.jsonl --n 800  # 切块
# 3. 量化精度回归(必跑,红线:数字类)
python -m tn.serve.endpoint_eval --url http://127.0.0.1:8103 --n 2000
```

## 5. INT8(W8A8)备选路径

FP8 已过精度回归且一个 flag 落地,优先 FP8。若 L20 上想再压权重带宽或对比:

```bash
# llmcompressor 离线 W8A8(SmoothQuant + GPTQ),产物 vLLM 直接加载
uv pip install llmcompressor
# 校准集:data/train_all_v1.jsonl 抽 512 句 src 即可(脚本 ~30 行,需要时我补)
```

注意:任何量化产物上线前必须重跑 `endpoint_eval` 全量盲测,红线是数字类错读不回退(≤ bf16 + 0.1pp)。

## 6. 容量估算

H200 FP8 并发 8 实测 130 QPS/卡(闭环打满口径)。L20 按单步 ≈2.1ms、平均 13 步折算,保守 **60-100 QPS/卡**(EAGLE3 后 ×1.5-2);TTS 前端全流量一张 L20 + 一张热备足够(与方案 v1.1 §5.3 判断一致)。注意延迟型服务不要把卡打满:按 P99 预算控制在 ~40% 利用率以内。

## 7. 原始数据

见 `logs/final_bench.log`(四组矩阵)、`logs/ep_eval_fp8.log`(FP8 精度)、`logs/bench_tm.log`(TurboMind 对照)。
