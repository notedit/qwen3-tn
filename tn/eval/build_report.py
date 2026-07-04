"""三模型对比单页 HTML 报告生成器。

用法:
  .venv/bin/python -m tn.eval.build_report \
      --model 零样本Base=runs/base/eval_blind \
      --model "SFT v0 (4.3k)=runs/sft_v0/eval_blind2" \
      --model "SFT v1 (54k)=runs/sft_v1/eval_blind" \
      --badcases runs/sft_v1/eval_blind/badcases.jsonl \
      --out reports/model_compare.html

自包含(无外链),浅/深双主题,数据以 JSON 内嵌、前端 JS 渲染。
"""

import argparse
import json
import os

TEMPLATE = r"""<title>Qwen3-0.6B TN 模型对比报告</title>
<style>
:root{
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --baseline:#c3c2b7;
  --ring:rgba(11,11,11,.10); --good:#006300; --bad:#d03b3b;
  --s1:#2a78d6; --s2:#1baf7a; --s3:#eda100;
}
@media (prefers-color-scheme: dark){:root{
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --baseline:#383835;
  --ring:rgba(255,255,255,.10); --good:#0ca30c; --bad:#e66767;
  --s1:#3987e5; --s2:#199e70; --s3:#c98500;
}}
:root[data-theme="light"]{
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --baseline:#c3c2b7;
  --ring:rgba(11,11,11,.10); --good:#006300; --bad:#d03b3b;
  --s1:#2a78d6; --s2:#1baf7a; --s3:#eda100;
}
:root[data-theme="dark"]{
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --baseline:#383835;
  --ring:rgba(255,255,255,.10); --good:#0ca30c; --bad:#e66767;
  --s1:#3987e5; --s2:#199e70; --s3:#c98500;
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
  font:15px/1.65 system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:40px 24px 80px}
header h1{font-size:26px;line-height:1.3;margin:0 0 6px;text-wrap:balance}
header .sub{color:var(--ink2);margin:0 0 14px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 8px}
.chip{font-size:12.5px;color:var(--ink2);background:var(--surface);
  border:1px solid var(--ring);border-radius:999px;padding:3px 12px}
section{margin-top:36px}
h2{font-size:17px;margin:0 0 4px}
.note{color:var(--muted);font-size:12.5px;margin:0 0 14px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:16px 18px}
.tile .k{font-size:12px;letter-spacing:.05em;color:var(--muted);text-transform:uppercase}
.tile .v{font-size:30px;font-weight:650;margin:2px 0}
.tile .d{font-size:12.5px;color:var(--ink2)}
.tile .d b.up{color:var(--good)} .tile .d b.down{color:var(--bad)}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin:0 0 16px;font-size:13px;color:var(--ink2)}
.legend .sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.grp{margin-bottom:18px}
.grp:last-child{margin-bottom:0}
.grp .glabel{font-size:13.5px;margin-bottom:5px;display:flex;justify-content:space-between;color:var(--ink)}
.grp .glabel .warn{color:var(--bad);font-size:12px}
.bars{display:flex;flex-direction:column;gap:2px}
.brow{display:flex;align-items:center;gap:8px}
.btrack{flex:1;position:relative;height:14px;background:transparent}
.bar{position:absolute;left:0;top:0;height:14px;border-radius:0 4px 4px 0;min-width:2px}
.bval{width:64px;font-size:12px;color:var(--ink2);font-variant-numeric:tabular-nums;text-align:right;flex:none}
.bwho{width:14px;flex:none;font-size:11px;color:var(--muted)}
.axis{border-top:1px solid var(--baseline);margin-top:12px;padding-top:4px;
  display:flex;justify-content:space-between;font-size:11px;color:var(--muted);
  font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th{font-size:12px;color:var(--muted);text-align:right;font-weight:600;
  border-bottom:1px solid var(--baseline);padding:6px 10px;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td{padding:6px 10px;border-bottom:1px solid var(--grid);text-align:right;
  font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
td .flag{color:var(--bad)}
.best{font-weight:650}
details{margin-top:10px}
summary{cursor:pointer;color:var(--ink2);font-size:13.5px}
summary:focus-visible{outline:2px solid var(--s1);outline-offset:2px}
.bc{border-left:2px solid var(--grid);padding:8px 0 8px 14px;margin:10px 0}
.bc .src{margin-bottom:3px}
.bc .row{font-size:13px;color:var(--ink2)}
.bc code{background:var(--page);border:1px solid var(--ring);border-radius:4px;
  padding:1px 5px;font-size:12.5px}
.scroll{overflow-x:auto}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--page);
  font-size:12.5px;padding:5px 10px;border-radius:6px;opacity:0;transition:opacity .12s;
  z-index:9;max-width:320px}
@media (prefers-reduced-motion: reduce){#tip{transition:none}}
footer{margin-top:44px;color:var(--muted);font-size:12.5px;border-top:1px solid var(--grid);padding-top:14px}
footer ul{margin:6px 0 0;padding-left:18px}
</style>
<div class="wrap">
<header>
  <h1>Qwen3-0.6B Span-Edit TN:三模型盲测对比</h1>
  <p class="sub" id="sub"></p>
  <div class="chips" id="chips"></div>
</header>

<section>
  <h2>SFT v1 关键指标</h2>
  <p class="note">括号内为相对 SFT v0 的变化;acceptable 口径 = 读法属 verbalizer 合法变体集合(幺/一、两/二 等)即算对</p>
  <div class="tiles" id="tiles"></div>
</section>

<section>
  <h2>核心质量指标(越高越好)</h2>
  <p class="note" id="legend-note"></p>
  <div class="legend" id="legend1"></div>
  <div class="card" id="chartUp"></div>
</section>

<section>
  <h2>风险指标(越低越好)</h2>
  <p class="note">P0 红线:数字错读;解析失败走 WFST fallback,过度触发伤听感</p>
  <div class="card" id="chartDown"></div>
</section>

<section>
  <h2>按 semiotic class 分桶(句准率,exact 口径)</h2>
  <p class="note">⚠ = SFT v1 桶准确率低于 95% 出口线</p>
  <div class="card" id="chartCls"></div>
</section>

<section>
  <h2>全量指标表</h2>
  <div class="card scroll"><table id="bigtable"></table></div>
</section>

<section>
  <h2>SFT v1 典型 badcase</h2>
  <div class="card" id="bads"></div>
</section>

<footer id="foot"></footer>
</div>
<div id="tip" role="presentation"></div>

<script>
const D = __PAYLOAD__;
const fmtPct = x => (x*100).toFixed(x >= 0.995 || x === 0 ? 1 : 2) + '%';
const fmtPP = x => (x >= 0 ? '+' : '') + (x*100).toFixed(2) + ' pp';
const COLORS = ['var(--s1)','var(--s2)','var(--s3)'];
const tip = document.getElementById('tip');
function hover(el, text){
  el.addEventListener('mousemove', e => {
    tip.textContent = text; tip.style.opacity = 1;
    tip.style.left = Math.min(e.clientX + 14, innerWidth - 330) + 'px';
    tip.style.top = (e.clientY + 16) + 'px';
  });
  el.addEventListener('mouseleave', () => tip.style.opacity = 0);
}

// header
document.getElementById('sub').textContent =
  `盲测集 ${D.meta.blind_n} 句(独立 seed + 独立 prompt 措辞,${D.meta.date});greedy 解码`;
document.getElementById('chips').innerHTML = D.models.map((m,i) =>
  `<span class="chip"><span class="sw" style="background:${COLORS[i]};display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px"></span>${m.label} — ${m.train}</span>`).join('');

// tiles: v1 vs v0
const v1 = D.models[2].r, v0 = D.models[1].r;
const tiles = [
  ['句准率 (acceptable)', fmtPct(v1.sentence_acc_acceptable),
   fmtPP(v1.sentence_acc_acceptable - v0.sentence_acc_acceptable), true],
  ['数字错读率 (acceptable)', fmtPct(v1.digit_misread_acceptable),
   fmtPP(v1.digit_misread_acceptable - v0.digit_misread_acceptable), false],
  ['解析失败率', fmtPct(v1.parse_fail), fmtPP(v1.parse_fail - v0.parse_fail), false],
  ['生成 token 均值 / P99', v1.gen_tokens_mean.toFixed(1) + ' / ' + v1.gen_tokens_p99,
   null, null],
];
document.getElementById('tiles').innerHTML = tiles.map(([k,v,d,upGood]) => {
  let dh = '';
  if (d !== null) {
    const up = d.startsWith('+');
    const good = (up && upGood) || (!up && !upGood);
    dh = `<div class="d">vs v0 <b class="${good?'up':'down'}">${d}</b></div>`;
  }
  return `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div>${dh}</div>`;
}).join('');

// legend
document.getElementById('legend1').innerHTML = D.models.map((m,i) =>
  `<span><span class="sw" style="background:${COLORS[i]}"></span>${m.label}</span>`).join('');

// grouped horizontal bars
function chart(el, metrics, fmt, domainMax){
  const host = document.getElementById(el);
  host.innerHTML = metrics.map(([key, label, warnFn]) => {
    const rows = D.models.map((m,i) => {
      const val = m.r[key];
      const w = Math.max(0.4, val / domainMax * 100);
      return `<div class="brow">
        <span class="bwho" aria-hidden="true"></span>
        <span class="btrack"><span class="bar" data-t="${m.label}:${fmt(val)}"
          style="width:${w}%;background:${COLORS[i]}"></span></span>
        <span class="bval">${fmt(val)}</span></div>`;
    }).join('');
    const warn = warnFn && warnFn(D.models[2].r[key]) ? '<span class="warn">未达出口线</span>' : '';
    return `<div class="grp"><div class="glabel"><span>${label}</span>${warn}</div>
      <div class="bars">${rows}</div></div>`;
  }).join('') + `<div class="axis"><span>0</span><span>${fmt(domainMax)}</span></div>`;
  host.querySelectorAll('.bar').forEach(b => hover(b, b.dataset.t));
}
chart('chartUp', [
  ['sentence_acc', '句准率(exact)'],
  ['sentence_acc_acceptable', '句准率(acceptable)', v => v < 0.985],
  ['span_p', 'Span Precision'],
  ['span_r_acceptable', 'Span Recall(acceptable)'],
], fmtPct, 1);
chart('chartDown', [
  ['digit_misread_acceptable', '数字错读率(acceptable)', v => v > 0.0005],
  ['parse_fail', '解析失败率', v => v > 0.001],
  ['overtrigger_per_sent', '过度触发(编辑/句)'],
], x => (x*100).toFixed(2) + '%', Math.max(...D.models.map(m =>
  Math.max(m.r.digit_misread_acceptable, m.r.parse_fail, m.r.overtrigger_per_sent))) * 1.12);

// class buckets
(function(){
  const host = document.getElementById('chartCls');
  host.innerHTML = D.classes.map(c => {
    const rows = D.models.map((m,i) => {
      const b = m.r.buckets['class:' + c];
      const acc = b ? b.acc : 0;
      return `<div class="brow"><span class="bwho"></span>
        <span class="btrack"><span class="bar" data-t="${m.label}:${fmtPct(acc)} (n=${b?b.n:0})"
          style="width:${Math.max(0.4, acc*100)}%;background:${COLORS[i]}"></span></span>
        <span class="bval">${fmtPct(acc)}</span></div>`;
    }).join('');
    const v1b = D.models[2].r.buckets['class:' + c];
    const warn = v1b && v1b.acc < 0.95 ? ' <span class="warn">⚠</span>' : '';
    return `<div class="grp"><div class="glabel"><span>${c}${warn}
      <span style="color:var(--muted);font-size:12px">n=${v1b?v1b.n:0}</span></span></div>
      <div class="bars">${rows}</div></div>`;
  }).join('') + '<div class="axis"><span>0</span><span>100%</span></div>';
  host.querySelectorAll('.bar').forEach(b => hover(b, b.dataset.t));
})();

// big table
(function(){
  const rows = [
    ['句准率(exact)', 'sentence_acc', fmtPct, 1],
    ['句准率(acceptable)', 'sentence_acc_acceptable', fmtPct, 1],
    ['Span P', 'span_p', fmtPct, 1],
    ['Span R(exact)', 'span_r', fmtPct, 1],
    ['Span R(acceptable)', 'span_r_acceptable', fmtPct, 1],
    ['数字错读(exact)', 'digit_misread', fmtPct, -1],
    ['数字错读(acceptable)', 'digit_misread_acceptable', fmtPct, -1],
    ['解析失败', 'parse_fail', fmtPct, -1],
    ['过度触发/句', 'overtrigger_per_sent', x => x.toFixed(3), -1],
    ['生成 token 均值', 'gen_tokens_mean', x => x.toFixed(1), 0],
    ['生成 token P99', 'gen_tokens_p99', x => x, 0],
  ];
  const head = '<tr><th>指标</th>' + D.models.map(m => `<th>${m.label}</th>`).join('') + '</tr>';
  const body = rows.map(([label, key, fmt, dir]) => {
    const vals = D.models.map(m => m.r[key]);
    const best = dir === 1 ? Math.max(...vals) : dir === -1 ? Math.min(...vals) : null;
    return `<tr><td>${label}</td>` + vals.map(v =>
      `<td class="${best !== null && v === best ? 'best' : ''}">${fmt(v)}</td>`).join('') + '</tr>';
  }).join('');
  document.getElementById('bigtable').innerHTML = head + body;
})();

// badcases
document.getElementById('bads').innerHTML = D.badcases.length === 0
  ? '<p class="note" style="margin:0">无</p>'
  : D.badcases.map(b => `<div class="bc">
      <div class="src">${b.src}</div>
      <div class="row">gold:${b.gold.map(e => `<code>${e[0]}→${e[1]}</code>`).join(' ')}</div>
      <div class="row">pred:${b.pred_raw ? b.pred_raw.split('\n').map(l => `<code>${l}</code>`).join(' ') : '<code>(空)</code>'}${b.err && b.err.startsWith('parse') ? ` <span class="flag">${b.err}</span>` : ''}</div>
    </div>`).join('');

// footer
document.getElementById('foot').innerHTML = `口径说明
<ul>
  <li>盲测集与训练数据完全隔离:独立随机 seed、独立 prompt 措辞;gold 读法全部由程序化 verbalizer 生成,LLM 只产出语境句。</li>
  <li>acceptable 口径:预测读法落在该 NSW 的合法读法集合(如 幺/一、两/二、十/一十)即判对;exact 口径要求与 canonical 完全一致。</li>
  <li>零样本 Base 无法遵循编辑输出协议属预期,其"句准率"几乎全部来自碰巧输出空(无编辑句)或解析失败后的偶合,仅作参照下界。</li>
  <li>一期出口线:句准率 ≥98.5%、数字错读 ≤0.05%、解析失败 ≤0.1%、单桶 ≥95%(合成盲测口径,上线判定需人工测试集)。</li>
</ul>`;
</script>
"""


def load_report(d: str) -> dict:
    with open(os.path.join(d, "report.json"), encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="'标签=评测目录',按 base,v0,v1 顺序传三次")
    ap.add_argument("--train-info", action="append", default=[],
                    help="每个模型的训练数据描述,与 --model 同序")
    ap.add_argument("--badcases", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    models = []
    for i, spec in enumerate(args.model):
        label, d = spec.split("=", 1)
        r = load_report(d)
        train = args.train_info[i] if i < len(args.train_info) else ""
        models.append({"label": label, "train": train, "r": r})

    classes = sorted({k.split(":", 1)[1] for m in models
                      for k in m["r"]["buckets"] if k.startswith("class:") and
                      k != "class:NEG"})
    classes.append("NEG")

    badcases = []
    if args.badcases and os.path.exists(args.badcases):
        seen_err = {}
        for line in open(args.badcases, encoding="utf-8"):
            b = json.loads(line)
            key = b.get("err") or "wrong"
            if seen_err.get(key, 0) < 3 and len(badcases) < 10:
                badcases.append(b)
                seen_err[key] = seen_err.get(key, 0) + 1

    payload = {
        "meta": {"blind_n": models[-1]["r"]["n"], "date": args.date or "2026-07-04"},
        "models": models,
        "classes": classes,
        "badcases": badcases,
    }
    html = TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", args.out, len(html), "bytes")


if __name__ == "__main__":
    main()
