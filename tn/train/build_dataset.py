"""jsonl → token 化训练样本:loss 只算 completion(<|sep|> 之后)。"""

import json

from tn.constants import EOS_ID, SEP, TN_PREFIX
from tn.parser import render_output

MAX_LEN = 512


def encode_example(rec: dict, tok) -> dict | None:
    prompt = TN_PREFIX + rec["src"] + SEP
    completion = render_output([tuple(e) for e in rec["edits"]])
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion, add_special_tokens=False)["input_ids"] + [EOS_ID]
    if len(p_ids) + len(c_ids) > MAX_LEN:
        return None
    return {
        "input_ids": p_ids + c_ids,
        "labels": [-100] * len(p_ids) + c_ids,
    }


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class Collator:
    """右 padding:input_ids 用 pad_id,labels 用 -100。"""

    def __init__(self, pad_id: int = EOS_ID):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = maxlen - n
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }
