"""评测链路自检：gold 当 pred 必须 1.0；BIO 往返必须无损。

这是整份作业最便宜、也最该先做的一步——如果评测口径本身有 bug，
后面所有方法对比都是在比较「谁更会迁就 bug」。

用法：
    python scripts/selftest.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import get_logger, write_json, REPORTS  # noqa: E402
from src.data import bio_to_entities, entities_to_bio, load_labels, load_splits  # noqa: E402
from src.metrics import evaluate, format_table  # noqa: E402

log = get_logger("selftest")


def main() -> int:
    ok = True
    summary = {}
    for name in ("resume", "cluener", "msra"):
        splits = load_splits(name)
        labels = load_labels(name)

        # 1) gold = pred → 微平均 F1 必须是 1
        r = evaluate(splits["test"], splits["test"], labels)
        f1 = r["overall"]["f1"]
        summary[name] = {"gold_as_pred_f1": f1, "n_test": len(splits["test"]),
                         "n_gold": r["overall"]["n_gold"]}
        log.info("%-8s gold=pred F1=%.4f  (%d docs / %d entities)", name, f1,
                 len(splits["test"]), r["overall"]["n_gold"])
        if abs(f1 - 1.0) > 1e-9:
            log.error("  ✗ 评测口径有问题")
            ok = False

        # 2) BIO 往返：字符标签转实体再转回来，必须与金标逐条相同
        bad = 0
        for row in splits["test"]:
            tags = entities_to_bio(row["text"], row["entities"])
            back = bio_to_entities(tags)
            gold = sorted((e["start"], e["end"], e["label"]) for e in row["entities"])
            if sorted((e["start"], e["end"], e["label"]) for e in back) != gold:
                bad += 1
        log.info("%-8s BIO 往返不一致 %d 条", name, bad)
        if bad:
            ok = False

        # 3) 打乱预测 → F1 必须显著下降（否则说明评测没在看标签）
        rng = random.Random(0)
        shuffled = []
        for row in splits["test"]:
            copy = {"id": row["id"], "text": row["text"],
                    "entities": [dict(e) for e in row["entities"]]}
            for e in copy["entities"]:
                e["label"] = rng.choice(labels)
            shuffled.append(copy)
        r2 = evaluate(splits["test"], shuffled, labels)
        summary[name]["shuffled_label_f1"] = r2["overall"]["f1"]
        log.info("%-8s 标签打乱后 F1=%.4f（应显著低于 1）", name, r2["overall"]["f1"])
        if r2["overall"]["f1"] > 0.5:
            log.error("  ✗ 标签几乎不影响指标，评测有问题")
            ok = False

        if name == "resume":
            hard = [x for x in splits["test"] if x.get("meta", {}).get("hard")]
            log.info("        resume hard 子集 %d 条", len(hard))

    write_json(REPORTS / "selftest.json", {"ok": ok, "summary": summary})
    log.info("汇总:\n%s", format_table(evaluate(load_splits("resume")["test"],
                                                load_splits("resume")["test"])))
    log.info("自检 %s", "通过" if ok else "失败")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
