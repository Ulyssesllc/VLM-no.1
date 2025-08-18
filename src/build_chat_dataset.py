import os
import csv
import json
import argparse

QUESTION_DEFAULT = (
    "Hãy kiểm tra tính xác thực của sản phẩm này."  # Vietnamese default instruction
)

EXPLANATION_HINTS_FAKE = [
    "logo bị in lệch",
    "màu sắc không chuẩn",
    "chất liệu kém",
    "tem chống hàng giả sai màu",
    "kiểu chữ không đúng",
]
EXPLANATION_HINTS_REAL = [
    "logo sắc nét",
    "màu đồng nhất",
    "tem chuẩn xác",
    "đường may gọn",
    "chất liệu đúng chuẩn",
]


def build_output(label: int):
    import random

    if label == 0:
        reasons = ", ".join(random.sample(EXPLANATION_HINTS_FAKE, 2))
        return f"Đây có khả năng là hàng giả vì {reasons}."
    else:
        reasons = ", ".join(random.sample(EXPLANATION_HINTS_REAL, 2))
        return f"Có dấu hiệu là hàng thật: {reasons}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="datasets Adidas Samba")
    ap.add_argument("--labels_csv", default="labels.csv")
    ap.add_argument("--out_jsonl", default="chat_dataset.jsonl")
    ap.add_argument("--question", default=QUESTION_DEFAULT)
    args = ap.parse_args()

    csv_path = os.path.join(args.data_root, args.labels_csv)
    out_path = os.path.join(args.data_root, args.out_jsonl)

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            image = r["image_path"]
            label = int(r["label"])
            output = build_output(label)
            obj = {
                "id": f"{i:05d}",
                "image": image,
                "instruction": args.question,
                "output": output,
            }
            rows.append(obj)
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} records to {out_path}")


if __name__ == "__main__":
    main()
