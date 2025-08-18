import os
import csv
import argparse

VALID_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"}


def extract_text_from_filename(name: str):
    base = os.path.splitext(name)[0]
    # Remove leading numeric tokens if present
    return base.replace("_", " ").replace("-", " ")


def gather(root: str, sub: str, label: int):
    dir_path = os.path.join(root, sub)
    rows = []
    for fname in os.listdir(dir_path):
        ext = os.path.splitext(fname)[1]
        if ext in VALID_EXT:
            rel_path = f"{sub}/{fname}"
            text = extract_text_from_filename(fname)
            rows.append((rel_path, text, label))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="datasets Adidas Samba")
    ap.add_argument("--real_dir", default="real")
    ap.add_argument("--fake_dir", default="fake")
    ap.add_argument("--out_csv", default="labels.csv")
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    real_rows = gather(args.data_root, args.real_dir, 1)
    fake_rows = gather(args.data_root, args.fake_dir, 0)
    rows = real_rows + fake_rows
    if args.shuffle:
        import random

        random.shuffle(rows)

    out_path = os.path.join(args.data_root, args.out_csv)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "text", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
