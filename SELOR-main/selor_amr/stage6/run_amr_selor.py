"""Stage6: Pipeline runner for AMR-SELOR.

串行调度 Stage1→Stage5，默认对齐 yelp 示例参数（可通过 CLI 覆盖）。
不包含 AMR 解析功能，假定输入 CSV 已带 AMR/三元组列并已跑过 Stage1。"""
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    # 数据与列配置
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--csv_has_header", type=lambda x: str(x).lower() == "true", default=False)
    p.add_argument("--text_col", type=str, default="1")
    p.add_argument("--label_col", type=str, default="0")
    p.add_argument("--label_offset", type=int, default=1)

    # 路径配置
    p.add_argument("--triples_dir", type=str, default="result/triples")
    p.add_argument("--emb_path", type=str, default="result/embeddings/train_cls.pt")
    p.add_argument("--ce_dir", type=str, default="result/ce_triple")
    p.add_argument("--amr_selor_dir", type=str, default="result/amr_selor")
    p.add_argument("--eval_dir", type=str, default="result/amr_selor_eval")

    # 训练/评测超参（对齐当前基线）
    p.add_argument("--antecedent_len", type=int, default=3)
    p.add_argument("--max_triples", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)

    # 开关
    p.add_argument("--run_stage1", action="store_true")
    p.add_argument("--run_stage2", action="store_true")
    p.add_argument("--run_stage3", action="store_true")
    p.add_argument("--run_stage4", action="store_true")
    p.add_argument("--run_stage5", action="store_true")
    return p.parse_args()


def run_cmd(cmd_list):
    print("\n>>", " ".join(cmd_list))
    subprocess.run(cmd_list, check=True)


def main():
    a = parse_args()
    base = Path(__file__).resolve().parents[2]  # 项目根目录

    # Stage1: 提取三元组（需已有 AMR 列）
    if a.run_stage1:
        cmd = [
            sys.executable,
            str(base / "selor_amr/stage1/extract_triples.py"),
            "--train_csv", a.train_csv,
            "--test_csv", a.test_csv,
            "--csv_has_header", str(a.csv_has_header),
            "--triples_col", "2",  # 默认 AMR 列位置
            "--out_dir", a.triples_dir,
        ]
        run_cmd(cmd)

    # Stage2: 构建三元组池
    if a.run_stage2:
        cmd = [
            sys.executable,
            str(base / "selor_amr/stage2/build_triple_pool.py"),
            "--triples_dir", a.triples_dir,
            "--max_triples", str(a.max_triples),
            "--min_freq", "5",
            "--out_dir", a.triples_dir,
        ]
        run_cmd(cmd)

    # Stage3: 提取 CLS & 预训 CE
    if a.run_stage3:
        cmd_cls = [
            sys.executable,
            str(base / "selor_amr/stage3/extract_cls_embedding.py"),
            "--train_csv", a.train_csv,
            "--csv_has_header", str(a.csv_has_header),
            "--text_col", a.text_col,
            "--batch_size", "32",
            "--gpu", str(a.gpu),
            "--out_path", a.emb_path,
        ]
        run_cmd(cmd_cls)

        cmd_ce = [
            sys.executable,
            str(base / "selor_amr/stage3/pretrain_ce_triple.py"),
            "--train_csv", a.train_csv,
            "--csv_has_header", str(a.csv_has_header),
            "--label_col", a.label_col,
            "--label_offset", str(a.label_offset),
            "--triples_dir", a.triples_dir,
            "--train_embedding", a.emb_path,
            "--antecedent_len", str(a.antecedent_len),
            "--num_samples", "50000",
            "--min_coverage", "2",
            "--batch_size", "64",
            "--epochs", "20",
            "--learning_rate", "1e-4",
            "--weight_decay", "1e-5",
            "--gpu", str(a.gpu),
            "--save_dir", a.ce_dir,
            "--seed", str(a.seed),
            "--valid_split", "0.1",
        ]
        run_cmd(cmd_ce)

    # Stage4: 训练 AMR-SELOR
    if a.run_stage4:
        cmd = [
            sys.executable,
            str(base / "selor_amr/stage4/amr_selor.py"),
            "--train_csv", a.train_csv,
            "--test_csv", a.test_csv,
            "--csv_has_header", str(a.csv_has_header),
            "--text_col", a.text_col,
            "--label_col", a.label_col,
            "--label_offset", str(a.label_offset),
            "--triples_dir", a.triples_dir,
            "--train_embedding", a.emb_path,
            "--ce_path", str(Path(a.ce_dir) / "ce_triple_best.pt"),
            "--antecedent_len", str(a.antecedent_len),
            "--max_triples", str(a.max_triples),
            "--batch_size", str(a.batch_size),
            "--epochs", str(a.epochs),
            "--learning_rate", str(a.lr),
            "--weight_decay", str(a.weight_decay),
            "--gamma", str(a.gamma),
            "--gpu", str(a.gpu),
            "--save_dir", a.amr_selor_dir,
            "--seed", str(a.seed),
        ]
        run_cmd(cmd)

    # Stage5: 评测 + 解释导出
    if a.run_stage5:
        cmd = [
            sys.executable,
            str(base / "selor_amr/stage5/eval_amr_selor.py"),
            "--train_csv", a.train_csv,
            "--test_csv", a.test_csv,
            "--csv_has_header", str(a.csv_has_header),
            "--text_col", a.text_col,
            "--label_col", a.label_col,
            "--label_offset", str(a.label_offset),
            "--triples_dir", a.triples_dir,
            "--train_embedding", a.emb_path,
            "--ce_path", str(Path(a.ce_dir) / "ce_triple_best.pt"),
            "--model_path", str(Path(a.amr_selor_dir) / "amr_selor_best.pt"),
            "--antecedent_len", str(a.antecedent_len),
            "--max_triples", str(a.max_triples),
            "--batch_size", "32",
            "--gpu", str(a.gpu),
            "--save_dir", a.eval_dir,
            "--seed", str(a.seed),
        ]
        run_cmd(cmd)


if __name__ == "__main__":
    main()