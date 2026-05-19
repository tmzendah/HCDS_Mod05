"""Flag and optionally remove poor-quality images based on QC report."""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main(args):
    qc = pd.read_csv(args.qc_report)

    flagged = qc[qc["is_blank"] | qc["is_clipped"] | (qc["status"] != "ok")]
    print(f"Flagging {len(flagged)} of {len(qc)} images")

    flag_dir = Path(args.flag_dir)
    flag_dir.mkdir(parents=True, exist_ok=True)

    log = []
    for _, row in flagged.iterrows():
        src = Path(row["path"])
        if src.exists():
            dst = flag_dir / src.name
            if args.move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
            log.append({"original": str(src), "flagged_copy": str(dst), "reason": row.get("status", "qc_fail")})

    log_df = pd.DataFrame(log)
    log_path = flag_dir / "flagged_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flag poor-quality images")
    parser.add_argument("--qc_report", required=True)
    parser.add_argument("--flag_dir", default="reports/flagged_images")
    parser.add_argument("--move", action="store_true", help="Move instead of copy")
    main(parser.parse_args())
