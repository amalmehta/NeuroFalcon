"""EvalAI runfile: stream the GRU through the FALCON harness.

Local check, from the project root:
    EVAL_DATA_PATH=data python submission/decode.py --evaluation local --phase minival
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # local runs; the image puts neurofalcon on the path

from falcon_challenge.evaluator import FalconEvaluator

from neurofalcon.gru import GRUDecoder, load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation", choices=["local", "remote"], required=True)
    p.add_argument("--model-path", default=str(Path(__file__).resolve().parent / "decoder.pt"))
    p.add_argument("--split", default="m2", choices=["m2"])
    p.add_argument("--phase", default="minival", choices=["minival", "test"])
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args()
    decoder = GRUDecoder(load(args.model_path), batch_size=args.batch_size)
    evaluator = FalconEvaluator(eval_remote=args.evaluation == "remote", split=args.split, dataloader_workers=0)
    print(evaluator.evaluate(decoder, phase=args.phase))


if __name__ == "__main__":
    main()
