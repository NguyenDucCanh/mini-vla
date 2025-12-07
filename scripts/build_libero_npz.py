import os
import argparse
import numpy as np

from datasets import load_dataset
from utils.tokenizer import SimpleTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="physical-intelligence/libero",
        help="HuggingFace dataset ID"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="split name; check HF dataset card for available splits/configs"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/libero_bc.npz",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="max number of transitions to convert (-1 = all)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # 1. Load HF LIBERO dataset
    print(f"Loading HF dataset {args.hf_dataset}, split={args.split}...")
    ds = load_dataset(args.hf_dataset, split=args.split)

    # 2. Inspect one element to confirm keys
    example = ds[0]
    print("Example keys:", list(example.keys()))

    # Common LeRobot-style keys (adapt if needed):
    #   - "observation.image": RGB image
    #   - "observation.state": low-dim state (8,)
    #   - "action":            continuous action (7,)
    #   - "task.language_instruction": language string
    #
    # You can run this script once and print(example) to verify exact key names.

    all_images = []
    all_states = []
    all_actions = []
    all_texts = []

    # 3. Extract fields
    n_total = len(ds) if args.max_samples < 0 else min(len(ds), args.max_samples)
    print(f"Converting {n_total} samples...")

    for i in range(n_total):
        item = ds[i]

        # Adjust these keys if they differ in your dataset version
        img = item["image"]
        state = item["state"]
        action = item["actions"]
        task_idx = item["task_index"]

        instr = f"libero_task_{task_idx}"

        all_images.append(np.array(img, dtype=np.uint8))
        all_states.append(np.array(state, dtype=np.float32))
        all_actions.append(np.array(action, dtype=np.float32))
        all_texts.append(instr)

    # 4. Build tokenizer + text_ids
    tokenizer = SimpleTokenizer(vocab=None)
    tokenizer.build_from_texts(all_texts)

    text_ids_list = [tokenizer.encode(t) for t in all_texts]
    max_len = max(len(seq) for seq in text_ids_list)
    text_ids_padded = np.zeros((len(text_ids_list), max_len), dtype=np.int64)
    for i, seq in enumerate(text_ids_list):
        text_ids_padded[i, :len(seq)] = np.array(seq, dtype=np.int64)

    images = np.stack(all_images, axis=0)   # (N, H, W, 3)
    states = np.stack(all_states, axis=0)   # (N, state_dim=8)
    actions = np.stack(all_actions, axis=0) # (N, action_dim=7)

    print("Final shapes:")
    print("  images:", images.shape)
    print("  states:", states.shape)
    print("  actions:", actions.shape)
    print("  text_ids:", text_ids_padded.shape)

    np.savez_compressed(
        args.output_path,
        images=images,
        states=states,
        actions=actions,
        text_ids=text_ids_padded,
        vocab=tokenizer.vocab,
    )
    print("Saved NPZ dataset to", args.output_path)


if __name__ == "__main__":
    main()
