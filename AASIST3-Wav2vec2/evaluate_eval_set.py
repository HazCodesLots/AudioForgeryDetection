import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys

# Add AASIST directory to path so imports work at runtime
sys.path.append(os.path.abspath("AASIST"))
from AASIST3_Wav2Vec2 import AASIST3_Wav2Vec2  # type: ignore
from AASIST3 import AudioProcessor           # type: ignore

WINDOW_SECONDS = 4.0
OVERLAP_SECONDS = 2.0


class EvalDataset(Dataset):
    """
    Returns each file's full raw waveform (unpadded, uncropped) so that
    sliding-window inference can cover the whole utterance instead of a
    single deterministic center crop, matching the paper's approach
    (Section 3.1: 4s windows with 2s overlap, predictions averaged per
    utterance) rather than discarding most of longer files.
    """
    def __init__(self, audio_dir):
        self.flac_files = glob.glob(os.path.join(audio_dir, "*.flac"))
        self.processor = AudioProcessor(sample_rate=16000, max_length_seconds=WINDOW_SECONDS)

    def __len__(self):
        return len(self.flac_files)

    def __getitem__(self, idx):
        path = self.flac_files[idx]
        file_id = os.path.splitext(os.path.basename(path))[0]
        waveform, sr = self.processor.load_audio(path)
        return file_id, waveform


def _single_item_collate(batch):
    """Top-level (picklable) collate_fn — required for num_workers>0 on
    Windows, where DataLoader workers use 'spawn' and can't pickle lambdas."""
    return batch[0]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AASIST3-Wav2Vec2 Sliding-Window Evaluation")
    parser.add_argument("--eval_dir", type=str, default=r"M:\Datasets\ASVspoof5\flac_E",
                        help="Path to folder containing evaluation .flac files")
    parser.add_argument("--checkpoint", type=str, default=r"M:\Results\ASVspoof5\AASIST3Wav2Vec2\aasist3_wav2vec2\Checkpoints\AASIST3_Epoch13.pth",
                        help="Path to model checkpoint .pth file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for scores .txt file (auto-derived if not specified)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model = AASIST3_Wav2Vec2(
        wav2vec2_model_name="facebook/wav2vec2-base",
        freeze_feature_extractor=True,
        freeze_transformer_layers=True,
        unfreeze_top_n_transformer=6,
        encoder_out_dim=256,
        num_temporal_nodes=25,
        num_spatial_nodes=25,
        num_branches=4,
    )

    checkpoint_path = args.checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    eval_dir = args.eval_dir
    dataset = EvalDataset(eval_dir)
    print(f"Found {len(dataset)} flac files to evaluate.")

    loader = DataLoader(
        dataset, batch_size=1, num_workers=4, shuffle=False,
        collate_fn=_single_item_collate,
    )

    if args.output:
        output_file = args.output
    else:
        import os, re
        _ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
        _epoch_tag = re.sub(r'AASIST3_', '', _ckpt_stem).lower()
        output_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        output_file = os.path.join(output_dir, f"eval_scores_{_epoch_tag}.txt")

    print(f"Writing scores to {output_file}")
    print(f"Sliding-window inference: {WINDOW_SECONDS}s window, {OVERLAP_SECONDS}s overlap")

    processor = dataset.processor

    with open(output_file, 'w') as f, torch.no_grad():
        for file_id, waveform in tqdm(loader, desc="Evaluating"):
            windows = processor.create_sliding_windows(
                waveform, window_seconds=WINDOW_SECONDS, overlap_seconds=OVERLAP_SECONDS
            )  # (N_windows, max_len)

            # Add channel dim + pad/normalise each window: (N_windows, 1, max_len)
            windows = torch.stack([
                processor.process(w) for w in windows
            ]).to(device)

            # Use BF16 for fast inference
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(windows)                  # (N_windows, 2)
                # Average logits across windows before softmax, matching
                # TrainAASIST3.validate_sliding_window() in AASIST3.py.
                utt_logits = logits.mean(dim=0, keepdim=True)
                probs = F.softmax(utt_logits, dim=1)
                # Class 0 is bonafide, Class 1 is spoof.
                # We output the bonafide probability as the score.
                bonafide_score = probs[0, 0].item()

            f.write(f"{file_id} {bonafide_score:.6f}\n")

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
