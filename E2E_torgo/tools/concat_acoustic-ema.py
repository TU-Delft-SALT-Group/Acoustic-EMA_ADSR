import argparse
import kaldiio
import numpy as np
from kaldiio import WriteHelper
from tqdm import tqdm



# Helper function to normalize features
def normalize_features(features):
    # Mean-Variance normalization (Standardization)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    normalized_features = (features - mean) / std
    return normalized_features






def main(args):
    # Reading the features
    audio_feats = kaldiio.load_scp_sequential(args.audio_feats_scp)
    ema_feats = kaldiio.load_scp_sequential(args.ema_feats_scp)

    total_lines = sum(1 for _ in open(args.audio_feats_scp))
    
    # Open a writer to save the new concatenated features
    with WriteHelper(f'ark,scp:{args.output_ark},{args.output_scp}') as writer:
        for (audio_key, audio_feat), (ema_key, ema_feat) in tqdm(
                zip(audio_feats, ema_feats), total=total_lines, desc="Concatenating features"):
            assert audio_key == ema_key, f"Mismatch in keys: {audio_key} vs {ema_key}"
                
            # Normalize audio and ema features separately
            audio_feat_normalized = normalize_features(audio_feat)
            ema_feat_normalized = normalize_features(ema_feat)

            if ema_feat_normalized.shape[0] > audio_feat_normalized.shape[0]:
                ema_feat_normalized = ema_feat_normalized[:audio_feat_normalized.shape[0], :]
            elif ema_feat_normalized.shape[0] < audio_feat_normalized.shape[0]:
                pad_width = audio_feat_normalized.shape[0] - ema_feat_normalized.shape[0]
                ema_feat_normalized = np.pad(ema_feat_normalized, ((0, pad_width), (0, 0)), mode='edge')

            # Concatenate along the feature dimension (axis=1)
            concatenated_feat = np.concatenate((audio_feat_normalized, ema_feat_normalized), axis=1)

            # Save to the new ark and feats.scp
            writer(audio_key, concatenated_feat)

    print(f"Concatenated features saved to {args.output_ark} and {args.output_scp}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate audio and EMA features.")
    parser.add_argument("--audio_feats_scp", type=str, required=True, help="Path to the audio features .scp file.")
    parser.add_argument("--ema_feats_scp", type=str, required=True, help="Path to the EMA features .scp file.")
    parser.add_argument("--output_scp", type=str, required=True, help="Path to output .scp file.")
    parser.add_argument("--output_ark", type=str, required=True, help="Path to output .ark file.")

    args = parser.parse_args()
    main(args)

