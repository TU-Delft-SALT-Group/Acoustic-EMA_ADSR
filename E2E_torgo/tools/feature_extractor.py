import argparse
import torch, torchaudio
from espnet2.asr.frontend.whisper import WhisperFrontend
from espnet2.asr.frontend.s3prl import S3prlFrontend
import kaldiio
import numpy as np




def process_features(audio_path: str, frontend):

    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Ensure sample rate matches
    if sample_rate != 16000:
        raise ValueError(f"Expected sample rate of 16000, but got {sample_rate}")
    
    # Forward pass through WhisperFrontend
    audio_tensor = waveform.to(device)
    input_lengths = torch.tensor([audio_tensor.size(1)], dtype=torch.int32).to(device)  # Lengths of audio samples
    features, _ = frontend.forward(audio_tensor, input_lengths)  # Call forward method with positional arguments 
  
    # Check the shape
    print("Original shape:", features.shape)

    # Reshape if necessary (e.g., if the features are in the shape (1, time, feature_dim))
    if len(features.shape) == 3:
        features = features.squeeze(0)  # Remove the first dimension if it's 1

    return features

def save_features_to_kaldi(features_list, uttids, ark_path, scp_path):
    # Use kaldiio to save the features in Kaldi's ark and scp formats
    with kaldiio.WriteHelper(f'ark,scp:{ark_path},{scp_path}') as writer:
        for features, uttid in zip(features_list, uttids):
            features_np = features.cpu().numpy()
            writer[uttid] = features_np
            print(f'Saved features for {uttid}')






# Main function
def main(args):
    if args.extractor == "whisper":
         
        # Initialize the WhisperFrontend instance
        frontend = WhisperFrontend(whisper_model="large", download_dir="./whisper")
    else:
        # Initialize the S3prlFrontend instance for xlsr_53
        frontend = S3prlFrontend(frontend_conf={"upstream": "xlsr_53"}, multilayer_feature=False, layer=20, download_dir="./hub")

    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    frontend.to(device)

    features = []
    utt_ids = []

    # Process the input wav.scp file
    with open(args.input_file, "r") as f:
        for line in f:
            uttid = line.split(" ")[0]
            audio_path = line.split(" ")[1].strip()
            feature = process_features(audio_path, frontend)
            features.append(feature)
            utt_ids.append(uttid)

    # Save features to Kaldi format
    print(f"Extracted {len(features)} features from {len(utt_ids)} utterances.")
    save_features_to_kaldi(features, utt_ids, args.output_ark, args.output_scp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process features using WhisperFrontend and save in Kaldi format.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input wav.scp file.")
    parser.add_argument("--output_ark", type=str, required=True, help="Path to output .ark file.")
    parser.add_argument("--output_scp", type=str, required=True, help="Path to output .scp file.")
    parser.add_argument(
    "--extractor", 
    type=str, 
    required=True, 
    choices=["whisper", "xlsr53"], 
    help="Feature extractor to use. Options are 'whisper' or 'xlsr53'."
)

    args = parser.parse_args()
    main(args)








