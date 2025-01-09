import torch, torchaudio
import kaldiio
import numpy as np
from espnet2.tasks.asr import ASRTask
import argparse






def process_features(audio_path: str, asr_model):

    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Ensure sample rate matches
    if sample_rate != 16000:
        raise ValueError(f"Expected sample rate of 16000, but got {sample_rate}")
    
    # Pass the audio tensor to encoder
    audio_tensor = waveform.to(device)
    input_lengths = torch.tensor([audio_tensor.size(1)], dtype=torch.int32).to(device)  # Lengths of audio samples
    batch = {"speech": audio_tensor, "speech_lengths": input_lengths}
    
    with torch.no_grad():
        features, features_olens = asr_model.encode(**batch)


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




# Define the main function
def main(args):
    # Load the corresponding ASR pretrained model
    task = ASRTask
    asr_train_config = args.config
    asr_model_file = args.model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build and prepare the ASR model
    asr_model, asr_train_args = task.build_model_from_file(asr_train_config, asr_model_file, 'cuda')
    asr_model.to(dtype=getattr(torch, 'float32')).eval()
    asr_model.to(device)

    features = []
    utt_ids = []

    # Process the input wav.scp file
    with open(args.input, "r") as f:
        for line in f:
            uttid = line.split(" ")[0]
            audio_path = line.split(" ")[1].strip()
            feature = process_features(audio_path, asr_model)
            features.append(feature)
            utt_ids.append(uttid)

    # Save features to Kaldi format
    print(f"Extracted {len(features)} features from {len(utt_ids)} utterances.")
    save_features_to_kaldi(features, utt_ids, args.output_ark, args.output_scp)


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ASR features and save in Kaldi format.")
    parser.add_argument("--config", type=str, required=True, help="Path to ASR training configuration file.")
    parser.add_argument("--model", type=str, required=True, help="Path to ASR model file.")
    parser.add_argument("--input", type=str, required=True, help="Path to input wav.scp file.")
    parser.add_argument("--output_ark", type=str, required=True, help="Path to output .ark file.")
    parser.add_argument("--output_scp", type=str, required=True, help="Path to output .scp file.")

    args = parser.parse_args()
    main(args)










