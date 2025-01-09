# Acoustic-articulatory dysarthric speech recognition 

This is the official implementation of the paper [**End-to-end acoustic-articulatory dysarthric speech recognition leveraging large-scale pretrained acoustic features**](to-be-added).
This paper explores the integration of articulatory features, captured using Electromagnetic Articulography (EMA), with both conventional acoustic features and those extracted from large-scale pretrained models including Whisper
and XLSR-53 as well as the fine-tuned Whisper model. 

We used the TORGO dataset. E2E ASR experiments were conducted using the ESPnet toolkit (version: 202402); hybrid TDNNF experiments were conducted using the Kaldi toolkit (version: 5.5). Reproducing our experiments requires installed ESPnet and Kaldi toolkits.

The codes under the folder E2E_torgo should be put under the egs2/ directory of ESPnet. The codes under the folder hybrid_torgo should be put under the egs/ directory of Kaldi.

## Usage

### E2E Conformer experiments using acoustic features
Train Conformer ASR models from scratch using FBank features or features extracted by the Whisper-large-V2/XLSR-53 models:
```bash
cd ./E2E_torgo

./run_conformer.sh

```

To train the Conformer model using Whisper-FT features, the first step is to fine-tune the Whisper model:
```bash

./fine-tune_whisper.sh

```
Use the fine-tuned model to extract Whisper-FT features, save these features in .ark format, and generate feats.scp:

```bash

python./tools/whisper_FT_feature_extractor.py --config <CONFIG> \
                                              --model <MODEL> \
                                              --input <WAV_SCP> \
                                              --output_ark <OUTPUT_FEATURE> \
                                              --output_scp <OUTPUT_FEATS_SCP>

# Example

python ./tools/whisper_FT_feature_extractor.py --config exp/asr_train_FT_whisper/config.yaml \
                                              --model exp/asr_train_FT_whisper/model.pth \
                                              --input dump/raw/torgo_train_sp/wav.scp \
                                              --output_ark extracted_feats_ark/whisper_FT_feature_test.ark \
                                              --output_scp extracted_feats_scp/whisper_FT_feats_test.scp

```

Use the generated feats.scp (Whisper-FT features) to train the Conformer model:

```bash

./run_extracted-feature_conformer.sh

```
### E2E Conformer experiments using concatenated acoustic-EMA features

Extract large-pretrained features and save them in .ark format:
```bash

python ./tools/feature_extractor.py --input <WAV_SCP> \
                                    --output_ark  <OUTPUT_FEATURE> \
                                    --output_scp <OUTPUT_FEATS_SCP \
                                    --extractor <EXTRACTOR>

# Example
python ./tools/feature_extractor.py --input dump/raw/torgo_train_sp/wav.scp \
                                    --output_ark extracted_feats_ark/whisper_large_feature_torgo_train_sp.ark \
                                    --output_scp extracted_feats_scp/whisper_large_feats_torgo_train_sp.scp \
                                    --extractor whisper         

```
Concat acoustic features with EMA features:
```bash

python ./tools/concat_acoustic-ema.py   --audio_feats_scp <ACOUSTIC_FEATS_SCP> \
                                        --ema_feats_scp <EMA_FEATS_SCP> \
                                        --output_scp <OUTPUT_FEATS_SCP> \
                                        --output_ark <OUTPUT_FEATURE>

# Example
python ./tools/concat_acoustic-ema.py   --audio_feats_scp extracted_feats_scp/whisper_large_feats_torgo_train_sp.scp \
                                        --ema_feats_scp /path/to/ema_feats.scp \
                                        --output_scp /path/to/Whisper_large_ema_feats.scp \
                                        --output_ark /path/to/Whisper_large_ema_feats.ark
```

Use the concatenated feats.scp to train the Conformer model. Note: please modify the feature input size in ../conf/tuning/train_asr_conformer_extracted.yaml accordingly.

```bash

./run_extracted-feature_conformer.sh

```
Whisper-large-V2 0-shot testing:

```bash

./run_whisper_0-shot_test.sh

```

### Hybrid TDNNF experiments
```bash
cd ../hybrid_torgo

./run.sh

```

