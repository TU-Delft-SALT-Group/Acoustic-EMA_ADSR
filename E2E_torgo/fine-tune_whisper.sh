#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

istage=2
stop_stage=13
train_set=torgo_train
valid_set=torgo_dev
test_sets="torgo_dys_test torgo_typ_test torgo_mild_test torgo_moderate_test torgo_MorS_test torgo_severe_test"

asr_config=conf/tuning/train_asr_fine-tune_whisper.yaml
inference_config=conf/tuning/decode_asr_whisper_noctc_beam10.yaml
asr_stats_dir=exp/asr_stats
asr_exp=exp/asr_train_fine-tune_whisper
inference_asr_model=valid.acc.ave.pth


lm_config=conf/train_lm_transformer.yaml
use_lm=false
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

./asr.sh \
    --stage ${stage}                                   \
    --stop_stage ${stop_stage}                         \
    --nj 10 \
    --gpu_inference true \
    --inference_nj 1 \
    --lang en \
    --token_type whisper_multilingual \
    --feats_normalize "" \
    --audio_format "wav" \
    --feats_type raw \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --cleaner whisper_en                            \
    --asr_config "${asr_config}"                       \
    --asr_exp "${asr_exp}"                             \
    --asr_stats_dir "${asr_stats_dir}"                 \
    --inference_config "${inference_config}"           \
    --inference_asr_model "${inference_asr_model}"     \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_speech_fold_length 512 \
    --lm_train_text "data/${train_set}/text" "$@"
