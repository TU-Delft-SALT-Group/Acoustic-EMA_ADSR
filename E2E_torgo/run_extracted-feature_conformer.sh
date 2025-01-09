#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=2
stop_stage=13
train_set=torgo_train
valid_set=torgo_dev
test_sets="torgo_dys_test torgo_typ_test torgo_mild_test torgo_moderate_test torgo_MorS_test torgo_severe_test"
asr_config=conf/tuning/train_asr_conformer_extracted.yaml
inference_config=conf/decode_asr.yaml
asr_stats_dir=exp/asr_stats_extracted
asr_exp=exp/train_conformer_extracted
inference_asr_model=valid.acc.ave.pth

nbpe=500
pretrained_model=
ignore_init_mismatch=true

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
    --nbpe ${nbpe}                                     \
    --pretrained_model "${pretrained_model}"           \
    --ignore_init_mismatch ${ignore_init_mismatch}     \
    --gpu_inference true \
    --inference_nj 1 \
    --lang en \
    --token_type bpe \
    --feats_normalize "" \
    --audio_format "wav" \
    --feats_type extracted \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
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
