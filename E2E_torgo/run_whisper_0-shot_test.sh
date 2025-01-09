#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

whisper_tag=large    # whisper model tag, e.g., small, medium, large, etc
cleaner=whisper_en
hyp_cleaner=whisper_en
nj=2
test_sets="torgo_dys_test torgo_typ_test torgo_mild_test torgo_moderate_test torgo_MorS_test torgo_severe_test"
whisper_dir="./whisper_model"
decode_options="{language: en, task: transcribe, temperature: 0, beam_size: 10, fp16: False}"

for x in ${test_sets}; do
    wavscp=dump/raw/${x}/wav.scp    # path to wav.scp
    outdir=whisper-${whisper_tag}_outputs_torgo_test_sets/${x}  # path to save output
    gt_text=dump/raw/${x}/text      # path to groundtruth text file (for scoring only)

    scripts/utils/evaluate_asr.sh \
        --whisper_tag ${whisper_tag} \
        --nj ${nj} \
        --gpu_inference true \
        --stage 2 \
        --stop_stage 3 \
        --cleaner ${cleaner} \
        --hyp_cleaner ${hyp_cleaner} \
        --decode_options "${decode_options}" \
        --whisper_dir ${whisper_dir} \
        --gt_text ${gt_text} \
        ${wavscp} \
        ${outdir}
done


