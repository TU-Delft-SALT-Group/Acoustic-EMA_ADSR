#!/usr/bin/env bash


. ./cmd.sh
. ./path.sh

. utils/parse_options.sh
test_sets="torgo_dys_test torgo_typ_test torgo_mild_test torgo_moderate_test torgo_MorS_test torgo_severe_test torgo_test"
train_set=torgo_train
set -euo pipefail


stage=0
cmvn_dir=cmvn

if [ $stage -le 0 ]; then
  # when the "--stage 3" option is used below we skip the G2P steps, and use the
  # lexicon we have already downloaded from openslr.org/11/
  local/prepare_dict.sh --stage 3 --nj 20 --cmd "$train_cmd" \
   data/local/lm data/local/lm data/local/dict_nosp

  utils/prepare_lang.sh data/local/dict_nosp \
   "<UNK>" data/local/lang_tmp_nosp data/lang_nosp
fi

if [ $stage -le 1 ]; then
  local/format_lms.sh --src-dir data/lang_nosp data/local/lm
fi


if [ $stage -le 2 ]; then
  for part in $test_sets ${train_set} ${train_set}_sp torgo_dev; do
    steps/compute_cmvn_stats.sh data/$part exp/make_whisper_large/$part $cmvn_dir
  done
fi


# train a monophone system
if [ $stage -le 3 ]; then
  steps/train_mono.sh --boost-silence 1.25 --nj 6 --cmd "$train_cmd" \
    data/${train_set}_sp data/lang_nosp exp/mono

  steps/align_si.sh --boost-silence 1.25 --nj 6 --cmd "$train_cmd" \
    data/${train_set}_sp data/lang_nosp exp/mono exp/mono_ali_${train_set}_sp
fi


if [ $stage -le 4 ]; then
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/${train_set}_sp data/lang_nosp exp/mono_ali_${train_set}_sp exp/tri1

  steps/align_si.sh --nj 6 --cmd "$train_cmd" \
    data/${train_set}_sp data/lang_nosp exp/tri1 exp/tri1_ali_${train_set}_sp
fi

# train an LDA+MLLT system.
if [ $stage -le 5 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
    data/${train_set}_sp data/lang_nosp exp/tri1_ali_${train_set}_sp exp/tri2b

  # Align utts using the tri2b model
  steps/align_si.sh  --nj 6 --cmd "$train_cmd" --use-graphs true \
    data/${train_set}_sp data/lang_nosp exp/tri2b exp/tri2b_ali_${train_set}_sp
fi


# Train tri3b, which is LDA+MLLT+SAT
if [ $stage -le 6 ]; then
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
    data/${train_set}_sp data/lang_nosp exp/tri2b_ali_${train_set}_sp exp/tri3b
fi


# Now we compute the pronunciation and silence probabilities from training data,
# and re-create the lang directory.
if [ $stage -le 7 ]; then
  steps/get_prons.sh --cmd "$train_cmd" \
    data/${train_set}_sp data/lang_nosp exp/tri3b
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp \
    exp/tri3b/pron_counts_nowb.txt exp/tri3b/sil_counts_nowb.txt \
    exp/tri3b/pron_bigram_counts_nowb.txt data/local/dict

# prepare the librispeech LM
  utils/prepare_lang.sh data/local/dict \
    "<UNK>" data/local/lang_tmp data/lang

  local/format_lms.sh --src-dir data/lang data/local/lm


fi



if [ $stage -le 8 ]; then
  steps/align_fmllr.sh --nj 6 --cmd "$train_cmd" \
    data/${train_set}_sp data/lang exp/tri3b exp/tri3b_ali_${train_set}_sp
fi



if [ $stage -le 9 ]; then
  run_tdnnf.sh --stage 10
fi
