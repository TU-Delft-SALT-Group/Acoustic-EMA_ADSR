#!/usr/bin/env bash

#based on LF-MMI TORGO Hermann code

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

stage=10
decode_nj=6
train_set=torgo_train
test_sets="torgo_dys_test torgo_typ_test torgo_mild_test torgo_moderate_test torgo_MorS_test torgo_severe_test torgo_test"
dev_set="torgo_dev"
gmm=tri3b
nnet3_affix=

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=f   # affix for the TDNN directory name
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# training options
# training chunk-options
chunk_width=140,100,160
dropout_schedule='0,0@0.20,0.3@0.50,0'
common_egs_dir=
xent_regularize=0.1
frame_subsampling_factor=3

# training options
srand=0
remove_egs=true
reporting_email=

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi


# Problem: We have removed the "train_" prefix of our training set in
# the alignment directory names! Bad!
gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${affix}_sp
train_data_dir=data/${train_set}_sp



for f in $gmm_dir/final.mdl $train_data_dir/feats.scp \
    $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


if [ $stage -le 10 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt data/lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 11 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 6 --cmd "$train_cmd" ${train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 12 ]; then
 
  steps/nnet3/chain/build_tree.sh \
     --frame-subsampling-factor ${frame_subsampling_factor} \
     --context-opts "--context-width=2 --central-position=1" \
     --cmd "$train_cmd" 3500 ${train_data_dir} \
     $lang $ali_dir $tree_dir
fi


# $dir/configs will contain xconfig and config files for the initial
# models.  It's a scratch space used by this script but not by
# scripts called from here.
mkdir -p $dir/configs/
# $dir/init will contain the initial models
mkdir -p $dir/init/

tdnn_opts="l2-regularize=0.03 dropout-proportion=0.0 dropout-per-dim-continuous=true"
tdnnf_opts="l2-regularize=0.03 dropout-proportion=0.0 bypass-scale=0.66"
linear_opts="l2-regularize=0.03 orthonormal-constraint=-1.0"
prefinal_opts="l2-regularize=0.03"
output_opts="l2-regularize=0.015"
learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)

if [ $stage -le 14 ]; then

  num_leaves=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')

  echo "$0: creating top model"
  cat <<EOF > $dir/configs/default.xconfig
  input dim=1280 name=input #if the input is Whisper-FT feature, input dim =1280, if it is Whisper-FT concat EMA feature, input dim = 1283 

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1) affine-transform-file=$dir/configs/lda.mat
  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $tdnn_opts dim=768
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  linear-component name=prefinal-l dim=192 $linear_opts  
  
  
  ## adding the layers for chain branch
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts small-dim=192 big-dim=768
  output-layer name=output include-log-softmax=false dim=$num_leaves $output_opts

  # adding the layers for xent branch
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts small-dim=192 big-dim=768
  output-layer name=output-xent dim=$num_leaves learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/default.xconfig --config-dir $dir/configs/
fi
wait;


if [ $stage -le 16 ]; then
    python3 steps/nnet3/chain/train.py \
      --stage=$train_stage \
      --cmd="$cuda_cmd" \
      --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
      --chain.xent-regularize $xent_regularize \
      --chain.leaky-hmm-coefficient=0.1 \
      --chain.l2-regularize=0.0 \
      --chain.apply-deriv-weights=false \
      --chain.lm-opts="--num-extra-lm-states=2000" \
      --trainer.dropout-schedule $dropout_schedule \
      --trainer.add-option="--optimization.memory-compression-level=2" \
      --trainer.srand=$srand \
      --trainer.max-param-change=2.0 \
      --trainer.num-epochs=10 \
      --trainer.frames-per-iter=3000000 \
      --trainer.optimization.num-jobs-initial=1 \
      --trainer.optimization.num-jobs-final=1 \
      --trainer.optimization.initial-effective-lrate=0.001 \
      --trainer.optimization.final-effective-lrate=0.0001 \
      --trainer.num-chunk-per-minibatch=128,64 \
      --egs.chunk-width=$chunk_width \
      --egs.dir="$common_egs_dir" \
      --egs.opts="--frames-overlap-per-eg 0" \
      --cleanup.remove-egs=$remove_egs \
      --use-gpu=true \
      --reporting.email="$reporting_email" \
      --feat-dir=$train_data_dir \
      --tree-dir=$tree_dir \
      --lat-dir=$lat_dir \
      --dir=$dir || exit 1;
    wait;
fi


if [ $stage -le 23 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).

  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test_tgsmall \
    $tree_dir $tree_dir/graph_tgsmall || exit 1;

fi

if [ $stage -le 24 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  for data in $test_sets; do

       steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --frames-per-chunk $frames_per_chunk \
          --nj 1 --cmd "$decode_cmd"  --num-threads 4 \
          $tree_dir/graph_tgsmall data/${data} ${dir}/decode_tgsmall_${data}

  done
fi

exit 0;
