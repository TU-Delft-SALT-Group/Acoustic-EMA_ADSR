export KALDI_ROOT=`pwd`/../../..
export LD_LIBRARY_PATH=$KALDI_ROOT/tools/portaudio/lib:$KALDI_ROOT/tools/openfst-1.7.2/lib:$KALDI_ROOT/src/lib:$LD_LIBRARY_PATH
export PATH=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/yuanyuanzhang/ESPnet202402/espnet/tools/miniconda/envs/espnet/bin:$PATH
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/lib64/:$LD_LIBRARY_PATH


