set -e
:<<!
*****************Instruction*****************
Here you can easily creat a model by selecting
an arbitray backbone model and global method.
You can fine-tune it on your own datasets by
using a pre-trained model.
Modify the following settings as you wish !
*********************************************
!

#***************Backbone model****************
#Our code provides some mainstream architectures:
#alexnet
#vgg family:vgg11, vgg11_bn, vgg13, vgg13_bn,
#           vgg16, vgg16_bn, vgg19_bn, vgg19
#resnet family: resnet18, resnet34, resnet50,
#               resnet101, resnet152
#mpncovresnet: mpncovresnet50, mpncovresnet101
#inceptionv3
#You can also add your own network in src/network
arch=resnet50
#*********************************************

#***************global method****************
#Our code provides some global method at the end
#of network:
#GAvP (global average pooling),
#MPNCOV (matrix power normalized cov pooling),
#BCNN (bilinear pooling)
#CBP (compact bilinear pooling)
#...
#You can also add your own method in src/representation
image_representation=MPNCOV
# short description of method
description=reproduce
#*********************************************

#*******************Dataset*******************
#Choose the dataset folder
benchmark="Dataset Name"
datadir=/path/to/the/data
dataset=$datadir
num_classes=#classes
#*********************************************

#****************Hyper-parameters*************

# Freeze the layers before a certain layer.
freeze_layer=0
# Batch size
batchsize=10
# The number of total epochs for training
epoch=100
# The inital learning rate
# decreased by step method
lr=1.2e-3
lr_method=none
lr_params=100
# log method
# description: lr = logspace(params1, params2, #epoch)

#lr_method=log
#lr_params=-1.1\ -5.0
weight_decay=1e-3
classifier_factor=5
# correlation and log parameters
correlation=0
corr_method=olm
log_method=none
max_iter=100
log_order=8
corr_k=50
#*********************************************
echo "Start finetuning!"

datasets_arg=""
reps_arg=""
corr_arg="$correlation"
corrm_arg="$corr_method"
logm_arg="$log_method"
order_arg="$log_order"
maxiter_arg="$max_iter"
corrk_arg="$corr_k"
series_order=""
ddp=0
nproc=1
devices=""
master_port="29500"
amp=0
micro_bs=""
accum_steps=""
dim_reduction=""
lr_params_arg=""
cov_square=0
cov_pow1p5=0
cov_pow_n=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      datasets_arg="$2"; shift 2;;
    --representation)
      reps_arg="$2"; shift 2;;
    --correlation)
      corr_arg="$2"; shift 2;;
    --corr_method|--corr-method)
      corrm_arg="$2"; shift 2;;
    --log_method|--log-method)
      logm_arg="$2"; shift 2;;
    --order|--log-order)
      order_arg="$2"; shift 2;;
    --max_iter|--max-iter)
      maxiter_arg="$2"; shift 2;;
    --corr_k|--corr-k)
      corrk_arg="$2"; shift 2;;
    --corr_k=*|--corr-k=*)
      corrk_arg="${1#*=}"; shift;;
    --lr)
      lrs_arg="$2"; shift 2;;
    --lr_method|--lr-method)
      lr_method="$2"; shift 2;;
    --series_order|--series-order)
      series_order="$2"; shift 2;;
    lr_params)
      shift
      lr_params_arg=""
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        lr_params_arg+=" $1"
        shift
      done
      lr_params_arg="${lr_params_arg# }";;
    --lr_params|--lr-params)
      shift
      lr_params_arg=""
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        lr_params_arg+=" $1"
        shift
      done
      lr_params_arg="${lr_params_arg# }";;
    --lr_params=*|--lr-params=*)
      lr_params_arg="${1#*=}"
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        lr_params_arg+=" $1"
        shift
      done;;
    --batchsize|--bs)
      batchsize="$2"; shift 2;;
    -bs)
      batchsize="$2"; shift 2;;
    --ddp)
      ddp=1; shift;;
    --nproc)
      nproc="$2"; shift 2;;
    --devices)
      devices="$2"; shift 2;;
    --master_port)
      master_port="$2"; shift 2;;
    --amp)
      amp=1; shift;;
    --accum_steps|--accum-steps)
      accum_steps="$2"; shift 2;;
    --micro_batch_size|--micro-batch-size|--mbs)
      micro_bs="$2"; shift 2;;
    --dim_reduction|--dim-reduction|--dr)
      dim_reduction="$2"; shift 2;;
    --classifier_factor|--classifier-factor)
      classifier_factor="$2"; shift 2;;
    --cov_square|--cov-square)
      cov_square=1; shift;;
    --cov_power_1p5|--cov-power-1p5)
      cov_pow1p5=1; shift;;
    --cov_power_n|--cov-power-n)
      cov_pow_n="$2"; shift 2;;
    --cov_power_n=*|--cov-power-n=*)
      cov_pow_n="${1#*=}"; shift;;
    *)
      shift;;
  esac
done

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(dirname "$script_dir")
main_py="$project_root/main.py"

IFS=',' read -r -a DATASETS <<< "$datasets_arg"
IFS=',' read -r -a REPS <<< "$reps_arg"
IFS=',' read -r -a CORRMS <<< "$corrm_arg"
IFS=',' read -r -a LOGMS <<< "$logm_arg"
IFS=',' read -r -a LRS_RAW <<< "${lrs_arg}"

# sanitize lr list (remove empty tokens due to trailing comma)
LRS=()
if [ -n "$lrs_arg" ]; then
  for x in "${LRS_RAW[@]}"; do
    if [ -n "$x" ]; then
      LRS+=("$x")
    fi
  done
fi

if [ -z "$datasets_arg" ]; then DATASETS=("$benchmark"); fi
if [ -z "$reps_arg" ]; then REPS=("$image_representation"); fi
if [ -z "$corrm_arg" ]; then CORRMS=("$corr_method"); fi
if [ -z "$logm_arg" ]; then LOGMS=("$log_method"); fi
if [ ${#LRS[@]} -eq 0 ]; then LRS=("$lr"); fi

# if lr_params provided while lr_method is none, default to step
if [ -n "$lr_params_arg" ] && [ "$lr_method" = "none" ]; then
  lr_method="step"
fi

# expand lr_params list to space-separated floats
LP_STR=""
if [ -n "$lr_params_arg" ]; then
  LP_STR=$(echo "$lr_params_arg" | tr ',' ' ')
else
  LP_STR="$lr_params"
fi
LP_TAG=""
if [ -n "$LP_STR" ]; then
  # take first token as power/exponent if present
  for token in $LP_STR; do LP_TAG="$token"; break; done
fi

gpu_arg=""
if [ "$ddp" -eq 1 ] && [ "$nproc" -gt 1 ]; then
  gpu_arg=""
else
  if [ -n "$devices" ]; then gpu_arg="--gpu $devices"; fi
fi

dist_args=""
if [ "$ddp" -eq 1 ] && [ "$nproc" -gt 1 ]; then dist_args="--world-size $nproc --dist-url env://"; fi

normalize_rep() {
  case "$1" in
    none) echo "none";;
    mpncov|MPNCOV) echo "MPNCOV";;
    svd_pade|SVD_Pade) echo "SVD_Pade";;
    svd_taylor|SVD_Taylor) echo "SVD_Taylor";;
    svd_topn|SVD_TopN) echo "SVD_TopN";;
    svd_trunc|SVD_Trunc) echo "SVD_Trunc";;
    *) echo "$1";;
  esac
}

resolve_dataset() {
  case "$1" in
    cars|Cars)
      benchmark="datasets-fgvc-cars"
      datadir="/root/StanfordCars-Dataset-main/stanford_cars"
      dataset="$datadir"
      num_classes=196;;
    CUB|cub|CUB_200_2011)
      benchmark="CUB_200_2011"
      datadir="/root/data/CUB_200_2011"
      dataset="$datadir"
      num_classes=200;;
    aircraft|fgvc-aircraft)
      benchmark="fgvc-aircraft"
      datadir="/root/data/fgvc-aircraft-2013b"
      dataset="$datadir"
      num_classes=100;;
    *)
      benchmark="$1"
      dataset="$datadir";;
  esac
}

for d in "${DATASETS[@]}"; do
  resolve_dataset "$d"
  for r in "${REPS[@]}"; do
    image_representation=$(normalize_rep "$r")
    for cm in "${CORRMS[@]}"; do
      corr_method="$cm"
      for lm in "${LOGMS[@]}"; do
        log_method="$lm"
        correlation="$corr_arg"
        max_iter="$maxiter_arg"
        log_order="$order_arg"
        corr_k="$corrk_arg"
        for l in "${LRS[@]}"; do
          lr="$l"
          modeldir=Results/Finetune-$benchmark-$arch-$image_representation-$corr_method-ck$corr_k-$log_method-$lr_method
          if [ "$lr_method" = "cos" ] && [ -n "$LP_TAG" ]; then modeldir="$modeldir-p$LP_TAG"; fi
          if [ "$cov_pow1p5" -eq 1 ]; then modeldir="$modeldir-pow1p5"; fi
          if [ -n "$cov_pow_n" ]; then modeldir="$modeldir-pown$cov_pow_n"; fi
          if [ -n "$series_order" ]; then modeldir="$modeldir-so$series_order"; fi
          modeldir="$modeldir-lr$lr-bs$batchsize"
          if [ ! -d  "Results" ]; then mkdir Results; fi
          if ! compgen -G "$modeldir/*.pth.tar" > /dev/null; then
            if [ ! -d  "$modeldir" ]; then mkdir $modeldir; fi
            cp "$script_dir/finetune.sh" "$modeldir"
            runner="python $main_py $dataset"
            if [ "$ddp" -eq 1 ] && [ "$nproc" -gt 1 ]; then
              runner="torchrun --nproc_per_node $nproc --master_port $master_port $main_py $dataset --world-size $nproc --dist-url env://"
            fi
            extra_flags=""
            if [ "$amp" -eq 1 ]; then extra_flags="$extra_flags --amp"; fi
            if [ -n "$accum_steps" ]; then extra_flags="$extra_flags --accum-steps $accum_steps"; fi
            if [ -n "$micro_bs" ]; then extra_flags="$extra_flags --micro-batch-size $micro_bs"; fi
            if [ -n "$dim_reduction" ]; then extra_flags="$extra_flags --dim-reduction $dim_reduction"; fi
            if [ "$cov_square" -eq 1 ]; then extra_flags="$extra_flags --cov-square"; fi
            if [ "$cov_pow1p5" -eq 1 ]; then extra_flags="$extra_flags --cov-power-1p5"; fi
            if [ -n "$cov_pow_n" ]; then extra_flags="$extra_flags --cov-power-n $cov_pow_n"; fi
            if [ -n "$series_order" ]; then extra_flags="$extra_flags --series-order $series_order"; fi
            $runner \
                   --benchmark $benchmark \
                   --pretrained \
                   -a $arch \
                   -p 100 \
                   --epochs $epoch \
                   --lr $lr \
                   --lr-method $lr_method \
                   --lr-params $LP_STR \
                   -j 8 \
                   -b $batchsize \
                   --num-classes $num_classes \
                   --representation $image_representation \
                   --correlation $correlation \
                   --corr-method $corr_method \
                   --corr-k $corr_k \
                   --log-method $log_method \
                   --max-iter $max_iter \
                   --log-order $log_order \
                   --freezed-layer $freeze_layer \
                   --classifier-factor $classifier_factor \
                   --benchmark $benchmark \
                   --modeldir $modeldir \
                   $extra_flags \
                   $gpu_arg \
                   $dist_args
          else
          checkpointfile=$(ls -rt $modeldir/*.pth.tar | tail -1)
            runner="python $main_py $dataset"
            if [ "$ddp" -eq 1 ] && [ "$nproc" -gt 1 ]; then
              runner="torchrun --nproc_per_node $nproc --master_port $master_port $main_py $dataset --world-size $nproc --dist-url env://"
            fi
            extra_flags=""
            if [ "$amp" -eq 1 ]; then extra_flags="$extra_flags --amp"; fi
            if [ -n "$accum_steps" ]; then extra_flags="$extra_flags --accum-steps $accum_steps"; fi
            if [ -n "$micro_bs" ]; then extra_flags="$extra_flags --micro-batch-size $micro_bs"; fi
            if [ -n "$dim_reduction" ]; then extra_flags="$extra_flags --dim-reduction $dim_reduction"; fi
            if [ "$cov_square" -eq 1 ]; then extra_flags="$extra_flags --cov-square"; fi
            if [ "$cov_pow1p5" -eq 1 ]; then extra_flags="$extra_flags --cov-power-1p5"; fi
            if [ -n "$cov_pow_n" ]; then extra_flags="$extra_flags --cov-power-n $cov_pow_n"; fi
            if [ -n "$series_order" ]; then extra_flags="$extra_flags --series-order $series_order"; fi
            $runner \
                   --benchmark $benchmark \
                   --pretrained \
                   -a $arch \
                   -p 100 \
                   --epochs $epoch \
                   --lr $lr \
                   --lr-method $lr_method \
                   --lr-params $LP_STR \
                   -j 8 \
                   -b $batchsize \
                   --num-classes $num_classes \
                   --representation $image_representation \
                   --correlation $correlation \
                   --corr-method $corr_method \
                   --corr-k $corr_k \
                   --log-method $log_method \
                   --max-iter $max_iter \
                   --log-order $log_order \
                   --freezed-layer $freeze_layer \
                   --modeldir $modeldir \
                   --classifier-factor $classifier_factor \
                   --benchmark $benchmark \
                   --resume $checkpointfile \
                   $extra_flags \
                   $gpu_arg \
                   $dist_args
          fi
        done
      done
    done
  done
done

echo "Done!"
