set -e
arch=resnet50
batchsize=256
epoch=60
lr=0.0794
lr_method=step
lr_params="30 45 60"
weight_decay=1e-4
classifier_factor=1
freeze_layer=0
description=reproduce
benchmark=ILSVRC2012
datadir=${DATADIR:-/datasets}
dataset=$datadir/$benchmark
num_classes=1000
REPRS=""
LOG_METHOD="none"
ORDER="8"
CORRELATION="0"
CORR_METHOD="none"
DDP="0"
NPROC="1"
DEVICES="0"
MASTER_PORT="29500"
PRETRAINED="0"
LR_PARAMS_COLLECT=""
DATASET_OVERRIDDEN="0"
MAX_ITER="3"  # <--- [新增] 默认迭代次数为3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --representation)
      REPRS="$2"; shift 2;;
    --log_method)
      LOG_METHOD="$2"; shift 2;;
    --order)
      ORDER="$2"; shift 2;;
    --corr_method)
      CORR_METHOD="$2"; shift 2;;
    --correlation)
      CORRELATION="$2"; shift 2;;
    --max_iter|--max-iter)   # <--- [新增] 参数解析
      MAX_ITER="$2"; shift 2;;
    --ddp)
      DDP="1"; shift 1;;
    --nproc)
      NPROC="$2"; shift 2;;
    --devices)
      DEVICES="$2"; shift 2;;
    --master_port)
      MASTER_PORT="$2"; shift 2;;
    --pretrained)
      PRETRAINED="1"; shift 1;;
    --no_pretrained)
      PRETRAINED="0"; shift 1;;
    --data-root)
      dataset="$2"; DATASET_OVERRIDDEN="1"; shift 2;;
    -a|--arch)
      arch="$2"; shift 2;;
    -b|--batchsize)
      batchsize="$2"; shift 2;;
    --epochs)
      epoch="$2"; shift 2;;
    --lr)
      lr="$2"; shift 2;;
    --lr-method)
      lr_method="$2"; shift 2;;
    --lr-params)
      shift 1
      LR_PARAMS_COLLECT=""
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        token="$1"
        token="${token//,/ }"
        LR_PARAMS_COLLECT="$LR_PARAMS_COLLECT $token"
        shift 1
      done;;
    --weight-decay)
      weight_decay="$2"; shift 2;;
    --classifier-factor)
      classifier_factor="$2"; shift 2;;
    --freezed-layer)
      freeze_layer="$2"; shift 2;;
    --num-classes)
      num_classes="$2"; shift 2;;
    --benchmark)
      benchmark="$2"; shift 2;;
    *)
      shift 1;;
  esac
done

if [ "$DATASET_OVERRIDDEN" = "0" ]; then
  dataset="$datadir/$benchmark"
fi

mkdir -p Results
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR="$SCRIPT_DIR"

IFS=',' read -ra RP_ARR <<< "$REPRS"
if [ -z "$REPRS" ]; then RP_ARR=("SVD_Pade"); fi
IFS=',' read -ra LOG_ARR <<< "$LOG_METHOD"
if [ -z "$LOG_METHOD" ]; then LOG_ARR=("none"); fi
IFS=',' read -ra CORR_ARR <<< "$CORR_METHOD"
if [ -z "$CORR_METHOD" ]; then CORR_ARR=("none"); fi

echo "Start training!"
for rp in "${RP_ARR[@]}"; do
  rep_name="$rp"
  if [ "$rp" = "svd_pade" ]; then rep_name="SVD_Pade"; else
  if [ "$rp" = "svd_taylor" ]; then rep_name="SVD_Taylor"; else
  if [ "$rp" = "svd_trunc" ]; then rep_name="SVD_Trunc"; else
  if [ "$rp" = "mpncov" ]; then rep_name="MPNCOV"; else rep_name="$rp"; fi; fi; fi; fi

  for lm in "${LOG_ARR[@]}"; do
    for cm in "${CORR_ARR[@]}"; do
      # 注意：目录名并未包含 max_iter，如果需要区分实验，建议加进去
      modeldir="Results/FromScratch-${benchmark}-${arch}-${rep_name}-${lm}-corr${cm}-ord${ORDER}-lr${lr}-bs${batchsize}"
      if [ ! -d "$modeldir" ]; then mkdir -p "$modeldir"; fi
      cp "$SCRIPT_DIR/train_resnet.sh" "$modeldir" 2>/dev/null || true

      # <--- [修改] 在这里加入了 --max-iter "$MAX_ITER"
      CMD_ARGS=("$dataset" "--benchmark" "$benchmark" "-a" "$arch" "-p" "100" "--epochs" "$epoch" "--lr" "$lr" "--lr-method" "$lr_method" "-j" "4" "-b" "$batchsize" "--num-classes" "$num_classes" "--representation" "$rep_name" "--freezed-layer" "$freeze_layer" "--classifier-factor" "$classifier_factor" "--modeldir" "$modeldir" "--weight-decay" "$weight_decay" "--log-method" "$lm" "--log-order" "$ORDER" "--corr-method" "$cm" "--correlation" "$CORRELATION" "--dr" "256" "--max-iter" "$MAX_ITER")

      if [[ -n "$LR_PARAMS_COLLECT" ]]; then
        CMD_ARGS+=("--lr-params" $LR_PARAMS_COLLECT)
      else
        CMD_ARGS+=("--lr-params" "$lr_params")
      fi

      if [ "$PRETRAINED" = "1" ]; then
        CMD_ARGS=("${CMD_ARGS[@]}" "--pretrained")
      fi

      if compgen -G "$modeldir/*.pth.tar" > /dev/null; then
        checkpointfile=$(ls -rt $modeldir/*.pth.tar | tail -1)
        CMD_ARGS+=("--resume" "$checkpointfile")
      fi

      if [ "$DDP" = "1" ]; then
        export CUDA_VISIBLE_DEVICES="$DEVICES"
        torchrun --nproc_per_node "$NPROC" --master_port "$MASTER_PORT" "$ROOT_DIR/main.py" "${CMD_ARGS[@]}"
      else
        python "$ROOT_DIR/main.py" "${CMD_ARGS[@]}"
      fi
    done
  done
done
echo "Done!"