#!/usr/bin/env bash
# Reproduces the full baseline-vs-hybrid comparison. Run from src/ml with the
# project venv's python first on PATH. Resumable: a run whose predictions file
# already exists is skipped. ~2.5h on an RTX 3060 6GB (transformers dominate).
#
#   classical ML + ML ensemble   ~4 min   (CPU)
#   DL baselines x3 seeds        ~15 min
#   BERT-base baseline           ~15 min
#   RoBERTa plain + hybrid x3    ~1.7 h
set -euo pipefail
LOG=../../results/logs; PRED=../../results/predictions; mkdir -p "$LOG" "$PRED"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1

run() {  # run <prediction-name> <seed> <command...>
  local name=$1 seed=$2; shift 2
  if [ -f "$PRED/${name}__seed${seed}.npz" ]; then echo "[skip] $name seed $seed"; return; fi
  "$@" 2>&1 | grep -v -E "FutureWarning|warnings.warn|hf_xet|GradScaler" | tee "$LOG/${name}_seed${seed}.log"
}

[ -f "$PRED/ml_ensemble__seed42.npz" ] || python baselines_ml.py 2>&1 | tee "$LOG/baselines_ml.log"

for seed in 42 43 44; do
  run roberta_plain       $seed python train_transformer.py --arch plain  --encoder roberta-base --seed $seed
  run roberta_bilstm_attn $seed python train_transformer.py --arch hybrid --encoder roberta-base --seed $seed
  for arch in textcnn bilstm cnn_bilstm; do
    run $arch $seed python baselines_dl.py --arch $arch --seed $seed
  done
done
run bert_plain 42 python train_transformer.py --arch plain --encoder bert-base-uncased --seed 42

python fusion.py            2>&1 | tee "$LOG/fusion.log"
python compare.py --bootstrap 300 2>&1 | tee "$LOG/compare.log"
