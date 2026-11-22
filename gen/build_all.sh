#!/bin/sh
# 骨を N 並列で作って bones_data.js にまとめる。  sh gen/build_all.sh [N]
cd "$(dirname "$0")/.."
BL="${BLENDER:-/c/Program Files/Blender Foundation/Blender 4.5/blender.exe}"
N="${1:-10}"
i=0
while [ $i -lt $N ]; do
  "$BL" --background --python gen/build.py -- --shard $i/$N > gen/_shard_$i.log 2>&1 &
  i=$((i+1))
done
wait
"$BL" --background --python gen/build.py -- --merge 2>&1 | grep "骨 "
PYTHONIOENCODING=utf-8 python gen/realfit.py | head -3   # 実物スキャンに合わせて組み直す（最終の bones_data.js）
