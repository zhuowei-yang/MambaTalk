#!/bin/bash
set -e

BASE="/data1/yangzhuowei/MambaTalk_new"
DATA_NEW="$BASE/data_new"
DATA_REF="$BASE/data"
MFA_SRC="/data1/yangzhuowei/MambaTalk/data_old/mfa_input"
CONDA_ENV="mambatalk"

echo "============================================"
echo "  MambaTalk data_new 训练数据构建脚本"
echo "============================================"

# ---- 1. 创建目录结构 ----
echo ""
echo "[1/6] 创建目录结构..."
mkdir -p "$DATA_NEW/smplxflame_30"
mkdir -p "$DATA_NEW/textgrid"
mkdir -p "$DATA_NEW/weights"

# ---- 2. 复制 npz → smplxflame_30/ ----
echo "[2/6] 复制 npz 文件到 smplxflame_30/..."
cp "$DATA_NEW/train/"*.npz "$DATA_NEW/smplxflame_30/"
cp "$DATA_NEW/test/"*.npz "$DATA_NEW/smplxflame_30/"
echo "  smplxflame_30: $(ls "$DATA_NEW/smplxflame_30/"*.npz | wc -l) files"

# ---- 3. 验证 wave16k/ ----
echo "[3/6] 验证 wave16k/..."
WAV_COUNT=$(ls "$DATA_NEW/wave16k/"*.wav 2>/dev/null | wc -l)
NPZ_COUNT=$(ls "$DATA_NEW/smplxflame_30/"*.npz | wc -l)
echo "  wave16k: $WAV_COUNT files, smplxflame_30: $NPZ_COUNT files"
if [ "$WAV_COUNT" -lt "$NPZ_COUNT" ]; then
    echo "  wave16k 数量不足，从 train/test 补充..."
    cp "$DATA_NEW/train/"*.wav "$DATA_NEW/wave16k/" 2>/dev/null || true
    cp "$DATA_NEW/test/"*.wav "$DATA_NEW/wave16k/" 2>/dev/null || true
    echo "  补充后 wave16k: $(ls "$DATA_NEW/wave16k/"*.wav | wc -l) files"
fi

# ---- 4. 准备并运行 MFA 生成 TextGrid ----
echo "[4/6] 准备 MFA 输入并生成 TextGrid..."
MFA_WORK="$DATA_NEW/mfa_work"
MFA_OUT="$DATA_NEW/mfa_output"
rm -rf "$MFA_WORK" "$MFA_OUT"
mkdir -p "$MFA_WORK"

python3 << 'PYEOF'
import os

data_new = os.environ.get("DATA_NEW", "/data1/yangzhuowei/MambaTalk_new/data_new")
mfa_src = os.environ.get("MFA_SRC", "/data1/yangzhuowei/MambaTalk/data_old/mfa_input")
mfa_work = os.path.join(data_new, "mfa_work")

npz_ids = sorted(set(
    f.replace('.npz','')
    for f in os.listdir(os.path.join(data_new, 'smplxflame_30'))
    if f.endswith('.npz')
))

linked = 0
missing_txt = []
for sid in npz_ids:
    wav_src = os.path.abspath(os.path.join(data_new, 'wave16k', f'{sid}.wav'))
    txt_src = os.path.abspath(os.path.join(mfa_src, f'{sid}.txt'))
    if not os.path.exists(txt_src):
        missing_txt.append(sid)
        continue
    if not os.path.exists(wav_src):
        continue
    os.symlink(wav_src, os.path.join(mfa_work, f'{sid}.wav'))
    os.symlink(txt_src, os.path.join(mfa_work, f'{sid}.txt'))
    linked += 1

print(f"  MFA 工作目录准备完成: {linked} 对文件")
if missing_txt:
    print(f"  警告: {len(missing_txt)} 个缺少转录文本，将被跳过")
PYEOF

echo "  运行 MFA 对齐 (这可能需要几分钟)..."
conda run -n "$CONDA_ENV" mfa align --single_speaker --clean \
    "$MFA_WORK" english_us_arpa english_us_arpa "$MFA_OUT" 2>&1 | \
    grep -E "INFO|WARNING|ERROR|Done"

TG_GEN=$(ls "$MFA_OUT/"*.TextGrid 2>/dev/null | wc -l)
echo "  MFA 生成了 $TG_GEN 个 TextGrid 文件"

cp "$MFA_OUT/"*.TextGrid "$DATA_NEW/textgrid/"
echo "  已复制到 textgrid/"

# ---- 5. 生成 train_test_split.csv ----
echo "[5/6] 生成 train_test_split.csv..."
python3 << 'PYEOF'
import os

data_new = os.environ.get("DATA_NEW", "/data1/yangzhuowei/MambaTalk_new/data_new")
train_ids = sorted(set(f.replace('.npz','') for f in os.listdir(os.path.join(data_new, 'train')) if f.endswith('.npz')))
test_ids = sorted(set(f.replace('.npz','') for f in os.listdir(os.path.join(data_new, 'test')) if f.endswith('.npz')))

csv_path = os.path.join(data_new, 'train_test_split.csv')
with open(csv_path, 'w') as f:
    f.write("id,type\n")
    for sid in train_ids:
        f.write(f"{sid},train\n")
    for sid in test_ids:
        f.write(f"{sid},test\n")
print(f"  train_test_split.csv: {len(train_ids)} train + {len(test_ids)} test = {len(train_ids)+len(test_ids)} total")
PYEOF

# ---- 6. 复制 weights ----
echo "[6/6] 复制 weights..."
cp "$DATA_REF/weights/"* "$DATA_NEW/weights/" 2>/dev/null || true
echo "  weights: $(ls "$DATA_NEW/weights/" | tr '\n' ' ')"

# ---- 清理临时文件 ----
echo ""
echo "清理临时文件..."
rm -rf "$MFA_WORK" "$MFA_OUT"

# ---- 验证 ----
echo ""
echo "============================================"
echo "  数据验证"
echo "============================================"
python3 << 'PYEOF'
import os, csv

data_new = os.environ.get("DATA_NEW", "/data1/yangzhuowei/MambaTalk_new/data_new")

train_ids, test_ids = [], []
with open(os.path.join(data_new, 'train_test_split.csv')) as f:
    for row in csv.DictReader(f):
        (train_ids if row['type'] == 'train' else test_ids).append(row['id'])
csv_ids = set(train_ids + test_ids)

npz_ids = set(f.replace('.npz','') for f in os.listdir(os.path.join(data_new, 'smplxflame_30')) if f.endswith('.npz'))
wav_ids = set(f.replace('.wav','') for f in os.listdir(os.path.join(data_new, 'wave16k')) if f.endswith('.wav'))
tg_ids = set(f.replace('.TextGrid','') for f in os.listdir(os.path.join(data_new, 'textgrid')) if f.endswith('.TextGrid'))

complete = csv_ids & npz_ids & wav_ids & tg_ids
train_ok = len(set(train_ids) & complete)
test_ok = len(set(test_ids) & complete)
missing_tg = len(csv_ids - tg_ids)

print(f"train_test_split.csv:  {len(csv_ids)} IDs ({len(train_ids)} train, {len(test_ids)} test)")
print(f"smplxflame_30 (npz):   {len(npz_ids)} files")
print(f"wave16k (wav):         {len(wav_ids)} files")
print(f"textgrid (TextGrid):   {len(tg_ids)} files")
print(f"weights:               {os.listdir(os.path.join(data_new, 'weights'))}")
print()
print(f"完整样本 (npz+wav+TextGrid): {len(complete)}")
print(f"  训练集: {train_ok}/{len(train_ids)}")
print(f"  测试集: {test_ok}/{len(test_ids)}")
if missing_tg:
    print(f"  缺失TextGrid: {missing_tg} (MFA对齐失败, dataloader会自动跳过)")
print()

import subprocess
result = subprocess.run(['du', '-sh', data_new], capture_output=True, text=True)
print(f"总磁盘占用: {result.stdout.strip()}")
PYEOF

echo ""
echo "============================================"
echo "  构建完成!"
echo "============================================"
echo ""
echo "最终目录结构:"
echo "  data_new/"
echo "  ├── train_test_split.csv"
echo "  ├── smplxflame_30/  (npz)"
echo "  ├── wave16k/        (wav)"
echo "  ├── textgrid/       (TextGrid)"
echo "  └── weights/"
