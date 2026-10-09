#!/usr/bin/env bash
#
# Download the quantized Qwen-Image-2.1 files into a ComfyUI models tree.
#
# Three files are required; the DiT alone is not enough:
#
#   unet/qwen_image_2.1-Q4_K_M.gguf            4.34 GB   diffusion transformer
#   text_encoders/qwen3vl_8b_w4a8.safetensors  6.31 GB   Qwen3-VL-8B text encoder
#   vae/qwen_image_2.1_vae_bf16.safetensors    0.68 GB   VAE
#
# huggingface.co is often unreachable from mainland China; HF_ENDPOINT defaults
# to a mirror. Sizes are verified after download.
#
set -euo pipefail

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MODELS="${MODELS_DIR:-/opt/ComfyUI/models}"

DiT_REPO="${DiT_REPO:-pottokao/Qwen-Image-2.1-DiT-GGUF}"
DiT_FILE="${DiT_FILE:-qwen_image_2.1-Q4_K_M.gguf}"
TE_REPO="Comfy-Org/Qwen-Image-2.1"
TE_FILE="text_encoders/qwen3vl_8b_w4a8.safetensors"
VAE_FILE="vae/qwen_image_2.1_vae_bf16.safetensors"

declare -A EXPECT=(
  ["$MODELS/unet/$DiT_FILE"]=4335931552
  ["$MODELS/text_encoders/qwen3vl_8b_w4a8.safetensors"]=6312105364
  ["$MODELS/vae/qwen_image_2.1_vae_bf16.safetensors"]=675509688
)

mkdir -p "$MODELS/unet" "$MODELS/text_encoders" "$MODELS/vae"

command -v wget >/dev/null || { echo "wget is required"; exit 1; }

fetch() {  # repo remote_path dest
  local url="$HF_ENDPOINT/$1/resolve/main/$2"
  local dest="$3"
  local want="${EXPECT[$dest]:-0}"

  if [ -f "$dest" ]; then
    local have; have=$(stat -c %s "$dest")
    if [ "$want" != "0" ] && [ "$have" = "$want" ]; then
      echo "  skip   $(basename "$dest")  (already complete)"
      return 0
    fi
    echo "  resuming $(basename "$dest")  ($have / $want bytes)"
  fi

  echo "  get    $2"
  wget -c --tries=8 --timeout=60 --waitretry=5 --progress=dot:giga \
       -O "$dest.part" "$url" 2>&1 | tail -2
  mv -f "$dest.part" "$dest"

  local have; have=$(stat -c %s "$dest")
  if [ "$want" != "0" ] && [ "$have" != "$want" ]; then
    echo "  ERROR: size mismatch for $(basename "$dest"): $have != $want" >&2
    return 1
  fi
  echo "  ok     $(basename "$dest")  $((have / 1048576)) MiB"
}

echo "Downloading Qwen-Image-2.1 (quantized) from $HF_ENDPOINT"
fetch "$DiT_REPO" "$DiT_FILE" "$MODELS/unet/$DiT_FILE"
fetch "$TE_REPO"  "$TE_FILE"  "$MODELS/text_encoders/$(basename "$TE_FILE")"
fetch "$TE_REPO"  "$VAE_FILE" "$MODELS/vae/$(basename "$VAE_FILE")"

echo
echo "Verifying GGUF architecture string..."
python3 - "$MODELS/unet/$DiT_FILE" <<'PY'
import struct, sys
path = sys.argv[1]
with open(path, "rb") as f:
    if f.read(4) != b"GGUF":
        print("  ERROR: not a GGUF file"); sys.exit(1)
    struct.unpack("<I", f.read(4))[0]
    struct.unpack("<Q", f.read(8))[0]          # n_tensors
    n_kv = struct.unpack("<Q", f.read(8))[0]

    def rd_str():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    arch = None
    for _ in range(n_kv):
        k = rd_str()
        t = struct.unpack("<I", f.read(4))[0]
        if t == 8:
            v = rd_str()
        elif t in (0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12):
            v = struct.unpack("<" + {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",
                                     6:"f",7:"?",10:"Q",11:"q",12:"d"}[t],
                              f.read(struct.calcsize("<" + {0:"B",1:"b",2:"H",3:"h",
                                     4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}[t])))[0]
        else:
            break
        if k == "general.architecture":
            arch = v
print(f"  general.architecture = {arch}")
if arch != "qwen_image":
    print("  ERROR: expected 'qwen_image' -- wrong file?"); sys.exit(1)
print("  OK: this is a Qwen-Image DiT")
PY

echo
echo "All model files present and verified."
