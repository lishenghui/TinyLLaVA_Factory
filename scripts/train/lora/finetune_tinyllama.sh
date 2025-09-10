# DATA_PATH=/home/ai/data/llava/dataset/text_files/blip_laion_cc_sbu_558k.json
FINETUNE_DATA_PATH=/home/ai/data/llava/dataset/text_files/llava_v1_5_mix665k.json
IMAGE_PATH=/home/ai/data/llava/dataset/llava/llava_pretrain/images
FINETUNE_IMAGE_PATH=/home/ai/data/llava/dataset

LLM_VERSION=microsoft/phi-2
VT_VERSION=google/siglip-so400m-patch14-384
VT_VERSION2=""
CN_VERSION=mlp2x_gelu
CONV_VERSION=phi
VERSION=base-lora-zero2-r128
PRETRAIN_TRAIN_RECIPE=common
MODEL_MAX_LENGTH=3072


FINETUNE_DATA_PATH=/mimer/NOBACKUP/groups/bloom/shenghui/LLaVA-Steering/datasets/train/text_files/mini_train.json
FINETUNE_IMAGE_PATH=/mimer/NOBACKUP/groups/bloom/shenghui/LLaVA-Steering/datasets/train

VERSION=
FINETUNE_TRAIN_RECIPE=lora
OUTPUT_DIR=/mimer/NOBACKUP/groups/bloom/shenghui/TinyLLaVA_Factory/outputs_2nd_stage
RUN_NAME=tiny-llava-phi-2-siglip-so400m-patch14-384-base-pretrain-full
LLM_VERSION=TinyLlama/TinyLlama-1.1B-Chat-v1.0
VT_VERSION=google/siglip-so400m-patch14-384
VT_VERSION2=""
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
MODEL_MAX_LENGTH=2048
CONV_VERSION=llama

# bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$PRETRAIN_TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
bash scripts/train/lora/finetune_lora.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$FINETUNE_TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
