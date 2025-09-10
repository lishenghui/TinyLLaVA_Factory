#!/bin/bash

# MODEL_PATH="/mnt/data/sata/yinghu/checkpoints/llava_factory/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune/"
# MODEL_PATH="tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"
# MODEL_PATH="/mimer/NOBACKUP/groups/bloom/shenghui/TinyLLaVA_Factory/outputs_2nd_stage/checkpoint-900"
MODEL_PATH="/mimer/NOBACKUP/groups/bloom/shenghui/TinyLLaVA_Factory/lora_tinyllama/checkpoint-1600"
# MODEL_PATH="/mimer/NOBACKUP/groups/bloom/shenghui/TinyLLaVA_Factory/tiny-llava-TinyLlama-1.1B-Chat-v1.0-siglip-so400m-patch14-384-base-pretrain/checkpoint-100"
# MODEL_PATH="/mimer/NOBACKUP/groups/bloom/shenghui/TinyLLaVA_Factory/lora_tinyllama/checkpoint-300"

echo "Evaluating model at $MODEL_PATH"
# MODEL_PATH="/mimer/NOBACKUP/groups/bloom/shenghui/LLaVA-Steering/outputs/checkpoint-4361"
MODEL_NAME="tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune"
EVAL_DIR="/mimer/NOBACKUP/groups/bloom/shenghui/TinyLLaVA_Factory/datasets/eval"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $EVAL_DIR/textvqa/train_images \
    --answers-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode llama

python -m tinyllava.eval.eval_textvqa \
    --annotation-file $EVAL_DIR/textvqa/TextVQA_0.5.1_val.json \
    --result-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl

