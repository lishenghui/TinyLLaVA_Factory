#!/bin/bash

MODEL_PATH="/mimer/NOBACKUP/groups/bloom/shenghui/TinyLLaVA_Factory/lora_tinyllama/checkpoint-2200"
# MODEL_PATH="/mimer/NOBACKUP/groups/bloom/shenghui/TinyLLaVA_Factory/lora_tinyllama/checkpoint-300"

echo "Evaluating model at $MODEL_PATH"
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

