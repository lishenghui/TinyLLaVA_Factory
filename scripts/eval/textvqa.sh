#!/bin/bash

# MODEL_PATH="/mimer/NOBACKUP/groups/bloom/shenghui/PaaA/paaa/ray_results/LoRA_TEST/FEDLLM_none_922a9_00000_0_base_model=unsloth_Llama-3_2-1B-Instruct,momentum=0.0000,num_clients=1,random_seed=121,aggregator=type_M_2025-09-21_16-30-50/checkpoint_000002"
MODEL_PATH="/mimer/NOBACKUP/groups/bloom/shenghui/TinyLLaVA_Factory/lora_tinyllama/checkpoint-1100"

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

