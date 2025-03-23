ROOT_DIR=/share/project/weiyifan/KG_RAG/llm_rlhf

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# DeepSpeed Team

OUTPUT=$1
ZERO_STAGE=$2

if [ "$OUTPUT" == "" ]; then
    OUTPUT=/share/project/weiyifan/KG_RAG/results/qwen_ckpt/step1_Qwen2-7B
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

deepspeed  main2.py \
    --deepspeed \
    --project_name kg_rag \
     --experiment_name sft_kg_Qwen2-7B \
    --data_path /share/project/weiyifan/KG_RAG/data/benchmark_data/PMC_sft_data_llama.jsonl \
    --model_name_or_path /share/project/weiyifan/KG_RAG/results/checkpoints/output_step1.1_Qwen2-7B/epoch-2-87105 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --save_interval 5000 \
    --eval_interval 100 \
    --max_seq_len 1024 \
    --learning_rate 9.65e-6 \
    --loss_scale 0.0 \
    --loss_scale_window 100 \
    --weight_decay -1 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --num_warmup_steps -1 \
    --seed 1234 \
    --gradient_checkpointing \
    --zero_stage $ZERO_STAGE \
    --output_dir $OUTPUT \
    --part_data_size -1 \
    | tee $OUTPUT/training.log \
