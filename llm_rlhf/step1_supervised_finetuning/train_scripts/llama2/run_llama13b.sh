ROOT_DIR=/share/project/weiyifan/KG_RAG/llm_rlhf

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# DeepSpeed Team

OUTPUT=$1
ZERO_STAGE=$2

if [ "$OUTPUT" == "" ]; then
    OUTPUT=/share/project/weiyifan/KG_RAG/results/checkpoints/output_step1_Llama-3-8B
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT

deepspeed  main2.py \
    --deepspeed \
    --project_name rlhf_llama \
    --experiment_name sft_kg_Llama-3-8B \
    --data_path /share/project/duli/content_relation_ana/tag_interaction_analyses/down_stream_ana/data/post_pret_reformat/deita_20k.jsonl \
    --model_name_or_path /share/models/Llama-3-8B \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --save_interval 4000 \
    --eval_interval 500 \
    --max_seq_len 1024 \
    --learning_rate 9.65e-6 \
    --loss_scale 0.0 \
    --loss_scale_window 100 \
    --weight_decay -1 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps -1 \
    --seed 1234 \
    --gradient_checkpointing \
    --zero_stage $ZERO_STAGE \
    --output_dir $OUTPUT \
    --part_data_size -1 \
    | tee $OUTPUT/training.log \


    #--hostfile ${ROOT_DIR}/config/step1hostfile