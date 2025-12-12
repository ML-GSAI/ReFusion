# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

export CUDA_VISIBLE_DEVICES=0

# refusion
accelerate launch eval.py --tasks gsm8k_cot --model qwen3_dist --batch_size 1 --num_fewshot 0 --log_samples --apply_chat_template --output_path output_reproduce/gsm8k_cot --model_args model_path='./output_checkpoints',gen_length=512,slot_size=32,serial_num_blocks=4,slot_threshold=0.9,token_threshold=0.4,save_dir="./output_statistics/gsm8k_cot"

accelerate launch eval.py --tasks gpqa_main_generative_n_shot --model qwen3_dist --confirm_run_unsafe_code --num_fewshot 0 --log_samples --apply_chat_template --output_path output_reproduce/gpqa_main_generative_n_shot --model_args model_path='./output_checkpoints',gen_length=512,slot_size=8,serial_num_blocks=32,slot_threshold=0.8,token_threshold=0.2,save_dir="./output_statistics/gpqa_main_generative_n_shot"

accelerate launch eval.py --tasks minerva_math --model qwen3_dist --num_fewshot 0 --output_path output_reproduce/minerva_math --log_samples --apply_chat_template --model_args model_path='./output_checkpoints',gen_length=512,slot_size=32,serial_num_blocks=8,slot_threshold=0.8,token_threshold=0.6,save_dir="./output_statistics/minerva_math"

accelerate launch eval.py --tasks mbpp --model qwen3_dist --confirm_run_unsafe_code --num_fewshot 0 --output_path output_reproduce/mbpp --log_samples --apply_chat_template --model_args model_path='./output_checkpoints',gen_length=512,slot_size=32,serial_num_blocks=4,slot_threshold=0.9,token_threshold=0.3,save_dir="./output_statistics/mbpp"

accelerate launch eval.py --tasks humaneval_instruct_noprefix --model qwen3_dist --confirm_run_unsafe_code --num_fewshot 0 --output_path output_reproduce/humaneval_instruct_noprefix --log_samples --apply_chat_template --model_args model_path='./output_checkpoints',gen_length=512,slot_size=32,serial_num_blocks=4,slot_threshold=0.9,token_threshold=0.4,save_dir="./output_statistics/humaneval_instruct_noprefix"

accelerate launch eval.py --tasks arc_challenge_chat --model qwen3_dist --confirm_run_unsafe_code --num_fewshot 0 --output_path output_reproduce/arc_challenge_chat --log_samples --apply_chat_template --model_args model_path='./output_checkpoints',gen_length=512,slot_size=8,serial_num_blocks=64,slot_threshold=0.8,token_threshold=0.1,save_dir="./output_statistics/arc_challenge_chat"

accelerate launch eval.py --tasks mmlu_pro --model qwen3_dist --confirm_run_unsafe_code --num_fewshot 0 --output_path output_reproduce/mmlu_pro --log_samples --apply_chat_template --model_args model_path='./output_checkpoints',gen_length=512,slot_size=16,serial_num_blocks=4,slot_threshold=0.9,token_threshold=0.4,save_dir="./output_statistics/mmlu_pro"
