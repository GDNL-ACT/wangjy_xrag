# 评估ROUGE-L
python rouge.py --input_file eval_results.json --tokenizer_path wandb/offline-run-20250713_114215-jpddi4z3/files/checkpoint/last --output_file eval_results.json

# 评估F1
python f1.py --input_file eval_results.json --tokenizer_path wandb/offline-run-20250713_114215-jpddi4z3/files/checkpoint/last --output_file eval_results.json

# 评估cos sim
python sim.py --input_file eval_results.json --output_file eval_results.json
