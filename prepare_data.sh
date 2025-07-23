python data/pretrain/wikipedia/download.py \
    --corpus corpora/wiki/enwiki-dec2021 \
    --output_dir data/pretrain/wikipedia
    --max_lines 2002000

python data/pretrain/wikipedia/split.py

python ift_data_prepare.py