def quick_split_jsonl(input_file='data/pretrain/wikipedia/text.jsonl'):
    """
    最简版本：快速按比例分割JSONL文件
    """
    # 计算分割点
    train_lines = 2000000
    dev_lines = 1000
    
    print(f"训练集: {train_lines}, 验证集: {dev_lines}")

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 写入文件
    with open('data/pretrain/wikipedia/train.jsonl', 'w', encoding='utf-8') as f:
        f.writelines(lines[:train_lines])
    
    with open('data/pretrain/wikipedia/dev.jsonl', 'w', encoding='utf-8') as f:
        f.writelines(lines[train_lines:train_lines+dev_lines])
    
    print("分割完成!")

# 使用
quick_split_jsonl()
