def quick_split_jsonl(input_file='data.jsonl', train_ratio=0.8, dev_ratio=0.1):
    """
    最简版本：快速按比例分割JSONL文件
    """
    # 统计总行数
    with open(input_file, 'rb') as f:
        total_lines = sum(1 for _ in f)
    
    # 计算分割点
    train_lines = 2000000
    dev_lines = 1000
    
    print(f"总行数: {total_lines}, 训练集: {train_lines}, 验证集: {dev_lines}, 测试集: {total_lines-train_lines-dev_lines}")
    
    # 分割文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 写入文件
    with open('train.jsonl', 'w', encoding='utf-8') as f:
        f.writelines(lines[:train_lines])
    
    with open('dev.jsonl', 'w', encoding='utf-8') as f:
        f.writelines(lines[train_lines:train_lines+dev_lines])
    
    with open('test.jsonl', 'w', encoding='utf-8') as f:
        f.writelines(lines[train_lines+dev_lines:])
    
    print("分割完成!")

# 使用
quick_split_jsonl()
