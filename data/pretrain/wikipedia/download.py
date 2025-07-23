# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import requests
from urllib.parse import urlparse

from download_tools import get_download_path, get_s3_path

AVAILABLE_CORPORA = {
    "corpora/wiki/enwiki-dec2017": {
        "corpus": "corpora/wiki/enwiki-dec2017",
        "description": "Wikipedia dump from Dec 2017, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl"],
    },
    "corpora/wiki/enwiki-dec2018": {
        "corpus": "corpora/wiki/enwiki-dec2018",
        "description": "Wikipedia dump from Dec 2018, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl"],
    },
    "corpora/wiki/enwiki-aug2019": {
        "corpus": "corpora/wiki/enwiki-aug2019",
        "description": "Wikipedia dump from Aug 2019, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl"],
    },
    "corpora/wiki/enwiki-dec2020": {
        "corpus": "corpora/wiki/enwiki-dec2020",
        "description": "Wikipedia dump from Dec 2020, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl"],
    },
    "corpora/wiki/enwiki-dec2021": {
        "corpus": "corpora/wiki/enwiki-dec2021",
        "description": "Wikipedia dump from Dec 2021, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl"],
    },
}


def maybe_download_file_with_limit(source, target, max_lines=None, save_path=None):
    """
    下载文件并在下载过程中限制行数
    """
    # 检查文件是否已存在
    if os.path.exists(save_path):
        print(f"File {save_path} already exists, skipping download.")
        return
    
    # 确保目标目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Downloading {source} to {save_path}")
    
    try:
        # 流式下载
        response = requests.get(source, stream=True)
        response.raise_for_status()
        
        # 如果是JSONL文件且设置了行数限制
        if target.endswith('.jsonl') and max_lines:
            print(f"Limiting download to {max_lines} lines...")
            
            with open(save_path, 'w', encoding='utf-8') as f:
                line_count = 0
                buffer = ""
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        # 将字节转换为字符串
                        try:
                            chunk_str = chunk.decode('utf-8')
                        except UnicodeDecodeError:
                            # 如果UTF-8解码失败，尝试其他编码
                            try:
                                chunk_str = chunk.decode('latin-1')
                            except UnicodeDecodeError:
                                chunk_str = chunk.decode('utf-8', errors='ignore')
                        
                        buffer += chunk_str
                        
                        # 处理完整的行
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            
                            if line.strip():  # 跳过空行
                                f.write(line + '\n')
                                line_count += 1
                                
                                if line_count >= max_lines:
                                    print(f"Reached maximum lines ({max_lines}), stopping download.")
                                    return
                                
                                # 每10万行显示一次进度
                                if line_count % 100000 == 0:
                                    print(f"Downloaded {line_count} lines...")
                
                # 处理最后一行（如果没有换行符结尾）
                if buffer.strip() and line_count < max_lines:
                    f.write(buffer.rstrip() + '\n')
                    line_count += 1
                
                print(f"Download completed. Total lines: {line_count}")
        
        else:
            # 正常下载（非JSONL文件或无行数限制）
            with open(save_path, 'wb') as f:
                total_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
                        # 每100MB显示一次进度
                        if total_size % (100 * 1024 * 1024) == 0:
                            print(f"Downloaded {total_size // (1024 * 1024)} MB...")
            
            print(f"Download completed: {target}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {source}: {e}")
        # 如果下载失败，删除部分下载的文件
        if os.path.exists(target):
            os.remove(target)
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        # 如果下载失败，删除部分下载的文件
        if os.path.exists(target):
            os.remove(target)
        raise


def _helpstr():
    helpstr = "The following corpora are available for download: "
    for m in AVAILABLE_CORPORA.values():
        helpstr += f'\nCorpus name: {m["corpus"]:<30} Description: {m["description"]}'
    helpstr += "\ndownload by passing --corpus {corpus name}"
    return helpstr


def main(output_directory, requested_corpus, max_lines=None):
    if requested_corpus not in AVAILABLE_CORPORA:
        raise ValueError(f"Unknown corpus: {requested_corpus}")
    
    corpus_info = AVAILABLE_CORPORA[requested_corpus]
    print(f"Downloading corpus: {corpus_info['description']}")
    
    if max_lines:
        print(f"Line limit set to: {max_lines}")
    
    for filename in corpus_info["files"]:
        path = f"{requested_corpus}/{filename}"
        source = get_s3_path(path)
        target = get_download_path(output_directory, path)
        save_path = get_download_path(output_directory, "text.jsonl")
        
        print(f"\nProcessing file: {filename}")
        
        # 对JSONL文件应用行数限制
        if filename.endswith('.jsonl'):
            maybe_download_file_with_limit(source, target, max_lines, save_path)
        else:
            maybe_download_file_with_limit(source, target)


if __name__ == "__main__":
    help_str = _helpstr()
    choices = list(AVAILABLE_CORPORA.keys())
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Path to the directory where the dataset will be saved.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        choices=choices,
        required=True,
        help=help_str,
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=None,
        help="Maximum number of lines to download for JSONL files (default: download all lines). Example: 2000000 for 2M lines",
    )
    
    args = parser.parse_args()
    
    try:
        main(args.output_dir, args.corpus, args.max_lines)
        print("\nDownload process completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
