import os
import json
import glob
import argparse
from collections import defaultdict
import pandas as pd
from transformers import AutoTokenizer
import wandb
from tqdm import tqdm
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor
import threading
import matplotlib.pyplot as plt
import re

# Create a thread-local storage for tokenizer
thread_local = threading.local()

def extract_last_boxed(text):
    """Extract content inside the last \boxed in LaTeX text"""
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(0)
    return None

def get_tokenizer(model_name):
    """Get or create thread-local tokenizer"""
    if not hasattr(thread_local, 'tokenizer'):
        thread_local.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return thread_local.tokenizer

def normalize_model_name(path):
    """Extract and normalize model name from path"""
    parts = path.split('/')
    # First check for checkpoint pattern
    for part in parts[::-1]:
        if 'checkpoint' in part:
            idx = parts.index(part)
            model_name = parts[idx-1]
            checkpoint = part
            return f"{model_name}-{checkpoint}"
        # Add check for global_step pattern
        if 'global_step' in part:
            idx = parts.index(part)
            model_name = parts[idx-1]
            return f"{model_name}-{part}"
    
    # If no checkpoint or global_step found, use the last meaningful part and add checkpoint-final
    for part in reversed(parts):
        if any(x in part.lower() for x in ['llama', 'qwen', 'gpt', 'mistral']):
            return f"{part}-checkpoint-final"
    
    return "unknown_model"

def get_benchmark_name(path):
    """Extract benchmark name from path"""
    parts = path.split('/')
    return parts[-2]


import os
import json
import jieba
import re


def contains_chinese(string):
    for char in string:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

def jaccard_similarity(sentence1, sentence2):
    if contains_chinese(sentence1):
        set1 = set(jieba.cut(sentence1))
    else:
        if " " not in sentence1 or "\n" not in sentence1:
            set1 = set(sentence1)
        else:
            set1 = set(sentence1.split())
    if contains_chinese(sentence2):
        set2 = set(jieba.cut(sentence2))
    else:
        if " " not in sentence2 or "\n" not in sentence2:
            set2 = set(sentence2)
        else:
            set2 = set(sentence2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def is_repeat(text, window_size=10, threshold=0.85, min_length=20):
    if len(text) <= window_size:
        return False
    pre = text[:window_size]
    for i in range(1, len(text) // window_size):
        cur = text[window_size * i : window_size * (i + 1)]
        if jaccard_similarity(pre, cur) >= threshold:
            return True
        pre = cur

    for char in ["\n", ".", "。"]:
        text_split = text.split(char)
        if len(text_split) == 1:
            return False
        text_split = [t for t in text_split if len(t) >= min_length]
        pre = text_split[0]
        for cur in text_split[1:]:
            if jaccard_similarity(pre, cur) >= threshold:
                return True
            pre = cur
  
    return False

def get_jsonl_path(metrics_file):
    """Get corresponding jsonl file path"""
    # Get the directory containing the metrics file
    metric_folder = os.path.dirname(metrics_file)
    
    # The JSONL file should be in the same directory with a .jsonl extension
    # and without the '_metrics' suffix
    base_name = os.path.basename(metrics_file).replace('_metrics.json', '')
    jsonl_file = os.path.join(metric_folder, f"{base_name}.jsonl")
    
    if not os.path.exists(jsonl_file):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")
    
    return jsonl_file

def calculate_avg_tokens_and_keywords(jsonl_path, tokenizer):
    """Calculate average tokens and keyword frequencies in the first code element"""
    if not os.path.exists(jsonl_path):
        print(f"Warning: JSONL file not found: {jsonl_path}")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0

    keywords = {"recheck", "rethink", "try again", "wait", "alternatively", "retry", "however"}
    total_tokens = 0
    total_keywords = 0
    total_correct_tokens = 0
    total_wrong_tokens = 0
    total_stop_tokens = 0
    clip_count = 0
    total_repeats = 0
    count = 0
    correct_count = 0
    wrong_count = 0
    stop_count = 0
    box_count = 0
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'code' in data and isinstance(data['code'], list) and len(data['code']) > 0:
                    code_text = data['code'][0].lower()
                    tokens = len(tokenizer.encode(code_text))
                    total_tokens += tokens
                    
                    # Count keywords
                    keyword_count = sum(code_text.count(keyword.lower()) for keyword in keywords)
                    total_keywords += keyword_count
                    # Check for \boxed occurrences
                    if extract_last_boxed(code_text) is not None:
                        box_count += 1
                    # Check finish reason
                    if data.get('finish_reason', [None])[0] == 'length':
                        clip_count += 1
                    elif data.get('finish_reason', [None])[0] == 'stop':
                        total_stop_tokens += tokens
                        stop_count += 1
                    
                    # Separate tokens for correct and wrong answers
                    is_correct = data.get('score', [False])[0] if isinstance(data.get('score', []), list) else False
                    if is_correct:
                        total_correct_tokens += tokens
                        correct_count += 1
                    else:
                        total_wrong_tokens += tokens
                        wrong_count += 1
                    try:
                        if is_repeat(code_text):
                    #    repeat_count += 1
                            total_repeats += 1
                    except Exception as e:
                        # print("test")
                        total_repeats += 1
                    count += 1
    except Exception as e:
        print(f"Error processing {jsonl_path}: {e}")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0
        
    avg_correct_tokens = total_correct_tokens / correct_count if correct_count > 0 else 0
    avg_wrong_tokens = total_wrong_tokens / wrong_count if wrong_count > 0 else 0
    clip_ratio = clip_count / count if count > 0 else 0
    avg_stop_tokens = total_stop_tokens / stop_count if stop_count > 0 else 0
    box_ratio = box_count / count if count > 0 else 0  # Calculate the ratio of boxed occurrences
    repeat_ratio = total_repeats / count if count > 0 else 0  # Calculate the repeat ratio
    return (total_tokens / count if count > 0 else 0,
            total_keywords / count if count > 0 else 0,
            avg_correct_tokens,
            avg_wrong_tokens,
            clip_ratio,
            avg_stop_tokens,
            box_ratio,  # Return the boxed ratio
            stop_count / count if count > 0 else 0,
            repeat_ratio)

def process_file(args):
    """Process a single metrics file"""
    metrics_file, model_name = args
    try:
        model_name_norm = normalize_model_name(metrics_file)
        benchmark = get_benchmark_name(metrics_file)
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            acc = metrics.get('acc', 0)
            pass_acc = metrics.get('pass_acc', 0)
        
        jsonl_file = get_jsonl_path(metrics_file)
        tokenizer = get_tokenizer(model_name)
        avg_tokens, avg_keywords, avg_correct_tokens, avg_wrong_tokens, clip_ratio, avg_stop_tokens, box_ratio, stop_ratio, repeat_ratio = calculate_avg_tokens_and_keywords(jsonl_file, tokenizer)
        
        return model_name_norm, benchmark, {
            'acc': acc,
            "pass_acc": pass_acc,
            'tokens': avg_tokens,
            'keywords': avg_keywords,
            'correct_tokens': avg_correct_tokens,
            'wrong_tokens': avg_wrong_tokens,
            'clip_ratio': clip_ratio,
            'avg_stop_tokens': avg_stop_tokens,
            'stop_ratio': stop_ratio,
            'box_ratio': box_ratio,
            'repeat_ratio': repeat_ratio
        }
        
    except Exception as e:
        print(f"Error processing {metrics_file}: {e}")
        return None

def collect_results(base_dir, model_name, num_threads=8, temperature=None):
    # Initialize results storage
    results = defaultdict(lambda: defaultdict(dict))
    
    # Find all metrics.json files
    metrics_files = glob.glob(f"{base_dir}/**/test_*metrics.json", recursive=True)
    
    if temperature is not None:
        metrics_files = [f for f in metrics_files if f"t{temperature}" in f]
    
    print("metrics_files ==== ", metrics_files)
    
    # Create arguments for parallel processing
    process_args = [(f, model_name) for f in metrics_files]
    print("process_args ==== ", process_args)
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = list(tqdm(
            executor.map(process_file, process_args),
            total=len(metrics_files),
            desc="Processing files"
        ))
        
        # Collect results
        for result in futures:
            if result is not None:
                model_name, benchmark, metrics = result
                results[model_name][benchmark] = metrics
    
    return results
def create_summary(results):
    # Convert results to DataFrame
    print("results ==== ")
    for itm in results.items():
        print(itm)
    rows = []
    for model, benchmarks in results.items():
        row = {'model': model}
        print("model ==== ", model)
        total_acc = 0
        total_pass_acc = 0
        total_tokens = 0
        total_keywords = 0
        total_correct_tokens = 0
        total_wrong_tokens = 0
        total_clip_ratio = 0
        total_stop_tokens = 0
        total_stop_ratio = 0
        total_box_ratio = 0
        total_repeat_ratio = 0  # Track total repeat ratio
        count = 0
        
        for benchmark, metrics in benchmarks.items():
            # Add accuracy and token metrics
            row[f'{benchmark}_acc'] = metrics['acc']
            row[f'{benchmark}_pass_acc'] = metrics['pass_acc']
            row[f'{benchmark}_tokens'] = metrics['tokens']
            row[f'{benchmark}_keywords'] = metrics['keywords']
            row[f'{benchmark}_correct_tokens'] = metrics['correct_tokens'] 
            row[f'{benchmark}_wrong_tokens'] = metrics['wrong_tokens']
            row[f'{benchmark}_clip_ratio'] = metrics['clip_ratio']
            row[f'{benchmark}_stop_tokens'] = metrics['avg_stop_tokens']
            row[f'{benchmark}_stop_ratio'] = metrics['stop_ratio']
            row[f'{benchmark}_box_ratio'] = metrics['box_ratio']  # Add box_ratio to the row
            row[f'{benchmark}_repeat_ratio'] = metrics['repeat_ratio']  # Add repeat_ratio to the row
            
            # Accumulate totals
            total_acc += metrics['acc']
            total_pass_acc += metrics['pass_acc']
            total_tokens += metrics['tokens']
            total_keywords += metrics['keywords']
            total_correct_tokens += metrics['correct_tokens']
            total_wrong_tokens += metrics['wrong_tokens']
            total_clip_ratio += metrics['clip_ratio']
            total_stop_tokens += metrics['avg_stop_tokens']
            total_stop_ratio += metrics['stop_ratio']
            total_box_ratio += metrics['box_ratio']
            total_repeat_ratio += metrics['repeat_ratio']  # Add repeat_ratio to the total
            count += 1
        
        if count > 0:
            # Calculate averages across all benchmarks
            row['avg_acc'] = total_acc / count
            row['avg_pass_acc'] = total_pass_acc / count
            row['avg_tokens'] = total_tokens / count
            row['avg_keywords'] = total_keywords / count
            row['avg_correct_tokens'] = total_correct_tokens / count
            row['avg_wrong_tokens'] = total_wrong_tokens / count
            row['avg_clip_ratio'] = total_clip_ratio / count
            row['avg_stop_tokens'] = total_stop_tokens / count
            row['avg_stop_ratio'] = total_stop_ratio / count
            row['avg_box_ratio'] = total_box_ratio / count  # Average box_ratio
            row['avg_repeat_ratio'] = total_repeat_ratio / count  # Average repeat_ratio
        
        rows.append(row)
    
    print("rows ==== ", rows)
    df = pd.DataFrame(rows)
    
    # Sort DataFrame by checkpoint/global_step number
    def get_step_number(model_name):
        if 'checkpoint-final' in model_name:
            return float('inf')
        # Check for checkpoint pattern
        checkpoint_match = re.search(r'checkpoint-(\d+)', model_name)
        if checkpoint_match:
            return int(checkpoint_match.group(1))
        # Check for global_step pattern
        global_step_match = re.search(r'global_step[_]?(\d+)', model_name)
        if global_step_match:
            return int(global_step_match.group(1))
        return float('inf')
    
    # Sort DataFrame based on step numbers
    print("df ==== ", df)
    df['sort_key'] = df['model'].apply(get_step_number)
    df = df.sort_values('sort_key')
    df = df.drop('sort_key', axis=1)
    
    return df

def sync_to_wandb(args, results, project_name, df, plot_dir, csv_path):
    """Sync results, CSV table and plots to wandb"""
    # Initialize wandb run
    run = wandb.init(
        project=project_name,
        name=args.wandb_run_name,
        reinit=True
    )
    
    # Log the CSV table as a wandb Table
    table = wandb.Table(dataframe=df)
    wandb.log({"results_table": table})
    
    # Also save the CSV file as an artifact
    artifact = wandb.Artifact('evaluation_results', type='dataset')
    artifact.add_file(csv_path)
    run.log_artifact(artifact)
    
    # Log plots
    if os.path.exists(plot_dir):
        for plot_file in os.listdir(plot_dir):
            if plot_file.endswith('_progress.png'):
                plot_path = os.path.join(plot_dir, plot_file)
                wandb.log({f"plots/{plot_file}": wandb.Image(plot_path)})
                
            if plot_file.endswith('_tokens_keywords.png'):
                plot_path = os.path.join(plot_dir, plot_file)
                wandb.log({f"plots/{plot_file}": wandb.Image(plot_path)})
                
            if plot_file.endswith('_acc_tokens.png'):
                plot_path = os.path.join(plot_dir, plot_file)
                wandb.log({f"plots/{plot_file}": wandb.Image(plot_path)})
                
            if plot_file.endswith('_acc_keywords.png'):
                plot_path = os.path.join(plot_dir, plot_file)
                wandb.log({f"plots/{plot_file}": wandb.Image(plot_path)})
            if plot_file.endswith('_correct_tokens.png'):
                plot_path = os.path.join(plot_dir, plot_file)
                wandb.log({f"plots/{plot_file}": wandb.Image(plot_path)})
            if plot_file.endswith('_wrong_tokens.png'):
                plot_path = os.path.join(plot_dir, plot_file)
                wandb.log({f"plots/{plot_file}": wandb.Image(plot_path)})
            if plot_file.endswith('_clip_ratio.png'):
                plot_path = os.path.join(plot_dir, plot_file)
                wandb.log({f"plots/{plot_file}": wandb.Image(plot_path)})
            if plot_file.endswith('_avg_stop_tokens.png'):
                plot_path = os.path.join(plot_dir, plot_file)
                wandb.log({f"plots/{plot_file}": wandb.Image(plot_path)})
            if plot_file.endswith('box_ratio_and_token_length.png'):
                plot_path = os.path.join(plot_dir, plot_file)
                wandb.log({f"plots/{plot_file}": wandb.Image(plot_path)})
            if plot_file.endswith('repeat_ratio_and_token_length.png'):
                plot_path = os.path.join(plot_dir, plot_file)
                wandb.log({f"plots/{plot_file}": wandb.Image(plot_path)})
            if plot_file.endswith('pass_acc.png'):
                plot_path = os.path.join(plot_dir, plot_file)
                wandb.log({f"plots/{plot_file}": wandb.Image(plot_path)})
            

    run.finish()

def sort_checkpoints(models):
    """Sort checkpoints numerically with final checkpoint at the end"""
    def get_checkpoint_num(model_name):
        if 'checkpoint-final' in model_name:
            return float('inf')
        # Check for checkpoint pattern
        checkpoint_match = re.search(r'checkpoint-(\d+)', model_name)
        if checkpoint_match:
            return int(checkpoint_match.group(1))
        # Check for global_step pattern
        global_step_match = re.search(r'global_step[_]?(\d+)', model_name)
        if global_step_match:
            return int(global_step_match.group(1))
        return float('inf')
    
    # Group models by base name (everything before checkpoint- or global_step)
    model_groups = defaultdict(list)
    for model in models:
        # Split on either checkpoint- or global_step
        base_name = re.split(r'(?:checkpoint-|global_step)', model)[0].rstrip('-')
        model_groups[base_name].append(model)
    
    # Sort each group's checkpoints
    sorted_models = []
    for base_name, checkpoints in model_groups.items():
        sorted_checkpoints = sorted(checkpoints, key=get_checkpoint_num)
        sorted_models.extend(sorted_checkpoints)
    
    return sorted_models

def main(args):
    base_dir = args.base_dir
    model_name = args.model_name
    print("model_name:", model_name)
    
    # Parse benchmarks if specified
    benchmarks = None
    if args.benchmarks:
        benchmarks = set(args.benchmarks.split(','))
    
    # Collect results
    print("Collecting results...")
    results = collect_results(base_dir, model_name, args.num_threads, args.temperature)
    
    # Filter results if benchmarks specified
    if benchmarks:
        filtered_results = defaultdict(lambda: defaultdict(dict))
        for model, model_results in results.items():
            for benchmark, metrics in model_results.items():
                if benchmark in benchmarks:
                    filtered_results[model][benchmark] = metrics
        results = filtered_results
    
    # Create summary DataFrame
    print("\nCreating summary...")
    df = create_summary(results)
    print("\nResults summary:")
    print(df)
    
     # Save to CSV
    output_file = args.output_path
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--model_name", type=str, default="Qwen-math-7B-S100-qwq-fs-7k8-8192len-5e-6-rope10-bsz64")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--plot_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="math-eval-results")
    parser.add_argument("--wandb_api_key", type=str, default="1234567890")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--benchmarks", type=str, 
                       default="gsm8k,math,minerva_math,olympiadbench,college_math,aime24,amc23",
                       help="Comma-separated list of benchmarks to include")
    parser.add_argument("--temperature", type=float, default=None)
    
    args = parser.parse_args()
    
    if args.temperature == -1:
        args.temperature = None
    
    if args.output_path is None:
        args.output_path = os.path.join(args.base_dir, "eval_results.csv")
    
    if args.plot_dir is None:
        args.plot_dir = os.path.join(args.base_dir, "plots")
        
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir, exist_ok=True)
        
    main(args)
