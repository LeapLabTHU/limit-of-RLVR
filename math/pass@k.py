import numpy as np
import json

def get_pass_k_data(file_dir_path, model_name, benchmark_name, template, temperature, total_questions_num, n, seed_max):
    '''
    Assume that the scripts you have run use the run_name

        Qwen2.5-7B_aime24_temp0.6_n512_seed1
        Qwen2.5-7B_aime24_temp0.6_n512_seed2

    And the results are saved in

        /examples/math_eval/EVAL/checkpoints/
            Qwen2.5-7B_aime24_temp0.6_n512_seed1/
            Qwen2.5-7B_aime24_temp0.6_n512_seed2/

    Then you should set:

        file_dir_path = "/examples/math_eval/EVAL/checkpoints/"
        model_name = "Qwen2.5-7B"
        benchmark_name = "aime24"
        template = "qwen-boxed"
        temperature = 0.6
        total_questions_num = 30 # Since aime24 has 30 questions
        n = 512
        seed_max = 2

    '''

    N = n * seed_max

    def read_scores_from_jsonl(file_path):
        try:
            with open(file_path, 'r') as file:
                data = [json.loads(line) for line in file]
                scores = [item.get('score') for item in data]
                return scores
        except FileNotFoundError:
            print(f"Error: {file_path} not found.")
        except json.JSONDecodeError:
            print(f"Error: {file_path} invalid.")
        except Exception as e:
            print(f"Unknown error: {e}")
        return []

    def read_jsonl_file(file_dir_path, model_name, True_n_ls, total_questions_num, n, seed_max):
        '''
        Example
        '''
        for seed in range(1, seed_max + 1):
            file_path = f'{file_dir_path}/{model_name}_{benchmark_name}_temp{temperature}_n{n}_seed{seed}/eval_results/global_step_0/{benchmark_name}/test_qwen-boxed_-1_seed{seed}_t{temperature}_s0_e-1.jsonl'
            scores = read_scores_from_jsonl(file_path=file_path)
            print(f"Read {file_path} successfully.")
            for i in range(total_questions_num):
                for j in range(n):
                    True_n_ls[(seed-1) * n + j][i] = int(scores[i][j])
    
    def pass_at_k(n, c, k):
        if n - c < k: return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    True_n_ls = [[0 for i in range(total_questions_num)] for j in range(N)]
    read_jsonl_file(file_dir_path, model_name, True_n_ls, total_questions_num, n, seed_max)
    True_n_ls = np.array(True_n_ls)

    correct_ls = [0 for i in range(total_questions_num)]
    for i in range(total_questions_num):
        for j in range(n * seed_max):
            correct_ls[i] += True_n_ls[j][i]

    # print("correct_ls = ", correct_ls)
    pass_at_k_ls = [[] for i in range(N)]
    for i in range(N):
        for j in range(total_questions_num):
            pass_at_k_ls[i].append(pass_at_k(n * seed_max, correct_ls[j], i+1))
    
    # print("pass_at_k_ls = ", pass_at_k_ls)

    pass_at_k_mean_ls = [np.mean(pass_at_k_ls[i]) for i in range(N)]
    # print("pass_at_k_mean_ls = ", pass_at_k_mean_ls)

    pass_at_k_mean_ls = np.array(pass_at_k_mean_ls)

    return pass_at_k_mean_ls

'''
pass_at_k_mean_ls is a numpy array, pass_at_k_mean_ls[k-1] means pass@k.
'''