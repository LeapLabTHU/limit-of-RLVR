## Quick Start

### Installation

```bash
conda create -n verl python==3.9
conda activate verl
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install -e . 
```

### Generation & Evaluation

```bash
bash eval_math_nodes.sh \
    --run_name Qwen2.5-7B_minerva_math_temp0.6_n32_seed1 \
    --init_model Qwen2.5-7B \
    --template qwen-boxed  \
    --tp_size 1 \
    --add_step_0 true  \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens 16000 \
    --benchmarks minerva_math \
    --n_sampling 32 \
    --just_wandb false \
    --seed 1
```

After running the script, the evaluation results will be saved in `examples/math_eval/EVAL/checkpoints/$RUN_NAME/eval_results`, with the metrics saved in `$RUN_NAME/eval_results/eval_results.csv`. A useful function to get the *pass@k* data is given in `pass@k.py`. You can modify it and get the *pass@k* data.

**An example is given in `pass@k.py` to print the *pass@k* data. You can follow the example.**

**An important matter** is that since we use the same model to sample multiple times on the same dataset, it is essential to ensure that the responses obtained from different runs are different, as well as the responses from different samplings within a single run. To this end, the functionality has been integrated into the top-level interface, and you only need to pass parameters in the following manner.

To make responses from different runs distinct, simply set the random seed as follows:

```bash
bash eval_math_nodes.sh \
    --seed 1 # The seed you set should be different
```

To ensure that responses from different samplings within a single run differ, simply pass the number of samplings for a single run as follows, without needing to perform any other actions:

```bash
bash eval_math_nodes.sh \
    --n_sampling 32 \
```

### Applicability

The framework is applicable to *SimpleRL-Zoo*, *Oat-Zero*, *DAPO* series and corresponding *base* models.
