<div align="center">
<h1>Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?</h1>


[Yang Yue](https://yueyang130.github.io/)<sup>1*</sup>†,  [Zhiqi Chen](https://zhiqichen05.github.io/)<sup>1*</sup>,  [Rui Lu](https://lr32768.github.io/)<sup>1</sup>,  [Andrew Zhao](https://andrewzh112.github.io/)<sup>1</sup>,  [Zhaokai Wang](https://www.wzk.plus/)<sup>2</sup>,  [Yang Yue](https://scholar.google.com/citations?user=Q9cLkdcAAAAJ&hl=en)<sup>1</sup>, [Shiji Song](https://scholar.google.com/citations?user=rw6vWdcAAAAJ&hl=zh-TW)<sup>1</sup>,  [Gao Huang](http://www.gaohuang.net/)<sup>1‡</sup>  

<sup>1</sup> Tsinghua University, LeapLab  <sup>2</sup> Shanghai Jiao Tong University  

<sup>*</sup> Equal Contribution <sup>†</sup> Project Lead <sup>‡</sup> Corresponding Author  


<a href="https://arxiv.org/abs/2504.13837"><img src='https://img.shields.io/badge/arXiv-limit_of_RLVR-red' alt='Paper PDF'>  </a><a href='https://limit-of-rlvr.github.io/'><img src='https://img.shields.io/badge/Project_Page-limit_of_RLVR-green' alt='Project Page'></a> 
 <!-- <a href='https://huggingface.co/datasets/magicr/phyworld'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-phyworld-blue'></a> -->
</div>


## News

- **2025.5.17**: Updated the paper on [arXiv](https://arxiv.org/abs/2504.13837) with new experiments involving DAPO and DeepScaler. Added detailed analysis on entropy, KL divergence, and the impact of rollout numbers.
- **2025.5.24**: Released evaluation code for Math and updated the README to reflect these changes.



## Overview


Recent breakthroughs in reasoning-focused large language models (LLMs)—like OpenAI-o1, DeepSeek-R1, and Kimi-1.5—have heavily relied on **Reinforcement Learning with Verifiable Rewards (RLVR)**. RLVR replaces human annotations with automated rewards (such as verified math answers or passed code tests) to enable scalable self-improvement. While RLVR enhances behaviors like **self-reflection** and **iterative refinement**, a critical question remains in the pursuit of continually self-evolving reasoning abilities:

> **Does RLVR actually expand LLMs' reasoning capabilities, or does it merely optimize existing ones?**

To answer this, we evaluate models using the **pass@k** metric—where success requires only one correct solution among *k* attempts.


![Video Overview](./static/introvideo.gif)  


> *Video: pass@k curves of base models and their zero-RL-trained counterparts across multiple mathematical benchmarks. When k is small, RL-trained models outperform their base versions. However, as k increases to the tens or hundreds, base models consistently catch up with RL-trained models across all benchmarks and LLM families without exception. Eventually, base models surpass RL-trained models.*

Our conclusion:

1. **RL-trained models perform worse than base models in pass@k at large *k*.**  
  

2. **RL boosts sampling efficiency but reduces the reasoning capacity boundary.**  
  

3. **RLVR algorithms perform similarly and remain far from optimal.**  
   

4. **RLVR and distillation are fundamentally different.**  
 

## Mutiple Sampling in vLLM

In our experiments, we ultilize two key mechanisms of vLLM to ensure response diversity across different runs and within a single run's multiple samplings:

#### 1. Cross-Run Diversity via Seed Control

When initializing the `LLM` engine:

```python
LLM(seed=args.seed, ...)
```

vLLM uses the provided seed to initialize its internal random number generator. This means that different runs with different seeds will produce completely different response sequences, and changing the seed (e.g., `--seed 1` vs `--seed 2`) creates distinct sampling trajectories.

#### 2. Intra-Run Diversity

When performing multiple samplings in a single run (e.g., `--n_sampling 32`):

```python
SamplingParams(n=32, T=0.6, ...)  # Per-call sampling
```

vLLM automatically manages randomness by progressing the random state sequentially for each sampling call, and maintaining independent sampling trajectories even with identical parameters, thus ensuring diversity across samplings without manual seed adjustment.

## Evaluation

### Math

Enter `math` and read `README`.


## Acknowledgements

Our evaluation code is based on the following open-source projects:

- [SimpleRLZoo](https://github.com/hkust-nlp/simpleRL-reason)
- [Code-R1](https://github.com/ganler/code-r1)
- [EasyR1](https://github.com/hiyouga/EasyR1/)
- [DeepCoder](https://github.com/agentica-project/rllm)

We also extend our gratitude for the open-sourced checkpoints from:

- DAPO: [BytedTsinghua-SIA/DAPO-Qwen-32B](https://huggingface.co/BytedTsinghua-SIA/DAPO-Qwen-32B)
- Oat-Zero: [sail/Qwen2.5-Math-7B-Oat-Zero](https://huggingface.co/sail/Qwen2.5-Math-7B-Oat-Zero)



## Citation

If you use this work, please cite:

```bibtex
@article{yue2025limit-of-rlvr,
  title={Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?},
  author={Yue, Yang and Chen, Zhiqi and Lu, Rui and Zhao, Andrew and Wang, Zhaokai and Yue, Yang and Song, Shiji and Huang, Gao},
  journal={arXiv preprint arXiv:2504.13837},
  year={2025}
}
```




