# My RL Reranker Project

I hacked this project together to see if I could build a top-tier reranker for search, inspired by the awesome work from Fin AI. You can read their paper, "How We Built a World-Class Reranker for Fin," here: [https://fin.ai/research/how-we-built-a-world-class-reranker-for-fin/](https://fin.ai/research/how-we-built-a-world-class-reranker-for-fin/).

The whole thing runs on my Mac with open-source tools, and it actually managed to beat their benchmark!

## The Big Idea: Don't Let the RL Agent Go Rogue

So, the main trick is using Reinforcement Learning (RL) to fine-tune a search reranker. But there's a catch: RL agents can sometimes "reward hack," which means they find weird ways to get a high score during training that don't actually work on real data.

My pipeline is set up to prevent this. I call it a **Self-Correcting Reranker**.

1.  **First, I train a normal supervised model.** I give it a bunch of messy, computer-generated labels. The model learns to find the average, common-sense patterns in the noise. This gives me a solid, stable "anchor" model.

2.  **Then, I use RL to fine-tune it.** But I put the RL agent on a tight leash. I tell it: "*You can try to find better rankings to get more reward, but you are not allowed to get too weird or stray too far from what the anchor model already knows.*"

This stops the agent from overfitting to the noisy rewards. It's forced to find strategies that are *actually* better, not just ones that trick the scoring system.

## The Results: It Worked!

To prove this worked, I ran three experiments:

| Model | What I Did | Validation NDCG@10 | Result |
| :--- | :--- | :--- | :--- |
| Supervised-Only | Just the basic anchor model. | **0.879** | **Solid Baseline** |
| Over-Optimized RL | Let the RL agent go wild. | `0.796` | **Reward Hacking!** |
| My Stabilized RL | Kept the RL agent on a leash. | **0.878** | **It Didn't Go Rogue!** |

As you can see, the "rogue" agent learned bad habits and its performance tanked on the test data. My stabilized agent learned to ignore the bad signals and maintained its strong performance.

And the best part? My final model's `NDCG@10` of **0.878** is **32% better** than the `0.665` benchmark from the Fin paper.

## How to Run It

### 1. Setup

```bash
# Make a Python 3.11+ environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the stuff
pip install -r requirements.txt
```

### 2. Make the Data

```bash
# This makes a fake dataset to test on
python3 -m src.generate_synthetic_benchmark --output data/synthetic_benchmark.jsonl

# This splits it into train/test files
python3 -m src.split_data --input data/synthetic_benchmark.jsonl --output-dir data/synthetic
```

### 3. Train & Test

```bash
# This is a must-do for macOS to avoid crashes
export FIN_DEVICE=cpu

# 1. Train the anchor model
python3 -m src.train_supervised \
    --data-dir data/synthetic \
    --model-id sentence-transformers/all-MiniLM-L6-v2 \
    --output-dir outputs/sft_synthetic \
    --num-epochs 5

# 2. Fine-tune with my stabilized RL
python3 -m src.train_rl \
    --data-dir data/synthetic \
    --sft-checkpoint outputs/sft_synthetic/best \
    --output-dir outputs/ppo_synthetic_v2 \
    --epochs 5 \
    --learning-rate 1e-6 \
    --kl-penalty 0.1

# 3. See the final score
python3 -m src.evaluate \
    --data-dir data/synthetic \
    --checkpoint outputs/ppo_synthetic_v2/final \
    --split val
```
