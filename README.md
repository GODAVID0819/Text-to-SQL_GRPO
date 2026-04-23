# GRPO for Text-to-SQL with Verifiable Rewards

This project extends a supervised T5 text-to-SQL pipeline into a reinforcement learning setup using **GRPO** (Group Relative Policy Optimization). The main goal is to improve SQL generation quality by optimizing directly for **execution-based correctness**, rather than only token-level supervised loss.

## Motivation

In the current supervised setup, the model is trained with teacher forcing using cross-entropy against gold SQL queries. While this helps the model learn the general structure of SQL, it does not directly optimize for what actually matters at inference time: whether the generated query executes correctly and returns the right result.

Text-to-SQL is a strong fit for reinforcement learning with verifiable rewards because generated SQL can be executed against a database and compared to the gold answer. This makes it possible to train the model using outcome-based feedback rather than relying only on next-token prediction.

## Current Baseline

The starting point is a T5-based text-to-SQL system:

- Input: natural language question
- Output: SQL query
- Training: supervised fine-tuning with cross-entropy
- Evaluation:
  - SQL exact match
  - record exact match
  - record F1
  - SQL execution error rate

The existing training loop uses gold decoder inputs and minimizes standard sequence loss. The evaluation pipeline already executes generated SQL and compares returned records against ground truth, which provides the core ingredients needed for RL.

## Why GRPO

GRPO is a natural choice for this setting because it does not require a learned reward model. Instead, it uses **verifiable rewards** computed directly from generated outputs.

For each input question:

1. Sample multiple candidate SQL queries from the current policy
2. Execute each query against the database
3. Assign a reward to each candidate
4. Normalize rewards within the group
5. Increase the probability of better candidates and decrease the probability of worse ones

This allows training to focus on relative quality among multiple sampled outputs for the same prompt.

## Training Strategy

The project follows a two-stage pipeline:

### 1. Supervised Fine-Tuning (SFT)

First, train T5 in the standard way on paired natural language and SQL data. This stage is essential because RL from scratch would be too unstable: most sampled queries would be invalid, leading to sparse or uninformative rewards.

### 2. GRPO Post-Training

After SFT, continue training with GRPO:

- start from the best SFT checkpoint
- sample multiple SQL candidates per prompt
- compute execution-based rewards
- normalize rewards within each group
- apply a policy optimization objective
- optionally add KL regularization to keep the policy close to the SFT model

## Reward Design

Several reward choices are possible.

### Binary execution reward
- `1` if the generated query returns exactly the same records as the gold query
- `0` otherwise

### Record F1 reward
Use the record-level F1 score between predicted and gold query results. This provides a smoother signal than binary correctness and is likely better for early experiments.

## Group Structure in GRPO

For each natural language input `x`, generate a group of candidate outputs:

- `y1, y2, ..., yG`

Each candidate gets a reward:

- `r1, r2, ..., rG`

These rewards are normalized within the group to produce relative advantages. A simple formulation is:

- subtract the group mean reward
- divide by the group standard deviation

This ensures the model learns from relative ranking within the sampled set for each prompt.

## Policy Update Intuition

The model should:

- assign higher probability to candidates with above-average reward
- assign lower probability to candidates with below-average reward

Instead of optimizing against gold tokens directly, the RL objective uses the log-probability of sampled SQL sequences weighted by their relative advantage.

In practice, this means:
- sample candidate SQL queries
- compute rewards
- re-score sampled sequences under the current model
- compute a sequence-level policy loss
- backpropagate through that objective

## Role of the Reference Model

To stabilize RL training, a frozen copy of the SFT model can be used as a **reference policy**. A KL penalty discourages the updated policy from drifting too far away from the original supervised model.

This is important because pure RL can otherwise:
- overfit reward quirks
- collapse output quality
- drift toward degenerate SQL patterns

So the recommended setup is:
- trainable current policy
- frozen SFT reference model
- small KL regularization term

## Required Code Changes

The current codebase already contains execution and metric computation utilities, but it is organized for offline evaluation. To support GRPO training, the following changes are needed.

### Data loading
The training dataloader must expose enough information for reward computation, such as:
- encoder inputs
- gold SQL text
- example indices
- optionally cached gold execution results

### Reward computation
A training-time reward function is needed to:
- execute sampled SQL queries
- compare their returned records with the gold result
- assign one scalar reward per sampled candidate

### RL training loop
The supervised `train_epoch` must be replaced or supplemented with an RL version that:
- samples multiple completions per prompt
- computes rewards
- computes sequence log-probabilities
- applies the GRPO objective
- optionally applies KL regularization

### Decoding behavior
Training-time decoding should use **sampling**, not beam search, because GRPO requires multiple diverse candidates. Beam search remains useful for dev/test evaluation.

## Recommended First Implementation

A practical first version would look like this:

- initialize from the best SFT checkpoint
- freeze a copy as the reference model
- sample 4 SQL candidates per prompt
- use stochastic decoding during rollout
- use record F1 as the reward
- assign reward 0 to invalid SQL
- normalize rewards within each prompt group
- optimize a GRPO-style policy loss
- add a small KL penalty to the reference model

This setup should give a stable and interpretable first RL baseline.

## Why This Approach Matters

This project shifts text-to-SQL training closer to the true task objective. Instead of only asking whether the model predicts the exact gold sequence, it asks whether the generated SQL actually works.

That makes the system more aligned with practical correctness:

- semantically equivalent SQL can still be rewarded
- partially correct outputs can receive partial credit
- execution failures can be explicitly penalized
- training optimizes for result quality, not just token imitation

## Next Steps

Planned implementation order:

1. extend the dataset and batch format for RL training
2. build a training-time reward function
3. add stochastic multi-sample rollout
4. compute sequence log-probabilities for sampled SQL
5. implement the GRPO loss
6. add KL regularization to a frozen reference model
7. tune group size, temperature, and reward shaping

## Summary

This project explores how to turn a supervised T5 text-to-SQL system into a reinforcement learning pipeline using GRPO and verifiable rewards. Because SQL outputs can be executed and checked directly, the task is well suited to RL without a learned reward model. The overall objective is to improve execution correctness by training on sampled outputs and relative outcome-based feedback, while keeping the model anchored to a strong supervised baseline.
