import os
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from t5_utils import (
    initialize_model,
    initialize_reference_model,
    initialize_optimizer_and_scheduler,
    save_model,
    load_model_from_checkpoint,
)
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records, compute_records, compute_record_F1

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0
EPS = 1e-8


def get_args():
    parser = argparse.ArgumentParser(description='T5 SFT / GRPO training loop')

    parser.add_argument('--finetune', action='store_true', help='Whether to finetune T5 or not')
    parser.add_argument('--train_mode', type=str, default='sft', choices=['sft', 'grpo'],
                        help='Use supervised fine-tuning or GRPO RL training')

    parser.add_argument('--optimizer_type', type=str, default='AdamW', choices=['AdamW'],
                        help='What optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['none', 'cosine', 'linear'],
                        help='Whether to use a LR scheduler and what type to use if so')
    parser.add_argument('--num_warmup_epochs', type=int, default=1,
                        help='How many epochs to warm up the learning rate for if using a scheduler')
    parser.add_argument('--max_n_epochs', type=int, default=12,
                        help='How many epochs to train the model for')
    parser.add_argument('--patience_epochs', type=int, default=4,
                        help='If validation performance stops improving, how many epochs should we wait before stopping?')

    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help='How should we name this experiment?')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='How many new tokens can be generated?')
    parser.add_argument('--num_beams', type=int, default=4,
                        help='Number of beams used during eval/test decoding')

    parser.add_argument('--init_checkpoint', type=str, default=None,
                        help='Path to a HuggingFace checkpoint to initialize training from, e.g. checkpoints/.../best')
    parser.add_argument('--ref_checkpoint', type=str, default=None,
                        help='Optional frozen reference checkpoint for GRPO KL. Defaults to --init_checkpoint.')
    parser.add_argument('--resume_from_experiment', type=str, default=None,
                        help='Load checkpoints/<ft_or_scr>_experiments/<name>/<best_or_last> before training')
    parser.add_argument('--resume_best', action='store_true',
                        help='When using --resume_from_experiment, load best instead of last')

    parser.add_argument('--grpo_group_size', type=int, default=4,
                        help='Number of sampled completions per prompt')
    parser.add_argument('--grpo_temperature', type=float, default=1.0)
    parser.add_argument('--grpo_top_p', type=float, default=0.95)
    parser.add_argument('--grpo_beta', type=float, default=0.02,
                        help='KL penalty weight against frozen reference model')
    parser.add_argument('--grpo_clip_eps', type=float, default=0.2,
                        help='PPO-style clipping epsilon for ratio objective')
    parser.add_argument('--grpo_reward_type', type=str, default='binary', choices=['f1', 'binary'],
                        help='f1 gives partial credit using record F1; binary gives 1 only for exact record match')
    parser.add_argument('--invalid_sql_reward', type=float, default=-1.0,
                        help='Reward assigned to generated SQL that does not execute')
    parser.add_argument('--grpo_min_reward_std', type=float, default=1e-6,
                        help='Minimum within-group reward std. Groups below this contribute almost no learning signal')
    parser.add_argument('--grpo_logprob_microbatch_size', type=int, default=1,
                    help='Microbatch size for GRPO log-prob computation to avoid OOM')
    parser.add_argument('--fp16', action='store_true',
                        help='Use CUDA mixed precision during GRPO/SFT training to reduce memory')

    args = parser.parse_args()
    return args


def checkpoint_dir_for(args):
    model_type = 'ft' if args.finetune else 'scr'
    return os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)


def load_resume_checkpoint_if_needed(args):
    if not args.resume_from_experiment:
        return None
    model_type = 'ft' if args.finetune else 'scr'
    subdir = 'best' if args.resume_best else 'last'
    return os.path.join('checkpoints', f'{model_type}_experiments', args.resume_from_experiment, subdir)


def train(args, model, train_loader, dev_loader, optimizer, scheduler, ref_model=None):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = checkpoint_dir_for(args)
    gt_sql_path = os.path.join('data/dev.sql')
    gt_record_path = os.path.join('records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_dev.pkl')

    for epoch in range(args.max_n_epochs):
        if args.train_mode == 'grpo':
            train_stats = train_epoch_grpo(args, model, ref_model, train_loader, optimizer, scheduler)
            tr_loss = train_stats['loss']
            print(
                f"Epoch {epoch}: GRPO loss={train_stats['loss']:.6f}, "
                f"reward={train_stats['reward_mean']:.4f}, "
                f"valid_sql={train_stats['valid_sql_rate']:.4f}, "
                f"kl={train_stats['kl_mean']:.6f}"
            )
        else:
            tr_loss = train_epoch_sft(args, model, train_loader, optimizer, scheduler)
            train_stats = {'loss': tr_loss}
            print(f'Epoch {epoch}: Average train loss was {tr_loss}')

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
        )
        print(f'Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}')
        print(f'Epoch {epoch}: {error_rate * 100:.2f}% of the generated outputs led to SQL errors')

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break


def train_epoch_sft(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and DEVICE.type == 'cuda')

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        with torch.cuda.amp.autocast(enabled=args.fp16 and DEVICE.type == 'cuda'):
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']

            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)


def strip_decoder_start(generated_ids, decoder_start_token_id):
    if generated_ids.size(1) > 0 and torch.all(generated_ids[:, 0] == decoder_start_token_id):
        return generated_ids[:, 1:]
    return generated_ids


def sequence_logprobs(model, encoder_input, encoder_mask, target_ids):
    #Return per-sequence sum log p(target | input) and a non-pad token mask.

    labels = target_ids.clone()
    labels[labels == PAD_IDX] = -100
    outputs = model(input_ids=encoder_input, attention_mask=encoder_mask, labels=labels)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)

    gather_labels = labels.clone()
    gather_labels[gather_labels == -100] = PAD_IDX
    token_log_probs = log_probs.gather(dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)
    token_mask = (labels != -100).float()
    seq_log_probs = (token_log_probs * token_mask).sum(dim=-1)
    return seq_log_probs, token_mask

def sequence_logprobs_microbatched(model, encoder_input, encoder_mask, target_ids, microbatch_size):
    all_logprobs = []
    all_token_masks = []

    for start in range(0, target_ids.size(0), microbatch_size):
        end = start + microbatch_size

        mb_logprobs, mb_token_mask = sequence_logprobs(
            model,
            encoder_input[start:end],
            encoder_mask[start:end],
            target_ids[start:end],
        )

        all_logprobs.append(mb_logprobs)
        all_token_masks.append(mb_token_mask)

    return torch.cat(all_logprobs, dim=0), torch.cat(all_token_masks, dim=0)


def decode_gold_sql_from_targets(tokenizer, decoder_targets):
    labels = decoder_targets.detach().cpu().clone()
    labels[labels == PAD_IDX] = tokenizer.pad_token_id
    return tokenizer.batch_decode(labels, skip_special_tokens=True)


def records_f1_one(gt_rec, pred_rec):
    return float(compute_record_F1([gt_rec], [pred_rec]))


def compute_grpo_rewards(args, tokenizer, generated_ids, decoder_targets, group_size):
    # Execute sampled SQL and compare each sample with its prompt's gold SQL.
    # Returns rewards with shape [batch, group_size], plus valid-SQL rate.
    batch_size = decoder_targets.size(0)
    decoded_sql = tokenizer.batch_decode(generated_ids.detach().cpu(), skip_special_tokens=True)
    gold_sql = decode_gold_sql_from_targets(tokenizer, decoder_targets)

    pred_records, pred_errors = compute_records(decoded_sql)
    gold_records, _ = compute_records(gold_sql)

    rewards = []
    valid_count = 0
    for i in range(batch_size):
        gt_rec = gold_records[i]
        group_rewards = []
        for j in range(group_size):
            k = i * group_size + j
            if pred_errors[k] != '':
                reward = args.invalid_sql_reward
            elif args.grpo_reward_type == 'binary':
                reward = 1.0 if set(pred_records[k]) == set(gt_rec) else 0.0
            else:
                reward = records_f1_one(gt_rec, pred_records[k])
            if pred_errors[k] == '':
                valid_count += 1
            group_rewards.append(reward)
        rewards.append(group_rewards)

    rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    valid_sql_rate = valid_count / max(len(decoded_sql), 1)
    return rewards, valid_sql_rate


def train_epoch_grpo(args, model, ref_model, train_loader, optimizer, scheduler):
    model.train()
    ref_model.eval()

    total_loss = 0.0
    total_groups = 0
    reward_sum = 0.0
    reward_count = 0
    valid_sum = 0.0
    kl_sum = 0.0
    kl_count = 0

    tokenizer = train_loader.dataset.tokenizer
    decoder_start_token_id = model.config.decoder_start_token_id
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and DEVICE.type == 'cuda')

    for encoder_input, encoder_mask, _, decoder_targets, initial_decoder_inputs in tqdm(train_loader):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)
        initial_decoder_inputs = initial_decoder_inputs.to(DEVICE)

        batch_size = encoder_input.size(0)
        group_size = args.grpo_group_size

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=initial_decoder_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.grpo_temperature,
                top_p=args.grpo_top_p,
                num_return_sequences=group_size,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            sampled_target_ids = strip_decoder_start(generated_ids, decoder_start_token_id)
            old_encoder_input = encoder_input.repeat_interleave(group_size, dim=0)
            old_encoder_mask = encoder_mask.repeat_interleave(group_size, dim=0)
            old_logprobs, token_mask = sequence_logprobs_microbatched(
                model,
                old_encoder_input,
                old_encoder_mask,
                sampled_target_ids,
                args.grpo_logprob_microbatch_size,
            )
            ref_logprobs, _ = sequence_logprobs_microbatched(
                ref_model,
                old_encoder_input,
                old_encoder_mask,
                sampled_target_ids,
                args.grpo_logprob_microbatch_size,
            )

            rewards, valid_sql_rate = compute_grpo_rewards(
                args, tokenizer, sampled_target_ids, decoder_targets, group_size
            )
            group_mean = rewards.mean(dim=1, keepdim=True)
            group_std = rewards.std(dim=1, keepdim=True, unbiased=False)
            advantages = (rewards - group_mean) / torch.clamp(group_std, min=args.grpo_min_reward_std)
            advantages = advantages.reshape(-1)

        # Release any cached blocks left from generation / no-grad scoring before the gradient pass.
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16 and DEVICE.type == 'cuda'):
            new_logprobs, token_mask = sequence_logprobs_microbatched(
                model,
                old_encoder_input,
                old_encoder_mask,
                sampled_target_ids,
                args.grpo_logprob_microbatch_size,
            )
            token_counts = token_mask.sum(dim=-1).clamp(min=1.0)

            # Sequence-level PPO-style ratio with clipping.
            log_ratio = (new_logprobs - old_logprobs).clamp(min=-20.0, max=20.0)
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1.0 - args.grpo_clip_eps, 1.0 + args.grpo_clip_eps)
            policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()

            # Per-token KL on the sampled completions against reference model.
            seq_kl = (new_logprobs - ref_logprobs) / token_counts
            kl_loss = seq_kl.mean()
            loss = policy_loss + args.grpo_beta * kl_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * batch_size
        total_groups += batch_size
        reward_sum += rewards.sum().item()
        reward_count += rewards.numel()
        valid_sum += valid_sql_rate * batch_size
        kl_sum += seq_kl.detach().sum().item()
        kl_count += seq_kl.numel()

    return {
        'loss': total_loss / max(total_groups, 1),
        'reward_mean': reward_sum / max(reward_count, 1),
        'valid_sql_rate': valid_sum / max(total_groups, 1),
        'kl_mean': kl_sum / max(kl_count, 1),
    }


def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_tokens = 0
    generated_sql = []
    tokenizer = dev_loader.dataset.tokenizer

    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder_inputs in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            initial_decoder_inputs = initial_decoder_inputs.to(DEVICE)

            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']

            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=initial_decoder_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=args.num_beams,
            )
            generated_sql.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    save_queries_and_records(generated_sql, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    error_rate = sum(1 for msg in error_msgs if msg != '') / len(error_msgs)
    return total_loss / max(total_tokens, 1), record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    model.eval()
    tokenizer = test_loader.dataset.tokenizer
    generated_sql = []

    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)

    with torch.no_grad():
        for encoder_input, encoder_mask, initial_decoder_inputs in tqdm(test_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            initial_decoder_inputs = initial_decoder_inputs.to(DEVICE)

            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=initial_decoder_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=args.num_beams,
            )
            generated_sql.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    save_queries_and_records(generated_sql, model_sql_path, model_record_path)


def main():
    args = get_args()

    resume_checkpoint = load_resume_checkpoint_if_needed(args)
    if resume_checkpoint is not None:
        args.init_checkpoint = resume_checkpoint

    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    ref_model = initialize_reference_model(args, model) if args.train_mode == 'grpo' else None
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    train(args, model, train_loader, dev_loader, optimizer, scheduler, ref_model=ref_model)

    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join('data/dev.sql')
    gt_record_path = os.path.join('records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    print(f'Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}')
    print(f'Dev set results: {dev_error_rate * 100:.2f}% of the generated outputs led to SQL errors')

    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)


if __name__ == '__main__':
    main()

