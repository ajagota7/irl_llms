"""
SmolLM-optimized RLHF training (PPO) script.

Key changes vs. your original:
- QLoRA (4-bit NF4) + PEFT LoRA adapters
- FlashAttention-2 / SDPA, TF32, grad checkpointing, torch.compile
- Left padding + single batched generation per step
- Reward shaping fix (sigmoid -> centered reward)
- PagedAdamW32bit optimizer w/ cosine schedule + warmup
- Proper PPO knobs (adaptive KL, score norm, conservative clips)
- Cleaner, more robust W&B logging
- Save adapters-only checkpoints (tiny) while keeping push-to-hub flow

This keeps your Hydra config and utilities intact.
"""

import os
import time
import torch
import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

# Optimizers / schedulers
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# TRL / HF
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
)
from transformers import AutoTokenizer

# Quantization + PEFT
from transformers import BitsAndBytesConfig
from peft import LoraConfig

# Optional: bitsandbytes paged optimizer
try:
    from bitsandbytes.optim import PagedAdamW32bit  # type: ignore
    _HAS_BNB_OPT = True
except Exception:
    PagedAdamW32bit = None  # type: ignore
    _HAS_BNB_OPT = False

from huggingface_hub import HfApi

from rlhf_utilities import (
    build_dataset,
    collator,
    setup_wandb,
    load_reward_model,
    evaluate_toxicity,
    analyze_prompt_tracking,
    LengthSampler,
)

# ----------------------
# Helper toggles & speedups
# ----------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

torch.backends.cuda.matmul.allow_tf32 = True
try:
    from torch.backends.cuda import sdp_kernel
    # Prefer Flash or SDPA over math kernels
    sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
except Exception:
    pass


def _bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


# ----------------------
# Safe wrappers
# ----------------------

def safe_generate_batch(ppo_trainer: PPOTrainer, batch_input_ids, generation_kwargs):
    """Batched generation with fallback settings for stability."""
    try:
        # ppo_trainer.generate handles device placement
        return ppo_trainer.generate(batch_input_ids, **generation_kwargs)
    except RuntimeError as e:
        if "CUDA" in str(e) or "device-side assert" in str(e):
            print(f"[safe_generate_batch] CUDA error: {e}\nFalling back to deterministic decoding…")
            safe_kwargs = dict(generation_kwargs)
            safe_kwargs["do_sample"] = False
            safe_kwargs["top_k"] = None
            safe_kwargs["top_p"] = None
            safe_kwargs["num_beams"] = 1
            return ppo_trainer.generate(batch_input_ids, **safe_kwargs)
        raise


def safe_log_stats(ppo_trainer: PPOTrainer, stats: dict, batch: dict, rewards: list):
    """Cleans NaNs/Infs in stats and logs to TRL/Accelerate (and W&B if enabled)."""
    clean_stats = {}
    for k, v in stats.items():
        if isinstance(v, (int, float)):
            if np.isnan(v) or np.isinf(v):
                clean_stats[k] = 0.0
            else:
                clean_stats[k] = float(v)
        else:
            clean_stats[k] = v

    for key in ["ppo/advantages", "ppo/ratio", "ppo/policy_loss", "ppo/value_loss"]:
        if key in clean_stats and isinstance(clean_stats[key], list):
            clean_stats[key] = [
                float(x) for x in clean_stats[key]
                if isinstance(x, (int, float)) and not (np.isnan(x) or np.isinf(x))
            ] or [0.0]

    try:
        ppo_trainer.log_stats(clean_stats, batch, rewards)
    except Exception as e:
        print(f"[safe_log_stats] Logging error: {e}")
        try:
            minimal = {
                "rewards/mean": clean_stats.get("rewards/mean", 0.0),
                "train/step": clean_stats.get("train/step", 0),
            }
            ppo_trainer.accelerator.log(minimal)
        except Exception as e2:
            print(f"[safe_log_stats] Minimal logging failed: {e2}")


def safe_ppo_step(ppo_trainer: PPOTrainer, query_tensors, response_tensors, rewards):
    try:
        return ppo_trainer.step(query_tensors, response_tensors, rewards)
    except RuntimeError as e:
        if "CUDA" in str(e) or "device-side assert" in str(e):
            print(f"[safe_ppo_step] CUDA error during PPO step: {e}")
            return {"error": str(e)}
        raise


# ----------------------
# Main
# ----------------------

@hydra.main(config_path="configs", config_name="config", version_base=None)
def train_rlhf(cfg: DictConfig) -> float:
    """Main training function (SmolLM-optimized)."""
    cfg.now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Output dirs
    output_dir = os.path.join(os.getcwd(), f"outputs/{cfg.now}")
    eval_dir = os.path.join(output_dir, "evaluation")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Seeds
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)

    # W&B
    wandb_run = setup_wandb(cfg)
    if wandb_run:
        try:
            import wandb
            wandb_run.config.update(dict(now=cfg.now))
            wandb_run.define_metric("train/step")
            wandb_run.define_metric("*", step_metric="train/step", step_sync=True)
        except Exception:
            pass

    # Model/dataset inheritance from rlhf.* if not provided (keeps your logic)
    if cfg.model.name is None and hasattr(cfg.rlhf, 'model') and cfg.rlhf.model.name is not None:
        cfg.model.name = cfg.rlhf.model.name
    if cfg.dataset.name is None and hasattr(cfg.rlhf, 'dataset') and cfg.rlhf.dataset.name is not None:
        cfg.dataset.name = cfg.rlhf.dataset.name
    if cfg.model.reward_model is None and hasattr(cfg.rlhf, 'model') and cfg.rlhf.model.reward_model is not None:
        cfg.model.reward_model = cfg.rlhf.model.reward_model

    # Datasets + tokenizer
    print("Building dataset…")
    train_dataset, test_dataset, tokenizer = build_dataset(cfg)
    print(f"Train set: {len(train_dataset)} examples")
    print(f"Test  set: {len(test_dataset)} examples")

    # Tokenizer tweaks for decoder-only efficiency
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----------------------
    # Load policy with QLoRA + FA2/SDPA + grad checkpointing
    # ----------------------
    use_bf16 = _bf16_supported()

    # Try to use flash-attn-2; fallback to SDPA if not available
    attn_impl = getattr(cfg.model, "attn_implementation", None) or "flash_attention_2"

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # Load value-head model directly with PEFT + 4-bit
    try:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            cfg.model.name,
            quantization_config=bnb_cfg,
            peft_config=lora_cfg,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            attn_implementation=attn_impl,
            device_map="auto",
        )
    except Exception as e:
        print(f"[load] Falling back to SDPA/fp16 due to: {e}")
        try:
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                cfg.model.name,
                peft_config=lora_cfg,
                torch_dtype=torch.float16,
                attn_implementation="sdpa",
                device_map="auto",
            )
        except Exception as e2:
            print(f"[load] Falling back to full-precision, no quant: {e2}")
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                cfg.model.name,
                torch_dtype=torch.float16,
                device_map="auto",
            )

    # Enable grad checkpointing on the base (pretrained) module
    try:
        model.pretrained_model.gradient_checkpointing_enable()
    except Exception:
        pass

    # Keep value head in bf16/fp32 for PPO stability
    try:
        model.v_head = model.v_head.to(dtype=torch.bfloat16 if use_bf16 else torch.float32)
    except Exception:
        pass

    # Reference model (PEFT-aware)
    ref_model = create_reference_model(model)

    # Try torch.compile for extra speed on deep models
    try:
        model.pretrained_model = torch.compile(model.pretrained_model, mode="max-autotune")
    except Exception:
        pass

    # ----------------------
    # Optimizer + LR schedule
    # ----------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if _HAS_BNB_OPT:
        optimizer = PagedAdamW32bit(trainable_params, lr=cfg.model.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    else:
        optimizer = AdamW(trainable_params, lr=cfg.model.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    # Steps are defined like your original (loop is step-based, not true epochs)
    total_steps = cfg.training.num_train_epochs * (len(train_dataset) // max(1, cfg.model.batch_size) + 1)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(100, int(0.03 * total_steps)),
        num_training_steps=total_steps,
    )

    # ----------------------
    # PPO config
    # ----------------------
    ppo_params = {
        "model_name": cfg.model.name,
        "learning_rate": cfg.model.learning_rate,
        "log_with": "wandb" if wandb_run else None,
    }

    # Batch size compatibility logic (kept from your script)
    batch_size = cfg.model.batch_size
    mini_batch_size = cfg.model.mini_batch_size
    gradient_accumulation_steps = cfg.model.gradient_accumulation_steps

    if batch_size % (mini_batch_size * gradient_accumulation_steps) != 0:
        if batch_size >= gradient_accumulation_steps:
            new_mini_batch_size = batch_size // gradient_accumulation_steps
            print(f"Adjusting mini_batch_size {mini_batch_size} -> {new_mini_batch_size} for compatibility")
            mini_batch_size = new_mini_batch_size
        else:
            print(f"Adjusting gradient_accumulation_steps {gradient_accumulation_steps} -> 1 and mini_batch_size -> {batch_size}")
            gradient_accumulation_steps = 1
            mini_batch_size = batch_size

    ppo_params.update(
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # Add RLHF-specific knobs (safer defaults for small, deep models)
    if hasattr(cfg, 'rlhf') and hasattr(cfg.rlhf, 'model'):
        rlhf_model = cfg.rlhf.model
        for key in [
            'ppo_epochs', 'init_kl_coef', 'target', 'cliprange', 'cliprange_value',
            'vf_coef', 'adap_kl_ctrl', 'use_score_norm', 'ratio_threshold'
        ]:
            if hasattr(rlhf_model, key):
                ppo_params[key] = getattr(rlhf_model, key)

    ppo_params.setdefault("ppo_epochs", 2)
    ppo_params.setdefault("adap_kl_ctrl", True)
    ppo_params.setdefault("init_kl_coef", 0.02)
    ppo_params.setdefault("target", 6.0)
    ppo_params.setdefault("use_score_norm", True)
    ppo_params.setdefault("cliprange", 0.2)
    ppo_params.setdefault("cliprange_value", 0.2)

    ppo_config = PPOConfig(**ppo_params)

    # PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,  # kept as-is
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    # ----------------------
    # Reward model (optionally on a separate GPU)
    # ----------------------
    reward_device = ppo_trainer.accelerator.device
    if torch.cuda.device_count() > 1:
        reward_device = torch.device("cuda:1")
    print(f"Loading reward model on {reward_device}…")
    reward_model, reward_tokenizer = load_reward_model(
        cfg.model.reward_model,
        reward_device,
    )
    reward_model.eval()

    # Generation length sampler
    output_length_sampler = LengthSampler(
        cfg.model.generation.output_min_length,
        cfg.model.generation.output_max_length,
    )

    # ----------------------
    # Initial evaluation
    # ----------------------
    print("Performing initial evaluation…")
    initial_toxicity, _ = evaluate_toxicity(
        model=model,
        ppo_trainer=ppo_trainer,
        tokenizer=tokenizer,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        dataset=test_dataset,
        config=cfg,
        epoch="initial",
    )
    print(f"Initial average toxicity: {initial_toxicity:.4f}")

    if wandb_run:
        try:
            import wandb
            wandb_run.log({"eval/initial_toxicity": initial_toxicity, "train/step": 0})
            wandb_run.watch(model, log="gradients", log_freq=50)
        except Exception:
            pass

    # Reward stats across steps (your original structure)
    reward_stats = {
        'epoch': [],
        'raw_rewards_mean': [],
        'raw_rewards_std': [],
        'nan_inf_count': [],
    }

    print("Starting training loop…")
    training_start_time = time.time()

    for step, batch in enumerate(ppo_trainer.dataloader, start=1):
        if step > cfg.training.num_train_epochs:
            break

        step_start = time.time()
        query_tensors = batch["input_ids"]  # TRL provides already-prep'd tensors/lists

        # Ensure list-of-tensors for TRL.step
        if isinstance(query_tensors, torch.Tensor):
            query_list = [q for q in query_tensors]
        else:
            query_list = [q for q in query_tensors]

        # ---- Batched generation ----
        gen_len = int(output_length_sampler())
        generation_kwargs = {
            "min_length": getattr(cfg.model.generation, "min_length", 0),
            "top_k": cfg.model.generation.top_k,
            "top_p": cfg.model.generation.top_p,
            "do_sample": cfg.model.generation.do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "max_new_tokens": gen_len,
            "use_cache": True,
        }

        responses = safe_generate_batch(ppo_trainer, query_tensors, generation_kwargs)
        # responses: shape [B, total_len]; keep the generated tails of size gen_len
        response_tensors = []
        for resp in responses:
            resp = resp.squeeze()
            tail = resp[-gen_len:] if resp.size(0) >= gen_len else torch.nn.functional.pad(
                resp, (gen_len - resp.size(0), 0), value=tokenizer.pad_token_id
            )
            response_tensors.append(tail)

        # Decode batch responses
        texts = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

        # ---- Reward computation (toxicity sigmoid; centered reward) ----
        with torch.no_grad():
            tox_inputs = reward_tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to(reward_device)
            out = reward_model(**tox_inputs)
            logits = getattr(out, "logits", out)
            logits = logits.squeeze(-1)
            tox_prob = torch.sigmoid(logits).clamp(0.0, 1.0)

        # Centered reward is often stabler for PPO advantages
        centered_rewards = (0.5 - tox_prob).detach().cpu()
        rewards = [r for r in centered_rewards]

        # Reward stats
        rewards_tensor = centered_rewards
        raw_mean = rewards_tensor.mean().item()
        raw_std = rewards_tensor.std(unbiased=False).item()
        nan_inf_count = int(((~torch.isfinite(rewards_tensor)).sum()).item())

        reward_stats['epoch'].append(step)
        reward_stats['raw_rewards_mean'].append(raw_mean)
        reward_stats['raw_rewards_std'].append(raw_std)
        reward_stats['nan_inf_count'].append(nan_inf_count)

        # ---- PPO update ----
        stats = safe_ppo_step(ppo_trainer, query_list, response_tensors, rewards)

        # Tokens/sec logging
        tokens_this_step = len(response_tensors) * gen_len
        step_time = max(1e-6, time.time() - step_start)
        toks_per_sec = tokens_this_step / step_time

        # Augment stats
        stats.update({
            "rewards/mean": raw_mean,
            "rewards/std": raw_std,
            "rewards/nan_inf_count": nan_inf_count,
            "train/tokens_per_step": tokens_this_step,
            "train/tokens_per_second": toks_per_sec,
            "train/step": step,
        })

        # Log
        safe_log_stats(ppo_trainer, stats, batch, rewards)
        if wandb_run:
            try:
                wandb_run.log({
                    "train/step": step,
                    "train/tokens_per_second": toks_per_sec,
                    "rewards/mean": raw_mean,
                    "rewards/std": raw_std,
                })
            except Exception:
                pass

        # ---- Checkpointing (adapters only via TRL) ----
        if (step % cfg.training.save_freq) == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-step-{step}")
            print(f"Saving model checkpoint to {checkpoint_path}")
            if ppo_trainer.accelerator.is_main_process:
                ppo_trainer.save_pretrained(checkpoint_path)
                pd.DataFrame(reward_stats).to_csv(
                    os.path.join(checkpoint_path, "reward_stats.csv"), index=False
                )

        # ---- Optional: push checkpoint to Hub ----
        if (
            cfg.output.push_to_hub and cfg.output.push_checkpoints_to_hub and
            (step % cfg.output.checkpoint_push_freq) == 0
        ):
            try:
                if ppo_trainer.accelerator.is_main_process:
                    model_short_name = cfg.model.name.split('/')[-1]
                    repo_name = cfg.output.repository_name or f"{model_short_name}-detox"
                    repo_id = f"{cfg.output.organization}/{repo_name}" if cfg.output.organization else repo_name
                    checkpoint_repo_name = f"{repo_name}-checkpoint-step-{step}"
                    checkpoint_repo_id = f"{cfg.output.organization}/{checkpoint_repo_name}" if cfg.output.organization else checkpoint_repo_name

                    print(f"Pushing checkpoint to Hub: {checkpoint_repo_id}")

                    # Ensure directory has config + tokenizer
                    cp_path = os.path.join(checkpoint_dir, f"checkpoint-step-{step}")
                    os.makedirs(cp_path, exist_ok=True)
                    tokenizer.save_pretrained(cp_path)
                    with open(os.path.join(cp_path, "rlhf_config.yaml"), "w") as f:
                        f.write(OmegaConf.to_yaml(cfg))

                    api = HfApi()
                    if not api.repo_exists(repo_id=checkpoint_repo_id):
                        api.create_repo(repo_id=checkpoint_repo_id, private=cfg.output.private)
                    api.upload_folder(folder_path=cp_path, repo_id=checkpoint_repo_id, commit_message=f"Checkpoint step {step}")
                    print(f"Uploaded {checkpoint_repo_id}")
            except Exception as e:
                print(f"[Hub] Checkpoint push failed: {e}")

        # ---- Periodic evaluation ----
        if (step % cfg.training.eval_freq) == 0:
            print(f"\nEvaluating at step {step}…")
            avg_toxicity, _ = evaluate_toxicity(
                model=model,
                ppo_trainer=ppo_trainer,
                tokenizer=tokenizer,
                reward_model=reward_model,
                reward_tokenizer=reward_tokenizer,
                dataset=test_dataset,
                config=cfg,
                epoch=step,
            )
            print(f"Step {step}: Average toxicity = {avg_toxicity:.4f}")
            with open(os.path.join(eval_dir, "evaluation_results.txt"), "a") as f:
                f.write(f"Step {step}: Average toxicity = {avg_toxicity:.4f}\n")
            if wandb_run:
                try:
                    wandb_run.log({"eval/toxicity": avg_toxicity, "train/step": step})
                except Exception:
                    pass

    # ----------------------
    # Save final
    # ----------------------
    final_path = os.path.join(output_dir, "final-model")
    print(f"Saving final model to {final_path}")
    if ppo_trainer.accelerator.is_main_process:
        ppo_trainer.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        pd.DataFrame(reward_stats).to_csv(os.path.join(output_dir, "final_reward_stats.csv"), index=False)

        if cfg.output.push_to_hub:
            try:
                model_short_name = cfg.model.name.split('/')[-1]
                repo_name = cfg.output.repository_name or f"{model_short_name}-detox"
                repo_id = f"{cfg.output.organization}/{repo_name}" if cfg.output.organization else repo_name
                print(f"Pushing final model to Hub: {repo_id}")
                with open(os.path.join(final_path, "rlhf_config.yaml"), "w") as f:
                    f.write(OmegaConf.to_yaml(cfg))
                api = HfApi()
                if not api.repo_exists(repo_id=repo_id):
                    api.create_repo(repo_id=repo_id, private=False)
                api.upload_folder(folder_path=final_path, repo_id=repo_id, commit_message="Final model after RLHF training")
                print(f"Successfully pushed model to {repo_id}")
            except Exception as e:
                print(f"[Hub] Final push failed: {e}")

    # Final evaluation
    final_toxicity, _ = evaluate_toxicity(
        model=model,
        ppo_trainer=ppo_trainer,
        tokenizer=tokenizer,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        dataset=test_dataset,
        config=cfg,
        epoch="final",
    )
    print(f"Final evaluation: Average toxicity = {final_toxicity:.4f}")

    total_time = time.time() - training_start_time
    h, rem = divmod(total_time, 3600)
    m, s = divmod(rem, 60)
    print(f"Total training time: {int(h)}h {int(m)}m {int(s)}s")
    print(f"Training complete! Models and results saved to: {output_dir}")

    if wandb_run:
        try:
            wandb_run.log({"eval/final_toxicity": final_toxicity, "train/total_seconds": total_time, "train/step": step})
        except Exception:
            pass

    return float(final_toxicity)


if __name__ == "__main__":
    train_rlhf()
