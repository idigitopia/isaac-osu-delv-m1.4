# Command Library

This document contains commonly used commands for training and evaluation.



## Saw-Go2
<hr style="border-top: 3px dotted #bbb;">

### Training
<hr style="border-top: 3px dotted #bbb;">

```bash
WANDB_USERNAME=dacmdp python drail_extensions/drail_extensions/core/scripts/train_v2.py \
    --task SaW-Go2-Flat-v0 \
    --headless \
    --logger wandb \
    --max_iterations 1000 \
    --num_envs 4096
```

**Description:**
- Trains the SaW-Go2 task with a flat reward function
- Uses WandB logging. Make sure to set `WANDB_USERNAME` environment variable to your wandb entity name (not your wandb username).
- Runs in headless mode
- Uses 4096 parallel environments


<hr style="border-top: 3px dotted #bbb;">

### Evaluation
<hr style="border-top: 3px dotted #bbb;">

**Variant 1: Play from Local Checkpoint (Non Interactive Mode)**
```bash
WANDB_USERNAME=dacmdp python drail_extensions/drail_extensions/core/scripts/play_v2.py \
    --task SaW-Go2-Flat-v0 \
    --num_envs 32 \
    --raw_checkpoint drail_extensions/drail_extensions/core/data/pretrained_policy/bikram/bikram-AMP-Flat-Go2-Play-v0/team-osu/isaaclab/66sjhj1j/model_9999.pt
```

**Variant 2: Play from Local Checkpoint (Interactive Mode)**
```bash
WANDB_USERNAME=dacmdp python drail_extensions/drail_extensions/core/scripts/play_v2.py \
    --task SaW-Go2-Flat-v0 \
    --num_envs 32 \
    --interactive \
    --interactive_mode keyboard
    --raw_checkpoint drail_extensions/drail_extensions/core/data/pretrained_policy/bikram/bikram-AMP-Flat-Go2-Play-v0/team-osu/isaaclab/66sjhj1j/model_9999.pt
```

**Variant 3: Play from WandB Checkpoint (Interactive Mode)**
```bash
WANDB_USERNAME=dacmdp python drail_extensions/drail_extensions/core/scripts/play_v2.py \
    --task SaW-Go2-Flat-v0 \
    --num_envs 32 \
    --interactive \
    --interactive_mode keyboard
    --wandb_checkpoint dacmdp/isaaclab/6o3afjwc/model_950.pt
```
**Description:**
- Evaluates the SaW-Go2 task
- Uses WandB logging
- Runs in headless mode
- Uses 32 parallel environments

----

## SaW-Go2 with AMP and Penalties and Feet Air Time

<hr style="border-top: 3px dotted #bbb;">

### Training
<hr style="border-top: 3px dotted #bbb;">

```bash
WANDB_USERNAME=dacmdp python drail_extensions/drail_extensions/core/scripts/train_v2.py \
    --task SaW-Go2-Flat-AMP-Penalty-FeetAirTime-v0 \
    --headless \
    --logger wandb \
    --max_iterations 10000 \
    --num_envs 4096 \
    agent.discriminator_grad_penalty=40.0 \
    agent.discriminator_l2_reg=2.0 \
    agent.discriminator.actor_hidden_dims=[1024,256] \
    agent.discriminator.critic_hidden_dims=[1,1]
```


**Description:**
- Trains the SaW-Go2 task with AMP (Adversarial Motion Priors) and feet air time penalties
- Uses WandB logging
- Runs in headless mode
- Uses 4096 parallel environments
- Configures discriminator parameters including gradient penalty and network architecture

<hr style="border-top: 3px dotted #bbb;">

### Evaluation
<hr style="border-top: 3px dotted #bbb;">

```bash
WANDB_USERNAME=dacmdp python drail_extensions/drail_extensions/core/scripts/play_v2.py \
    --task SaW-Go2-Flat-AMP-Penalty-FeetAirTime-v0 \
    --num_envs 32 \
    --interactive \
    --interactive_mode keyboard \
    --wandb_checkpoint dacmdp/isaaclab/6o3afjwc/model_950.pt
```


## Utility Commands

[To be added]

## Notes
- Add `--headless` flag to run without visualization
- Set `WANDB_USERNAME` environment variable for WandB logging
- Adjust `num_envs` based on available computational resources
