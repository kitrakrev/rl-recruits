"""
Training package for the Staffing Agency RL environment.

Modules
-------
prompts     SYSTEM_PROMPT, TOOLS list, parse_tool_call()
rollout     rollout_full_episode() — live env interaction, returns per-step rewards
policies    Heuristic policies used by the dry-run simulator
metrics     _plot_reward_curves(), _save_metrics()
dry_run     dry_run_simulate() — validates the full HTTP stack without a GPU
reinforce   train_grpo() — REINFORCE-style policy gradient training loop
"""
