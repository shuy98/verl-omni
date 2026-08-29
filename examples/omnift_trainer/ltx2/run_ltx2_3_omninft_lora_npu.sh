#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/omninft/vggsound/verl_omni}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/test.parquet}

if ! prerequisites=$(python3 - <<'PY' 2>&1
from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase
from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_loss_fn

missing = []
try:
    DiffusionModelBase.get_class_by_name("LTX2Pipeline", "omni_nft")
except NotImplementedError:
    missing.append("training adapter (stage 3)")
if VllmOmniPipelineBase.get_class("LTX2Pipeline", "omni_nft") is None:
    missing.append("rollout adapter (stage 2)")
try:
    get_diffusion_loss_fn("omni_nft")
except ValueError:
    missing.append("omni_nft loss (stage 5)")
if missing:
    print("OmniNFT training prerequisites missing: " + ", ".join(missing))
    raise SystemExit(2)
PY
); then
    printf '%s\n' "$prerequisites" >&2
    exit 2
fi

if [[ -z ${VAL_FILE:-} ]]; then
    echo "VAL_FILE must point to a separate validation JSONL for full training." >&2
    exit 2
fi

export WANDB_MODE=${WANDB_MODE:-offline}
ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit}
source "$ASCEND_HOME_PATH/set_env.sh"
source "$ASCEND_HOME_PATH/../nnal/atb/set_env.sh"

MODEL_PATH=${MODEL_PATH:-/home/y00609984/huggingface_models/LTX-2.3-Diffusers}
NUM_GPUS=${NUM_GPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-8}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
ltx_lora_targets="['attn1.to_q','attn1.to_k','attn1.to_v','attn1.to_out.0','attn2.to_q','attn2.to_k','attn2.to_v','attn2.to_out.0','audio_attn1.to_q','audio_attn1.to_k','audio_attn1.to_v','audio_attn1.to_out.0','audio_attn2.to_q','audio_attn2.to_k','audio_attn2.to_v','audio_attn2.to_out.0','audio_to_video_attn.to_q','audio_to_video_attn.to_k','audio_to_video_attn.to_v','audio_to_video_attn.to_out.0','video_to_audio_attn.to_q','video_to_audio_attn.to_k','video_to_audio_attn.to_v','video_to_audio_attn.to_out.0','ff.net.0.proj','ff.net.2','audio_ff.net.0.proj','audio_ff.net.2']"

script_path=$(readlink -f "$0")
script_name=$(basename "$script_path" .sh)
repo_root=$(dirname "$script_path")
while [[ "$repo_root" != "/" && ! -f "$repo_root/LICENSE" ]]; do
    repo_root=$(dirname "$repo_root")
done
if [[ ! -f "$repo_root/LICENSE" ]]; then
    echo "Unable to locate repo root from $script_path: no LICENSE found" >&2
    exit 1
fi

output_dir=${OUTPUT_DIR:-$repo_root/outputs/$script_name}
checkpoint_dir=$output_dir/checkpoints
run_timestamp=$(date +"%Y%m%d_%H%M")
log_file=$output_dir/logs/$run_timestamp/${NODE_RANK:-0}.log
rollout_data_dir=$output_dir/logs/$run_timestamp/rollout_videos
mkdir -p "$checkpoint_dir" "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1

python3 -m verl_omni.trainer.main_diffusion \
    trainer.device=npu \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.return_multi_modal_inputs=False \
    data.train_batch_size=1 \
    data.val_max_samples=1 \
    data.max_prompt_length=1024 \
    data.truncation=error \
    data.seed=42 \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.paired_preference=false \
    algorithm.timestep_fraction=1.0 \
    algorithm.old_policy_decay_schedule=delayed_linear_to_0_999 \
    algorithm.old_policy_update_interval=1 \
    algorithm.adv_mode=continuous \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.algorithm=omni_nft \
    actor_rollout_ref.model.model_type=omni_nft_model \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules="$ltx_lora_targets" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['transformer_blocks.']" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[LTX2VideoTransformerBlock]' \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=1e-4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=omni_nft \
    actor_rollout_ref.actor.diffusion_loss.video_weight=1.0 \
    actor_rollout_ref.actor.diffusion_loss.audio_weight=1.0 \
    actor_rollout_ref.actor.diffusion_loss.video_ref_kl_coef=0.0 \
    actor_rollout_ref.actor.diffusion_loss.audio_ref_kl_coef=0.0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.agent.default_agent_loop=ltx2_omni_nft_single_turn_agent \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=384 \
    actor_rollout_ref.rollout.pipeline.num_frames=81 \
    actor_rollout_ref.rollout.pipeline.frame_rate=24.0 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=24 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=1024 \
    +actor_rollout_ref.rollout.pipeline.output_type=pt \
    actor_rollout_ref.rollout.val_kwargs.pipeline.height=256 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.width=384 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_frames=81 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.frame_rate=24.0 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.guidance_scale=4.0 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.output_type=pt \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    reward.reward_model.enable=False \
    trainer.logger='["console"]' \
    trainer.project_name=omni_nft \
    trainer.experiment_name=ltx2_3_omninft_lora_npu \
    trainer.default_local_dir=$checkpoint_dir \
    trainer.validation_data_dir=$rollout_data_dir \
    trainer.validation_data_max_samples=4 \
    trainer.resume_mode=disable \
    trainer.log_val_generations=0 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    "$@"
