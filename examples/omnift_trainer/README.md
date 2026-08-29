# OmniNFT training

This directory contains the staged LTX-2.3 OmniNFT recipe.

## Prepare data

The native OmniNFT VGGSound metadata stores a joint generation prompt plus
separate video and audio reward prompts in JSONL records. Convert the training
and validation metadata to the standard RLHF parquet schema before launching:

```bash
python3 examples/omnift_trainer/ltx2/prepare_data.py \
  --train_file /path/to/train_metadata_20k.jsonl \
  --val_file /path/to/test_metadata.jsonl \
  --output_dir ./data/omninft/vggsound/verl_omni
```

Each output row contains a chat-formatted `prompt` built from `prompt_av`, the
stable prompt-group `uid`, and `reward_inputs.text.video` / `.audio` built from
`prompt_v` and `prompt_a`. Native `idx`, `category`, and the original record are
retained in `extra_info`. Validation records without `idx` and `category` are
assigned deterministic zero-based indices and the category `validation`.

The converter writes `train.parquet` and `test.parquet`. Use `--train_size` or
`--val_size` to convert only a prefix; both default to `-1` (all records).

## Launch

```bash
bash examples/omnift_trainer/ltx2/run_ltx2_3_omninft_lora_npu.sh
```

The launcher reads the converted files from
`data/omninft/vggsound/verl_omni` by default. Override `DATA_DIR`, or set
`TRAIN_FILE` and `VAL_FILE` individually.

## Frozen recipe contract

- `actor_rollout_ref.model.algorithm=omni_nft`
- `actor_rollout_ref.model.model_type=omni_nft_model`
- `actor_rollout_ref.actor.diffusion_loss.loss_mode=omni_nft`
- `algorithm.trainer_type=direct_preference`
- `algorithm.sample_source=online`
- `algorithm.paired_preference=false`
- `actor_rollout_ref.rollout.n=8`
- `actor_rollout_ref.rollout.calculate_log_probs=False`
- `actor_rollout_ref.model.policy_state_adapters=[default,old]`
- `actor_rollout_ref.rollout.rollout_adapter=old`

The converted files use the standard `RLHFDataset`; no runtime custom dataset
or collator is required.
