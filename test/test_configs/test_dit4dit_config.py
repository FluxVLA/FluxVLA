# Copyright 2026 Limx Dynamics

from mmengine import Config


def test_dit4dit_training_budget_matches_released_checkpoint():
    cfg = Config.fromfile(
        'configs/dit4dit/dit4dit_libero_all_full_finetune.py')

    assert cfg.train_dataloader.per_device_batch_size == 8
    assert cfg.runner.grad_accumulation_steps == 1
    assert cfg.runner.metric.grad_accumulation_steps == 1
    assert cfg.runner.max_steps == 160_000
    assert cfg.runner.lr_scheduler.warmup_steps == 10_000

    optimizer = cfg.runner.optimizer
    assert optimizer.lr == 3e-5
    assert optimizer.paramwise_learning_rate[
        'vlm_backbone.transformer'] == 1e-4
    assert optimizer.paramwise_learning_rate['vla_head'] == 1e-4


def test_dit4dit_core_source_contract_is_preserved():
    cfg = Config.fromfile(
        'configs/dit4dit/dit4dit_libero_all_full_finetune.py')

    assert cfg.model.type == 'DiT4DiTVLA'
    assert cfg.model.repeated_diffusion_steps == 4
    assert cfg.model.vlm_backbone.extract_layer == 17
    assert cfg.model.vlm_backbone.detach_hidden_states is True
    assert cfg.model.vlm_backbone.flow_matching_time_distribution == 'uniform'
    assert cfg.model.vlm_backbone.flow_matching_high_sigma_ratio is None
    assert cfg.model.vla_head.action_horizon == 8
    assert cfg.model.vla_head.num_inference_timesteps == 4
    assert cfg.eval.eval_chunk_size == 8

    train_transforms = cfg.train_dataloader.dataset.datasets[0].transforms
    train_tokenizer = next(t for t in train_transforms
                           if t.type == 'ProcessCosmos25Prompt')
    assert train_tokenizer.input_key == 'task_description'
    assert train_tokenizer.remove_input_key is True
    assert train_tokenizer.tokenizer.model_path == (
        cfg.model.vlm_backbone.base_model + '/tokenizer')
    assert train_tokenizer.tokenizer.model_max_length == 512
    assert 'lang_tokens' in cfg.runner.collator['keys']
    assert 'lang_masks' in cfg.runner.collator['keys']
    assert cfg.runner.collator.meta_keys == ['info', 'stats']
    assert cfg.runner.tokenizer.type == 'PretrainedTokenizer'

    eval_tokenizer = next(t for t in cfg.eval.dataset.transforms
                          if t.type == 'ProcessCosmos25Prompt')
    assert eval_tokenizer.input_key == 'task_description'
    assert eval_tokenizer.remove_input_key is True
    assert cfg.runner.sharding_strategy == 'global-shard-grad-op'
