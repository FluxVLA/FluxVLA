"""Native VPP2 inference service for RoboDojo EE16 observations."""

import os
from copy import deepcopy

_CHECKPOINT = os.environ.get('VPP2_CHECKPOINT',
                             './checkpoints/vpp2_joint2b_s100000_fluxvla')
_WAN = os.environ.get('VPP2_WAN_DIR', './checkpoints/Wan2.1-I2V-14B-480P')
_STATS = os.environ.get('VPP2_STATS',
                        os.path.join(_CHECKPOINT, 'dataset_statistics.json'))
_PROMPT = ("A video recorded from a robot's point of view "
           'executing the following instruction: {task}')
# Expected quotas for the existing server-side result validation.
# Simulator tasks and rollout settings live in FluxThemis.
_GENERALIZATION = ('stack_bowls', 'push_T', 'pack_objects_into_box',
                   'fold_clothes', 'hang_mugs', 'sweep_blocks',
                   'pour_liquid_into_cup', 'make_toast',
                   'arrange_largest_number', 'sort_nesting_dolls_by_size',
                   'store_laptop_and_headphones', 'stack_blocks')
_EPISODES = {
    name: 25
    for task in _GENERALIZATION for name in (task, task + '_random')
}

model = dict(
    type='VPP2VLA',
    pretrained_name_or_path=_CHECKPOINT,
    vlm_backbone=dict(type='VPP2Wan21Backbone', pretrained_name_or_path=_WAN),
    # Native architecture from joint2b_s100000 (VPP2 revision 3365872).
    vla_head=dict(
        type='VPP2Head',
        video_dit_config=dict(
            model_type='i2v',
            patch_size=[1, 2, 2],
            text_len=128,
            in_dim=36,
            dim=5120,
            hidden_dim=5120,
            ffn_dim=13824,
            freq_dim=256,
            text_dim=4096,
            out_dim=16,
            num_heads=40,
            attn_head_dim=128,
            num_layers=40,
            eps=1e-6,
            qk_norm=True,
            cross_attn_norm=True,
            use_gradient_checkpointing=False,
            video_attention_mask_mode='bidirectional'),
        action_dit_config=dict(
            action_dim=16,
            hidden_dim=1024,
            ffn_dim=4096,
            num_layers=40,
            num_heads=40,
            attn_head_dim=128,
            text_dim=4096,
            freq_dim=256,
            eps=1e-6,
            use_gradient_checkpointing=False),
        proprio_dim=16,
        video_scheduler=dict(num_train_timesteps=1000),
        action_scheduler=dict(infer_shift=3., num_train_timesteps=1000),
        mot_checkpoint_mixed_attn=False,
        action_visible_video_frames=5),
    action_horizon=32,
    num_video_frames=17,
    num_inference_steps=10,
    sigma_shift=1.,
    replan_seed_base=1,
    seed_mode='vpp2',
    torch_dtype='bf16')
inference_model = deepcopy(model)

eval = dict(
    report_kind='robodojo',
    task_suite_name='robodojo',
    model_family='vpp2',
    num_trials_per_task=50,
    num_trials_per_task_overrides=_EPISODES,
    dataset=dict(
        type='VPP2RoboDojoEvalDataset',
        norm_stats=_STATS,
        unnorm_key='robodojo_ee16',
        transforms=[
            dict(type='VPP2HistoryToConditionVideo'),
            dict(type='VPP2PackEEState'),
            dict(
                type='StateFromInputs',
                norm_type='mean_std',
                stat_key='proprio',
                mean_std_clip=5.),
            dict(
                type='LiberoPromptFromInputs',
                max_len=128,
                prompt_template=_PROMPT,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=os.path.join(_WAN, 'google/umt5-xxl')),
                use_conversation=False),
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        action_dim=16,
        norm_type='mean_std',
        clip_normalized_action=False,
        statistic_name='robodojo_ee16'))

themis = dict(
    transport=dict(
        image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        state_keys=[
            'left_ee_pose', 'left_gripper_qpos', 'right_ee_pose',
            'right_gripper_qpos'
        ],
        unnorm_key='robodojo_ee16'),
    server=dict(
        ckpt_path=_CHECKPOINT,
        norm_stats_path=_STATS,
        runner=dict(type='FluxVLAServingRunner'),
        evaluation_reporting=dict(enabled=True),
        denormalize_context=dict(task_suite_name='robodojo_ee16')))

seed = 100008
train_dataloader = None
val_dataloader = None
eval_dataset = None
