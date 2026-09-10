# Copyright 2026 Limx Dynamics
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
"""FastWAM uncond training on the RoboCasa GR1 24-task full dataset.

Video decoding explicitly uses TorchCodec, matching the FastWAM LIBERO
training recipes.

The intended 32-GPU launch uses a per-device microbatch of eight without
gradient accumulation, for global batch 256. On Alibaba Cloud DLC, launch
the same script once on every worker:
"""

import os

seed = 42
eval_seed = 7

_FASTWAM_ROOT = '/root/projects/ryanhu/FluxVLA/FastWAM/fastwam_base'
_FASTWAM_CHECKPOINT = os.environ.get(
    'FASTWAM_CHECKPOINT', f'{_FASTWAM_ROOT}/fastwam_base_full.safetensors')
_FASTWAM_TOKENIZER = os.environ.get('FASTWAM_TOKENIZER',
                                    f'{_FASTWAM_ROOT}/tokenizer')

_LOCAL_ROBOCASA_DATA_ROOT = './datasets/robocasa_lerobot_V2.1'
_SHARED_ROBOCASA_DATA_ROOT = (
    '/mnt/data/cpfs/mnt/data/yiming/fluxvla/upload_staging/'
    'robocasa_lerobot_V2.1')
_LOCAL_ROBOCASA_DATA_READY = os.path.isdir(
    f'{_LOCAL_ROBOCASA_DATA_ROOT}/PnPBottleToCabinetClose/videos')
_DEFAULT_ROBOCASA_DATA_ROOT = (
    _LOCAL_ROBOCASA_DATA_ROOT
    if _LOCAL_ROBOCASA_DATA_READY else _SHARED_ROBOCASA_DATA_ROOT)
_ROBOCASA_DATA_ROOT = os.environ.get('ROBOCASA_DATA_ROOT',
                                     _DEFAULT_ROBOCASA_DATA_ROOT)

# Generated from the exact 24-task full-data training roots with a 32-step
# action horizon and supervised terminal padding. Keeping these q01/q99
# statistics in the config avoids a separate runtime file dependency.
_FASTWAM_ROBOCASA_STATS = {
    'robocasa_gr1_24tasks_joint_delta_h32': {
        'proprio': {
            'mean': [
                -0.17102599661893628, 0.23514659219974143,
                -0.11291017724516655, -1.4712459937182183, 0.17786245443243578,
                0.10149730971890933, -0.006413776953445108, 0.1501827995845328,
                0.1494419544656495, 0.13239416445355168, 0.14923437345912333,
                0.03464704927460434, 0.6337165986729069, -0.32031619897959673,
                -0.3216767278879253, 0.08499577598004783, -1.4728465043313443,
                0.337759733973541, 0.0690973699961625, 0.16150938097923964,
                0.48730800244509526, 0.4577669563030523, 0.42192603277520035,
                0.4522622699974382, 0.07554572365544217, 1.6688002354105917,
                0.0035976467848950486, 0.004405950191815653,
                -8.76750048624244e-05
            ],
            'std': [
                0.37524638882030775, 0.17814931412288376, 0.2653912135335837,
                0.46622207697619644, 0.2902881388466011, 0.2859473255400859,
                0.3154311391865683, 0.41833180143882503, 0.40040645672050007,
                0.35368029790605027, 0.4026344337142858, 0.13890309096880296,
                0.8191262033029768, 0.5024563669478592, 0.2778612776919975,
                0.37864273716884367, 0.6704838466386019, 0.5231829479691116,
                0.3766345142460483, 0.5706149895448006, 0.5937571234710008,
                0.5513828952053532, 0.506991319063774, 0.5441043791035441,
                0.17027079980800192, 0.21279093629856186, 0.06621792097846523,
                0.01964003934576096, 0.007552978323748122
            ],
            'min': [
                -1.6789460182189941, -0.026101894676685333,
                -1.3480229377746582, -2.5160419940948486, -1.9940674304962158,
                -1.3795876502990723, -1.1958755254745483, -1.4389894008636475,
                -1.8303323984146118, -2.4635109901428223, -1.7167329788208008,
                -2.218892812728882, -1.526924967765808, -2.0664756298065186,
                -2.1021976470947266, -2.296651601791382, -2.5318210124969482,
                -3.0013694763183594, -1.4908946752548218, -1.2908861637115479,
                -1.4716511964797974, -2.0171985626220703, -2.412123203277588,
                -1.189025640487671, -0.8325809836387634, -0.21484142541885376,
                -0.5222951769828796, -0.42820972204208374, -0.39791223406791687
            ],
            'max': [
                1.3502349853515625, 1.2633577585220337, 1.2589013576507568,
                0.001734813442453742, 2.521491289138794, 1.526998519897461,
                1.496475338935852, 2.0179455280303955, 2.009377956390381,
                2.6196515560150146, 1.8978251218795776, 3.2151029109954834,
                2.7924649715423584, 1.5148204565048218, 0.003278259886428714,
                1.7851011753082275, 0.0016116079641506076, 3.0015335083007812,
                1.4080945253372192, 1.4516682624816895, 2.7859506607055664,
                2.1664254665374756, 3.0131356716156006, 2.69866681098938,
                1.4733597040176392, 2.079848289489746, 0.937696099281311,
                0.3457968235015869, 0.47687003016471863
            ],
            'q01': [
                -1.414753302335739, -0.0005171521747251973,
                -0.9782302141189575, -2.477926731109619, -0.34331061780452726,
                -0.6772882187366486, -0.9085569721460343, -0.2537090674042702,
                -0.01579869568347931, -0.010405048383399845,
                -0.002593582069966942, -0.14740002006292344,
                -0.0005192354379687458, -1.4483043837547303,
                -1.0833211290836333, -0.8002108770608902, -2.507189002037048,
                -0.7147443491220474, -0.9463434845209122, -1.0013096010684968,
                -0.004114496670663357, -0.004300017701461911,
                -0.0054274908918887374, -0.004352558837272227,
                -0.13891243800520897, 0.5844185560941696, -0.2750973534584045,
                -0.031067517586052418, -0.022482833340764046
            ],
            'q99': [
                0.7154520624876013, 0.7829802078008643, 0.4349772733449915,
                -0.170762614309788, 1.0356361699104308, 0.8310365939140318,
                0.7348084545135478, 1.500377825498581, 1.4995973110198975,
                1.2963906359672546, 1.5020229816436768, 0.6279667210578896,
                1.846041305065155, 0.9360333341360092, -0.00022184939269209443,
                0.8936860918998715, -0.08954395778477237, 1.5831496250629415,
                0.8285187083482737, 1.232260091304779, 1.497841477394104,
                1.4997276926040648, 1.5635861182212825, 1.518557515144348,
                0.6836496728658665, 1.8118253779411315, 0.2004491922259326,
                0.08782679289579387, 0.022298754360526682
            ],
            'count':
            6020058
        },
        'action': {
            'mean': [
                -0.011043914986309815, -0.010007360312809544,
                0.0042910483025987685, 0.0165095600485725,
                -0.0001492156577750663, -0.010921732040627618,
                -0.0027297371728046923, -0.21777099157516422,
                -0.21777099157516422, -0.21777099157516422,
                -0.21777099157516422, -0.43554198315032844, 1.1117459333448283,
                -0.03831366418639272, -0.01117492435450355,
                -0.03554679189270138, 0.05331557449699476,
                0.023938561955056795, 0.010414526599275284,
                0.015030495568124405, -0.5038520289173294, -0.5038520289173294,
                -0.5038520289173294, -0.5038520289173294, -1.0077040578346588,
                3.0, 0.0003764941150544639, 0.0028997816025660996,
                8.871427652477556e-05
            ],
            'std': [
                0.14597827628316, 0.07078327391810804, 0.10866330900772775,
                0.20285214165552168, 0.11364799247580508, 0.12208934715920647,
                0.13001478033922337, 0.8867836491092731, 0.8867836491092731,
                0.8867836491092731, 0.8867836491092731, 1.7735672982185462,
                1.448881906604776, 0.2569391969669199, 0.17057495830529007,
                0.20904752081440497, 0.3505219726115055, 0.2596407292681979,
                0.2487641347602507, 0.3223890608184807, 1.4128457554113676,
                1.4128457554113676, 1.4128457554113676, 1.4128457554113676,
                2.825691510822735, 0.0, 0.04150128705709914,
                0.014929787363288202, 0.0069895692825420255
            ],
            'min': [
                -1.22250497341156, -0.862007737159729, -1.3653254508972168,
                -1.7758779525756836, -1.9212892055511475, -1.4581422805786133,
                -1.41221022605896, -1.5, -1.5, -1.5, -1.5, -3.0, 0.0,
                -1.6761599779129028, -1.7168105840682983, -1.8580408096313477,
                -2.1325106620788574, -1.9244258403778076, -2.0225167274475098,
                -2.189816474914551, -1.5, -1.5, -1.5, -1.5, -3.0, 3.0,
                -0.4210658073425293, -0.3053489625453949, -0.3253996968269348
            ],
            'max': [
                1.4428503513336182, 1.1318122148513794, 1.2746416330337524,
                1.896183729171753, 1.9371355772018433, 1.541634440422058,
                1.4781038761138916, 1.5, 1.5, 1.5, 1.5, 3.0, 3.0,
                1.8569952249526978, 1.1954123973846436, 1.5323610305786133,
                2.0053327083587646, 2.2619383335113525, 1.8036917448043823,
                2.475435256958008, 1.5, 1.5, 1.5, 1.5, 3.0, 3.0,
                0.4945973753929138, 0.31382834911346436, 0.36962956190109253
            ],
            'q01': [
                -0.5177125602960586, -0.177993506193161, -0.4362335905432701,
                -0.6846377193927765, -0.30759522020816804, -0.354341921210289,
                -0.49349108487367627, -1.5, -1.5, -1.5, -1.5, -3.0, 0.0,
                -0.8105077147483826, -0.5372815400362014, -0.6666148573160171,
                -0.9314047515392303, -0.6511471807956696, -0.6852044403553009,
                -0.7683310508728027, -1.5, -1.5, -1.5, -1.5, -3.0, 3.0,
                -0.12688052654266357, -0.04177360907196999,
                -0.02322256090119481
            ],
            'q99': [
                0.4924031332135197, 0.28378541916608135, 0.2716618776321411,
                0.7499560654163346, 0.4148592755198468, 0.49025590568780864,
                0.42918470203876424, 1.5, 1.5, 1.5, 1.5, 3.0, 3.0,
                0.6885550558567033, 0.4373265013098706, 0.5224482685327523,
                1.0974481642246232, 0.807758775353431, 0.726399984955787,
                1.0922142267227173, 1.5, 1.5, 1.5, 1.5, 3.0, 3.0,
                0.12698104158043844, 0.0480356439948082, 0.02092310851439827
            ],
            'count':
            192641856
        }
    }
}

_FRAME_WINDOW_SIZE = 9
_ACTION_WINDOW_SIZE = 32
_ACTION_NORM_TYPE = 'quantile'
_FRAME_SAMPLE_STRIDE = 4
_STATISTIC_NAME = 'robocasa_gr1_24tasks_joint_delta_h32'
_JOINT_DELTA_MASK = ([True] * 7 + [False] * 6 + [True] * 7 + [False] * 6 +
                     [True] * 3)
_TASK_PREFIX = 'gr1_unified'
_ENV_SUFFIX = '_GR1ArmsAndWaistFourierHands_Env'
_TEXT_PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following "
    'instruction: {task}')

_TASK_DIRS = [
    'PnPBottleToCabinetClose',
    'PnPCanToDrawerClose',
    'PnPCupToDrawerClose',
    'PnPMilkToMicrowaveClose',
    'PnPPotatoToMicrowaveClose',
    'PnPWineToCabinetClose',
    'PosttrainPnPNovelFromCuttingboardToBasketSplitA',
    'PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA',
    'PosttrainPnPNovelFromCuttingboardToPanSplitA',
    'PosttrainPnPNovelFromCuttingboardToPotSplitA',
    'PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA',
    'PosttrainPnPNovelFromPlacematToBasketSplitA',
    'PosttrainPnPNovelFromPlacematToBowlSplitA',
    'PosttrainPnPNovelFromPlacematToPlateSplitA',
    'PosttrainPnPNovelFromPlacematToTieredshelfSplitA',
    'PosttrainPnPNovelFromPlateToBowlSplitA',
    'PosttrainPnPNovelFromPlateToCardboardboxSplitA',
    'PosttrainPnPNovelFromPlateToPanSplitA',
    'PosttrainPnPNovelFromPlateToPlateSplitA',
    'PosttrainPnPNovelFromTrayToCardboardboxSplitA',
    'PosttrainPnPNovelFromTrayToPlateSplitA',
    'PosttrainPnPNovelFromTrayToPotSplitA',
    'PosttrainPnPNovelFromTrayToTieredbasketSplitA',
    'PosttrainPnPNovelFromTrayToTieredshelfSplitA',
]


def _robocasa_data_path(task_name: str) -> str:
    return f'{_ROBOCASA_DATA_ROOT}/{task_name}'


def _robocasa_task_env(task_name: str) -> str:
    return f'{_TASK_PREFIX}/{task_name}{_ENV_SUFFIX}'


model = dict(
    type='FastWAMVLA',
    pretrained_name_or_path=_FASTWAM_CHECKPOINT,
    pretrained_skip_prefixes=[
        'vla_head.mot.mixtures.action.action_encoder.',
        'vla_head.mot.mixtures.action.head.',
        'vla_head.proprio_encoder.',
    ],
    torch_dtype='bf16',
    num_views=1,
    frame_window_size=_FRAME_WINDOW_SIZE,
    proprio_dim=29,
    action_horizon=_ACTION_WINDOW_SIZE,
    num_inference_steps=10,
    action_norm_type=_ACTION_NORM_TYPE,
    mot_checkpoint_mixed_attn=True,
    freeze_vlm_backbone=True,
    vlm_backbone=dict(
        type='Wan22Backbone',
        text_embed_cache_context_len=128,
        text_embed_cache_size=256,
        text_embed_cache_device='cpu',
    ),
    vla_head=dict(
        type='FastWAMHead',
        video_dit_config=dict(
            has_image_input=False,
            patch_size=[1, 2, 2],
            in_dim=48,
            hidden_dim=3072,
            ffn_dim=14336,
            freq_dim=256,
            text_dim=4096,
            out_dim=48,
            num_heads=24,
            attn_head_dim=128,
            num_layers=30,
            eps=1.0e-06,
            seperated_timestep=True,
            require_clip_embedding=False,
            require_vae_embedding=False,
            fuse_vae_embedding_in_latents=True,
            video_attention_mask_mode='first_frame_causal',
            action_conditioned=False,
            action_dim=29,
            action_group_causal_mask_mode='group_diagonal',
            use_gradient_checkpointing=True,
        ),
        action_dit_config=dict(
            action_dim=29,
            hidden_dim=1024,
            ffn_dim=4096,
            num_heads=24,
            attn_head_dim=128,
            num_layers=30,
            text_dim=4096,
            freq_dim=256,
            eps=1.0e-06,
            use_gradient_checkpointing=True,
        ),
        video_scheduler=dict(
            train_shift=5.0, infer_shift=5.0, num_train_timesteps=1000),
        action_scheduler=dict(
            train_shift=5.0, infer_shift=5.0, num_train_timesteps=1000),
        loss=dict(lambda_video=1.0, lambda_action=1.0),
    ),
)

inference_model = model.copy()

train_dataloader = dict(
    # 8 samples/GPU x 32 GPUs x 1 accumulation step = global batch 256.
    per_device_batch_size=8,
    per_device_num_workers=8,
    dataset=dict(
        type='DistributedBalancedRepeatingDataset',
        seed=seed,
        reshuffle_each_epoch=True,
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_STATISTIC_NAME,
        dataset_statistics=_FASTWAM_ROBOCASA_STATS,
        datasets=dict(
            type='ParquetDataset',
            data_root_path=[
                _robocasa_data_path(task_name) for task_name in _TASK_DIRS
            ],
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    parquet_keys=[
                        'observation.state',
                        'timestamp',
                        'actions',
                        'info',
                        'stats',
                        'action_masks',
                    ],
                    video_keys=['observation.images.ego_view'],
                    video_backend='torchcodec',
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions'],
                    },
                ),
                dict(
                    type='RelativeActions',
                    mask=_JOINT_DELTA_MASK,
                    state_key='states',
                    action_key='actions',
                ),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=29,
                    state_dim=29,
                    state_key='proprio',
                    action_key='action',
                    norm_type=_ACTION_NORM_TYPE,
                    output_dtype='float32',
                ),
                dict(type='RandomCropImages', scale=0.95),
                dict(type='ResizeImages', height=224, width=224),
                dict(
                    type='ColorJitterImages',
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.08,
                ),
                dict(type='SimpleNormalizeImages'),
                dict(
                    type='PrepareVideo',
                    num_views=1,
                    frame_window_size=_FRAME_WINDOW_SIZE,
                    tile_direction='horizontal',
                ),
                dict(
                    type='LiberoPromptFromInputs',
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=_FASTWAM_TOKENIZER,
                    ),
                    max_len=128,
                    use_conversation=False,
                    prompt_template=_TEXT_PROMPT_TEMPLATE,
                ),
            ],
            action_window_size=_ACTION_WINDOW_SIZE,
            action_key='action',
            statistic_name=_STATISTIC_NAME,
            supervise_terminal_padding=True,
            window_start_idx=0,
            frame_window_size=_FRAME_WINDOW_SIZE,
            frame_sample_stride=_FRAME_SAMPLE_STRIDE,
        ),
    ),
)

val_dataloader = None
eval_dataset = None

runner = dict(
    type='DDPTrainRunner',
    max_epochs=None,
    max_steps=100000,
    max_keep_ckpts=5,
    save_iter_interval=5000,
    optimizer=dict(lr=1e-4, type='AdamW', weight_decay=1e-2),
    max_grad_norm=1.0,
    collator=dict(
        type='DictCollator',
        keys=[
            'states',
            'images',
            'img_masks',
            'actions',
            'action_masks',
            'frame_masks',
            'lang_tokens',
            'lang_masks',
        ],
        meta_keys=['task_description', 'info', 'stats', 'timestamp'],
    ),
    sampler=None,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        window_size=1,
    ),
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay-min-lr',
        warmup_ratio=0.05,
        min_lr_ratio=0.05,
        betas=(0.9, 0.95),
        weight_decay_style='uniform',
    ),
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    grad_accumulation_steps=1,
    mixed_precision_dtype='bf16',
    static_graph=False,
    seed=seed,
    evaluator=dict(
        type='training-eval',
        eval_every=1000,
        num_inference_steps=10,
        seed=seed,
        save_video=True,
        video_fps=8,
    ),
)

eval = dict(
    type='RobocasaEvalRunner',
    benchmark='robocasa',
    task_suite_name='robocasa',
    model_family='fastwam',
    task_list=[_robocasa_task_env(task_name) for task_name in _TASK_DIRS],
    total_tasks=len(_TASK_DIRS),
    eval_chunk_size=8,
    max_episode_steps=720,
    num_trials_per_task=50,
    episode_seed_stride=50,
    seed=eval_seed,
    unnorm_key=_STATISTIC_NAME,
    action_order='fluxvla',
    dataset=dict(
        type='RobocasaEvalDataset',
        unnorm_key=_STATISTIC_NAME,
        transforms=[
            dict(
                type='ProcessRobocasaEvalInputs',
                img_key='video.ego_view_bg_crop_pad_res256_freq20',
                resize_size=224,
                center_crop_scale=0.95,
                normalize=True,
                value_range='tanh',
            ),
            dict(
                type='NormalizeStatesAndActions',
                state_dim=29,
                state_key='proprio',
                action_key='action',
                norm_type=_ACTION_NORM_TYPE,
                output_dtype='float32',
            ),
            dict(
                type='LiberoPromptFromInputs',
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=_FASTWAM_TOKENIZER,
                ),
                max_len=128,
                use_conversation=False,
                prompt_template=_TEXT_PROMPT_TEMPLATE,
            ),
            dict(
                type='PrepareVideo',
                num_views=1,
                frame_window_size=1,
                tile_direction='horizontal',
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizeDeltaAction',
        statistic_name=_STATISTIC_NAME,
        norm_type=_ACTION_NORM_TYPE,
        action_dim=29,
        delta_action_mask=_JOINT_DELTA_MASK,
    ),
)
