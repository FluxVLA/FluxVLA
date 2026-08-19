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
"""Low-change, BF16 PI0.5 RoboCasa score-target recipe.

This is the single production recipe selected after auditing the experiment
sheet, RLinf, OpenPI, and StarVLA. It deliberately starts from the official
PI0.5 base checkpoint instead of continuing the 31.58% RoboCasa checkpoint.

The public StarVLA 43.9% result is from QwenPI_v2, not OpenPI PI0.5, so 40% is
a target rather than a reproduced guarantee. Only its uniform 24-task mixture
and larger sample budget are transferred. The optimizer schedule follows the
RLinf/OpenPI values. Hybrid SHARD_GRAD_OP uses BF16 forward/backward compute
with FP32 master parameters, reductions, and buffers sharded within each node.

The converted dataset uses a single ego-view camera, 29-dimensional joint
states and absolute joint-position actions, q01/q99 quantile normalization,
and a 16-step action horizon. Set ``ROBOCASA_DATA_ROOT`` when the converted
LeRobot dataset is not in one of the checked default locations.

Expected topology: 32 H100 GPUs, for example 4 nodes x 8 GPUs. Per-device
batch 8 without accumulation gives an effective global batch 256.

Example for four 8-GPU nodes sharing MASTER_ADDR and MASTER_PORT:
    torchrun --nnodes=4 --nproc_per_node=8 \
        --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} scripts/train.py \
        --config \
        configs/pi05/\
pi05_paligemma_robocasa_full_data_full_finetune.py \
        --work-dir \
        work_dirs/pi05_paligemma_robocasa_full_data_full_finetune
"""

import os

# Generated from the exact default 24-task, 1000-episode-per-task
# training data with tools/compute_pi05_norm_stats.py.
_PI05_ROBOCASA_STATS = {
    'robocasa_gr1_24tasks_full': {
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
                -0.1775359626264437, 0.2284479243988733, -0.11081084655259829,
                -1.4611202696554626, 0.17829720094429205, 0.09361726611921038,
                -0.006783452401265867, -0.22126730340471804,
                -0.22126730340471804, -0.22126730340471804,
                -0.22126730340471804, -0.44253460680943607, 1.1117459333448283,
                -0.34179493812684864, -0.32883822428616244,
                0.06489898183750487, -1.4446330386524806, 0.3549797183859418,
                0.07628017885730197, 0.1775135722948219, -0.5055467696490632,
                -0.5055467696490632, -0.5055467696490632, -0.5055467696490632,
                -1.0110935392981264, 3.0, 0.0037711007729695077,
                0.006839471353927009, -0.00012166928530587137
            ],
            'std': [
                0.3879081074050779, 0.17765238673292205, 0.2720438511901107,
                0.4804564941142621, 0.3049441427473101, 0.29369821499250337,
                0.3279262188807178, 0.8859177354343982, 0.8859177354343982,
                0.8859177354343982, 0.8859177354343982, 1.7718354708687964,
                1.44888190662043, 0.522657539785618, 0.2942594437099049,
                0.4063810443537934, 0.7158570101306991, 0.5269314647148533,
                0.4040716204905002, 0.623581537540201, 1.412240228651775,
                1.412240228651775, 1.412240228651775, 1.412240228651775,
                2.82448045730355, 0.0, 0.07106289960401765,
                0.020791731558488753, 0.005434817471243943
            ],
            'min': [
                -1.710237741470337, -0.00042632038821466267,
                -1.4035195112228394, -2.801373243331909, -2.033860445022583,
                -1.4939793348312378, -1.3439968824386597, -1.5, -1.5, -1.5,
                -1.5, -3.0, 0.0, -2.1020591259002686, -2.2016870975494385,
                -2.3005123138427734, -2.848569631576538, -3.000068426132202,
                -1.4995447397232056, -1.489180326461792, -1.5, -1.5, -1.5,
                -1.5, -3.0, 3.0, -0.5332962870597839, -0.4947744905948639,
                -0.4128894805908203
            ],
            'max': [
                1.4058303833007812, 1.288877010345459, 1.2944501638412476,
                3.1500876502832398e-06, 2.563594102859497, 1.5000286102294922,
                1.4998135566711426, 1.5, 1.5, 1.5, 1.5, 3.0, 3.0,
                1.564226508140564, 0.00015454324602615088, 1.7984894514083862,
                5.365318065742031e-05, 3.0000767707824707, 1.4954047203063965,
                1.4975833892822266, 1.5, 1.5, 1.5, 1.5, 3.0, 3.0,
                0.9750414490699768, 0.34772318601608276, 0.5310259461402893
            ],
            'q01': [
                -1.4168124198913574, -1.65025339811109e-05,
                -0.9878814220428467, -2.587369680404663, -0.37570324540138245,
                -0.7240629196166992, -0.9552507400512695, -1.5, -1.5, -1.5,
                -1.5, -3.0, 0.0, -1.4654403924942017, -1.1332995891571045,
                -0.8891795873641968, -2.689577341079712, -0.735479474067688,
                -1.0292688608169556, -1.0800883769989014, -1.5, -1.5, -1.5,
                -1.5, -3.0, 3.0, -0.28852102160453796, -0.02933858148753643,
                -0.01415097527205944
            ],
            'q99': [
                0.71796053647995, 0.7908380627632141, 0.4401231110095978,
                -0.1634492427110672, 1.0564770698547363, 0.8349300420284376,
                0.7572318315505981, 1.5, 1.5, 1.5, 1.5, 3.0, 3.0,
                0.9571810364723206, -1.866471029643435e-05, 0.9322921633720398,
                -0.08507565408945084, 1.613282561302185, 0.8777858018875122,
                1.4806392192840576, 1.5, 1.5, 1.5, 1.5, 3.0, 3.0,
                0.21489860117435455, 0.09426142275333405, 0.011301719583570957
            ],
            'count':
            96320928
        }
    }
}

# The pre-rebase 51.42%-53.25% baseline used seed 42.  Keep it as the default
# now that deterministic CUDA kernels make the complete trajectory repeatable.
# An environment override supports deliberate, separately named seed sweeps.
train_seed = int(os.environ.get('PI05_TRAIN_SEED', '42'))
eval_seed = 7

_LOCAL_ROBOCASA_DATA_ROOT = './datasets/robocasa_lerobot_V2.1'
_SHARED_ROBOCASA_DATA_ROOT = (
    '/mnt/data/cpfs/mnt/data/yiming/fluxvla/upload_staging/'
    'robocasa_lerobot_V2.1')
_LOCAL_ROBOCASA_DATA_READY = os.path.isdir(
    f'{_LOCAL_ROBOCASA_DATA_ROOT}/PnPBottleToCabinetClose/videos')
_DEFAULT_ROBOCASA_DATA_ROOT = (
    _LOCAL_ROBOCASA_DATA_ROOT
    if _LOCAL_ROBOCASA_DATA_READY else _SHARED_ROBOCASA_DATA_ROOT)
_PI05_CHECKPOINT = os.environ.get('PI05_CHECKPOINT',
                                  './checkpoints/pi05_base/model.safetensors')
_PI05_TOKENIZER = os.environ.get('PI05_TOKENIZER', './checkpoints/pi05_base')

# The PI0.5 architecture matches the LIBERO and ALOHA variants. Its internal
# action dimension is 32; the 29 RoboCasa joints are padded with three zeros.
model = dict(
    type='PI05FlowMatching',
    # Match OpenPI's flow-matching objective and supervise all padded action
    # dimensions.
    time_sampler='beta',
    time_beta_alpha=1.5,
    time_beta_beta=1.0,
    loss_action_dim=32,
    openpi_fp32_flow=True,
    # PaliGemma backbone for image and language tokens.
    llm_backbone=dict(
        type='ConditionGemmaModel',
        adarms_cond_dim=None,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=2048,
        initializer_range=0.02,
        intermediate_size=16384,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        use_cache=True,
        vocab_size=257152,
    ),
    # SigLIP vision encoder.
    vision_backbone=dict(
        type='SigLIPViTBackbone',
        vision_backbone_id='siglip_224',
        openpi_stem_fp32=True,
        vision_config=dict(
            attention_dropout=0.0,
            hidden_act='gelu_pytorch_tanh',
            hidden_size=1152,
            image_size=224,
            intermediate_size=4304,
            layer_norm_eps=1e-06,
            model_type='siglip_vision_model',
            num_attention_heads=16,
            num_channels=3,
            num_hidden_layers=27,
            patch_size=14,
            projection_dim=2048,
            projector_hidden_act='gelu_fast',
            torch_dtype='float32',
            vision_use_head=False,
        ),
    ),
    # Vision-to-LLM projection.
    projector=dict(
        type='LinearProjector',
        in_dim=1152,
        out_dim=2048,
    ),
    # A 16-step chunk covers roughly 0.8 seconds at 20 Hz.
    proj_width=1024,
    n_action_steps=16,
    action_in_proj=dict(type='LinearProjector', in_dim=32, out_dim=1024),
    action_out_proj=dict(type='LinearProjector', in_dim=1024, out_dim=32),
    time_mlp_in=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    time_mlp_out=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    max_action_dim=32,
    # Gemma expert conditioned on state, action, and diffusion time through
    # adaptive RMS normalization.
    llm_expert=dict(
        type='ConditionGemmaModel',
        attention_bias=False,
        adarms_cond_dim=1024,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        pad_token_id=0,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        transformers_version='4.48.1',
        use_adarms=True,
        use_cache=True,
        vocab_size=257152),
    # PI0.5 injects the normalized 29D proprio state through discrete prompt
    # tokens, so the language backbone remains trainable during adaptation.
    freeze_llm_backbone=False,
    freeze_vision_backbone=False,
    # Initialize from the general PI0.5 base model rather than LIBERO weights.
    pretrained_name_or_path=_PI05_CHECKPOINT,
    # Map upstream PI0.5 checkpoint keys to FluxVLA parameter names.
    name_mapping={
        'llm_backbone': 'paligemma_with_expert.paligemma.model.language_model',
        'vision_backbone.vision':
        'paligemma_with_expert.paligemma.model.vision_tower',
        'projector.projector':
        'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',
        'llm_expert': 'paligemma_with_expert.gemma_expert.model',
        'time_mlp_in.projector': 'time_mlp_in',
        'time_mlp_out.projector': 'time_mlp_out',
        'action_in_proj.projector': 'action_in_proj',
        'action_out_proj.projector': 'action_out_proj',
        'llm_backbone.embed_tokens': 'paligemma_with_expert.paligemma.lm_head',
        'llm_expert.embed_tokens':
        'paligemma_with_expert.gemma_expert.lm_head',
    },
    strict_mapping=True,
    # Convert the large transformer modules to bf16 to reduce memory use.
    params_to_change_dtype=[
        'llm_expert.llm.model.layers',
        'vlm_backbone.vlm.model.language_model.layers',
        'vlm_backbone.vlm.model.vision_tower',
        'vlm_backbone.vlm.model.multi_modal_projector',
    ],
    ori_action_dim=29,
)

_ROBOCASA_STATISTIC_NAME = 'robocasa_gr1_24tasks_full'
_ROBOCASA_DATA_ROOT = os.environ.get('ROBOCASA_DATA_ROOT',
                                     _DEFAULT_ROBOCASA_DATA_ROOT)
_ROBOCASA_TASK_PREFIX = 'gr1_unified'
_ROBOCASA_ENV_SUFFIX = '_GR1ArmsAndWaistFourierHands_Env'


def _robocasa_data_path(task_name):
    return f'{_ROBOCASA_DATA_ROOT}/{task_name}'


def _robocasa_task_env(task_name):
    return f'{_ROBOCASA_TASK_PREFIX}/{task_name}{_ROBOCASA_ENV_SUFFIX}'


# The full dataset contains about 1,000 episodes for each of 24 tasks (6 seen
# and 18 novel), one 256x256 ego-view camera, 29-dimensional joint states and
# absolute actions, and fixed q01/q99 quantile statistics shared with eval.
train_dataloader = dict(
    # 8 samples/GPU x 32 GPUs x 1 accumulation step = global batch 256.
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        # Sample all 24 tasks uniformly, independently of episode count.
        type='DistributedBalancedRepeatingDataset',
        seed=train_seed,
        reshuffle_each_epoch=True,
        # Keep state and action statistics separate. Action statistics must
        # come from the action column rather than observation.state.
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_ROBOCASA_STATISTIC_NAME,
        # Use q01/q99 computed from these exact 24 full-data roots and
        # their supervised 16-step terminal padding.
        dataset_statistics=_PI05_ROBOCASA_STATS,
        datasets=dict(
            type='ParquetDataset',
            supervise_terminal_padding=True,
            # Converted task directories produced by
            # convert_robocasa_for_fluxvla.py.
            data_root_path=[
                _robocasa_data_path('PnPBottleToCabinetClose'),
                _robocasa_data_path('PnPCanToDrawerClose'),
                _robocasa_data_path('PnPCupToDrawerClose'),
                _robocasa_data_path('PnPMilkToMicrowaveClose'),
                _robocasa_data_path('PnPPotatoToMicrowaveClose'),
                _robocasa_data_path('PnPWineToCabinetClose'),
                _robocasa_data_path('PosttrainPnPNovelFromCuttingboard'
                                    'ToBasketSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromCuttingboard'
                                    'ToCardboardboxSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromCuttingboard'
                                    'ToPanSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromCuttingboard'
                                    'ToPotSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromCuttingboard'
                                    'ToTieredbasketSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlacemat'
                                    'ToBasketSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlacemat'
                                    'ToBowlSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlacemat'
                                    'ToPlateSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlacemat'
                                    'ToTieredshelfSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlateToBowlSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlate'
                                    'ToCardboardboxSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlateToPanSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlateToPlateSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromTray'
                                    'ToCardboardboxSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromTrayToPlateSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromTrayToPotSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromTray'
                                    'ToTieredbasketSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromTray'
                                    'ToTieredshelfSplitA'),
            ],
            transforms=[
                # Decode the requested Parquet columns and video frames.
                dict(
                    type='ProcessParquetInputs',
                    parquet_keys=[
                        'observation.state',  # 29D joint angles
                        'timestamp',  # Seconds
                        'actions',  # 29D target joint positions
                        'info',  # Dataset metadata
                        'stats',  # Normalization statistics
                        'action_masks',  # Valid-action masks
                    ],
                    # RoboCasa uses a single ego-view camera.
                    video_keys=[
                        'observation.images.ego_view',
                    ],
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions'],
                    },
                    # The pre- and post-rebase reference runs both decoded
                    # with torchvision/PyAV. Do not let an optional decoder
                    # installation silently change the training inputs.
                    video_backend='pyav'),
                # Preserve native state ordering and tokenize the normalized
                # 29D state, matching OpenPI.
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=32,  # Zero-pad to the model action dimension.
                    state_dim=29,
                    state_key='proprio',
                    action_key='action',
                    norm_type='quantile'),
                # Build the OpenPI-compatible state-conditioned prompt.
                dict(
                    type='PreparePromptWithState',
                    max_state_dim=29,
                    lowercase_task_description=False,
                    add_action_prefix=True),
                # Tokenize the prompt.
                dict(
                    type='ProcessPrompts',
                    max_len=200,
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=_PI05_TOKENIZER,
                    )),
                # Resize to 224 and apply the crop/color augmentations used by
                # the RoboCasa training recipe.
                dict(type='RandomCropImages', scale=0.95),
                dict(type='ResizeImages', height=224, width=224),
                dict(
                    type='ColorJitterImages',
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.08),
                # Match OpenPI PI0.5 image normalization: pixel / 255 * 2 - 1.
                dict(type='SimpleNormalizeImages'),
            ],
            action_window_size=16,
            action_key='action',
            # RoboCasa's 29D action contract is an absolute joint-position
            # target. Changing this to delta action is a separate experiment
            # that requires newly computed delta-action statistics and an
            # inverse transform in evaluation; it is not a rebase fix.
            use_delta=False,
            statistic_name=_ROBOCASA_STATISTIC_NAME,
            window_start_idx=0,
        )))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=None,
    # 100k global-256 updates expose 25.6M samples.
    max_steps=100000,
    grad_accumulation_steps=1,
    # CUDA SDPA's efficient backward kernel is otherwise not reproducible:
    # identical H100 runs already diverge after the first optimizer update.
    deterministic_algorithms=True,
    ema_decay=0.99,
    seed=train_seed,
    optimizer=dict(
        type='AdamW',
        lr=2.5e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        weight_decay_all_params=True,
        # Avoid the model-sized peak allocation from AdamW foreach state.
        foreach=False,
        fused=True,
    ),
    max_grad_norm=1.0,
    # Keep enough periodic checkpoints for closed-loop model selection.
    save_epoch_interval=1,
    save_iter_interval=10000,
    max_keep_ckpts=10,
    # Keep FP32 parameter all-gathers inside each 8-GPU node. Cross-node
    # global SHARD_GRAD_OP is much slower because every execution block then
    # all-gathers FP32 parameters over the inter-node fabric.
    sharding_strategy='shard-grad-op',
    fsdp_wrap_policy='execution-block',
    reduce_in_full_precision=True,
    collator=dict(
        type='DictCollator',
        keys=[
            'states',  # (B, 29) quantile-normalized joint state
            'observation.eepose',  # Optional; DictCollator skips missing keys.
            'timestamp',  # (B,)
            'images',  # (B, N_views, C, H, W)
            'img_masks',  # (B, N_views)
            'lang_tokens',  # (B, max_len)
            'lang_masks',  # (B, max_len)
            'actions',  # (B, chunk_size, 32), normalized and padded
            'action_masks',  # (B, chunk_size)
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=_PI05_TOKENIZER,
    ),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        window_size=1),
    lr_scheduler=dict(
        type='openpi-warmup+cosine-decay',
        warmup_steps=5000,
        # The three observed trajectories agree through roughly 30k steps,
        # then the old 100k decay keeps LR above 2e-5 while loss variance
        # grows sharply. Reach the floor before that instability dominates,
        # while retaining the 100k sample budget at a low adaptation LR.
        decay_steps=50000,
        min_lr=2.5e-6,
    ),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=True,
    change_key_name=False)

# Evaluate all 24 RoboCasa tasks.
# Example:
#   conda activate fluxvla && cd /root/projects/FluxVLA
#   CONFIG_DIR=configs/pi05
#   CONFIG=$CONFIG_DIR/pi05_paligemma_robocasa_full_data_full_finetune.py
#   CKPT=work_dirs/pi05_paligemma_robocasa_full_data_full_finetune/\
# checkpoints/latest-checkpoint.safetensors
#   NUM_GPUS=8 bash scripts/eval_robocasa_manager.sh "$CONFIG" "$CKPT"
#
# Optional override:
#   --cfg-options eval.num_trials_per_task=50 eval.seed=7
#
# unnorm_key must match the training statistic_name.
eval = dict(
    type='RobocasaEvalRunner',
    benchmark='robocasa',
    task_suite_name='robocasa',
    model_family='pi0',
    task_list=[
        _robocasa_task_env('PnPBottleToCabinetClose'),
        _robocasa_task_env('PnPCanToDrawerClose'),
        _robocasa_task_env('PnPCupToDrawerClose'),
        _robocasa_task_env('PnPMilkToMicrowaveClose'),
        _robocasa_task_env('PnPPotatoToMicrowaveClose'),
        _robocasa_task_env('PnPWineToCabinetClose'),
        _robocasa_task_env('PosttrainPnPNovelFromCuttingboard'
                           'ToBasketSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromCuttingboard'
                           'ToCardboardboxSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromCuttingboard'
                           'ToPanSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromCuttingboard'
                           'ToPotSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromCuttingboard'
                           'ToTieredbasketSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlacemat'
                           'ToBasketSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlacemat'
                           'ToBowlSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlacemat'
                           'ToPlateSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlacemat'
                           'ToTieredshelfSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlateToBowlSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlate'
                           'ToCardboardboxSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlateToPanSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlateToPlateSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromTray'
                           'ToCardboardboxSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromTrayToPlateSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromTrayToPotSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromTray'
                           'ToTieredbasketSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromTray'
                           'ToTieredshelfSplitA'),
    ],
    total_tasks=24,
    # Keep the 16-step prediction horizon, but replan halfway through it.
    # At 20 Hz this reduces open-loop execution from 0.8 s to 0.4 s without
    # changing the positive 100k-step training recipe.
    eval_chunk_size=8,
    max_episode_steps=720,
    num_trials_per_task=50,  # 1,200 episodes across 24 tasks.
    episode_seed_stride=50,
    seed=eval_seed,  # Match the GR00T RoboCasa evaluation initial states.
    unnorm_key=_ROBOCASA_STATISTIC_NAME,
    action_order='fluxvla',
    dataset=dict(
        type='RobocasaEvalDataset',
        unnorm_key=_ROBOCASA_STATISTIC_NAME,
        transforms=[
            # Evaluation preprocessing must match training: the 0.95 center
            # crop mirrors RandomCropImages, tanh maps pixels to [-1, 1], and
            # the bg_crop ego-view key matches the converted training camera.
            dict(
                type='ProcessRobocasaEvalInputs',
                img_key='video.ego_view_bg_crop_pad_res256_freq20',
                resize_size=224,
                center_crop_scale=0.95,
                normalize=True,
                value_range='tanh'),
            dict(
                type='NormalizeStatesAndActions',
                state_dim=29,
                state_key='proprio',
                action_key='action',
                norm_type='quantile'),
            dict(
                type='PreparePromptWithState',
                max_state_dim=29,
                lowercase_task_description=False,
                add_action_prefix=True),
            dict(
                type='ProcessPrompts',
                max_len=200,
                tokenizer=dict(
                    type='PretrainedTokenizer', model_path=_PI05_TOKENIZER)),
        ]),
    denormalize_action=dict(
        type='DenormalizeRobocasaAction',
        norm_type='quantile',
        action_dim=29,
        clip_actions=False,
        stats_order='native',
    ),
)

themis = dict(
    transport=dict(
        service_name='/fluxvla/predict_action',
        report_service_name='/fluxvla/report_evaluation',
        timeout_s=30.0,
        image_keys=['video.ego_view_bg_crop_pad_res256_freq20'],
        state_keys=[
            'state.left_arm',
            'state.left_hand',
            'state.right_arm',
            'state.right_hand',
            'state.waist',
        ],
        unnorm_key=_ROBOCASA_STATISTIC_NAME,
        image_encoding='rgb8',
    ),
    runner=dict(
        type='EvalRunner',
        environment=dict(
            type='RoboCasaEnvironment',
            task_list=eval['task_list'],
            action_order=eval['action_order'],
            deterministic_env=True,
            prompt_key='annotation.human.coarse_action',
            render_key='video.ego_view_pad_res256_freq20',
        ),
        model_client=dict(type='FluxVLAROSModelClient'),
        evaluator=dict(type='SuccessRateEvaluator'),
        seed=eval['seed'],
        # Preserve the old inherited config's base-time value. The formal
        # RobocasaEvalRunner protocol above still evaluates 50 trials/task.
        episodes_per_task=20,
        max_episode_steps=eval['max_episode_steps'],
        execute_horizon=eval['eval_chunk_size'],
        stop_on_success=True,
        parallel_workers=1,
        simulator_gpu_ids=None,
        work_dir='work_dirs/fluxthemis',
    ),
    ros_server=dict(
        ros_version=1,
        dataset_section='eval',
        evaluation_reporting=dict(
            result_output_dir='work_dirs/fluxthemis',
            report_kind='robocasa',
        ),
        device='cuda:0',
        workers=dict(
            startup_timeout_s=900.0,
            request_timeout_s=120.0,
            lease_timeout_s=900.0,
        ),
        mixed_precision_dtype='bf16',
        enable_mixed_precision=True,
        model_outputs_environment_actions=False,
        forward_seed=False,
        denormalize_context={},
        denormalize_per_action=True,
    ),
)
