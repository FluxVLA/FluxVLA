# Copyright 2026 Limx Dynamics
"""Standalone relative-action PI0.5 Candy training and inference config.

The PI0.5 architecture and training recipe follow the Aloha full-finetuning
config. Dataset roots, prompts and the 43-D state / 52-D action contract come
from the HUD04 Candy WAM config.
"""

import os

# Offline statistics after applying the 31-D relative-action transform.
_PI05_CANDY_STATS = {
    'private': {
        'proprio': {
            'mean': [
                -0.18119053733304075, 0.0360180894197736, -0.03865447971061218,
                0.08433490510900854, 0.0036357935833214567,
                -0.05662348552768057, -0.17758249232730797,
                -0.038448519268685075, -0.05923530082207567,
                0.11839816806488092, -0.03407377852483353, 0.09657260841305153,
                -0.021408170767755, 0.0020213507256407674, 0.1787013816913143,
                0.03410994696427701, 0.4286247379401394, -0.5051461707475623,
                0.3336838022493641, -0.8029909690859051, -0.9379799283228909,
                -0.46005260895569366, -0.06582849784824317,
                0.35190993985711605, 0.06644092153303154, -0.24858985073039055,
                0.5103445536439878, -1.5214839498560688, 0.006512862884973919,
                -0.5110928469102006, -0.35906797878364133, 37.84139063037578,
                2.063902480981689, 94.68441310829925, 97.00406169960625,
                42.649293742115525, 2.066585591956879, 87.95833890821514,
                2.0672474100691924, 87.91859398295041, 2.0639669903283764,
                87.91367932642684, 2.0627221988608127
            ],
            'std': [
                0.15128748039131654, 0.048799478709769126, 0.14521338993901006,
                0.06230411241327338, 0.05517437247637862, 0.034480450814396796,
                0.14681260660894324, 0.055998832686818184, 0.14110243995466704,
                0.06439120723472082, 0.053772874082342124,
                0.039553731380478005, 0.0686471948527093, 0.034909851245100654,
                0.10453329549744378, 0.08232399650725103, 0.10697263719263582,
                0.327896296526267, 0.14670968741985274, 0.24021996607164,
                0.2589073293196847, 0.26739926193534075, 0.24885313333510872,
                0.1579283787330976, 0.17563066355316445, 0.04982631023321773,
                0.11979938612788452, 0.17001076142967653, 0.17902954353148734,
                0.1462574987018676, 0.16326009956174553, 16.662594733131897,
                0.6520075774743664, 2.4878052639180073, 0.06360190407993668,
                14.702583419539218, 0.649677347712373, 24.066788852955398,
                0.6500999990587124, 24.141707203062772, 0.6530100311857292,
                24.165350743893537, 0.654261980362152
            ],
            'min': [
                -0.8380997776985168, -0.19300013780593872, -0.5158999562263489,
                -0.050000015646219254, -0.22513873875141144,
                -0.15765279531478882, -0.8264148831367493, -0.2316001057624817,
                -0.6469996571540833, -0.06350023299455643,
                -0.19028916954994202, -0.005182403605431318,
                -0.3502996563911438, -0.12988382577896118,
                -0.10532969981431961, -0.28249961137771606,
                0.018999576568603516, -1.8096998929977417,
                -0.11280350387096405, -1.6001297235488892, -1.7917999029159546,
                -1.2298997640609741, -1.0118998289108276, -0.26169997453689575,
                -0.5516999363899231, -0.5203962922096252, 0.039129674434661865,
                -1.977199912071228, -0.7268999218940735, -0.9370997548103333,
                -1.0417999029159546, 2.0, 1.0, 92.0, 97.0, 2.0, 1.0, 2.0, 1.0,
                2.0, 1.0, 2.0, 1.0
            ],
            'max': [
                0.3143000602722168, 0.18799996376037598, 0.6189999580383301,
                0.4344000816345215, 0.1804361492395401, 0.06875385344028473,
                0.3018849492073059, 0.21559998393058777, 0.5090000629425049,
                0.3424999713897705, 0.14933738112449646, 0.22688327729701996,
                0.1819000244140625, 0.14273610711097717, 0.5375312566757202,
                0.41140007972717285, 0.7713000774383545, 0.40960001945495605,
                1.0308961868286133, -0.05672961473464966, -0.18369990587234497,
                0.7499000430107117, 0.7195000648498535, 0.9586999416351318,
                0.5619999170303345, -0.04249627888202667, 0.9796299934387207,
                -0.6155999302864075, 0.5692000389099121, 0.13660001754760742,
                0.3794999122619629, 57.0, 15.0, 97.0, 98.0, 57.0, 15.0, 97.0,
                15.0, 97.0, 15.0, 97.0, 15.0
            ],
            'q01': [
                -0.5765568375587463, -0.0873001292347908, -0.35765711069107053,
                -0.026656786352395982, -0.12090455740690231,
                -0.13962708413600922, -0.5677149891853333,
                -0.18240004777908325, -0.3763567566871643,
                -0.0034001509193331003, -0.1405610264837742,
                0.018060583621263504, -0.24539977312088013,
                -0.07860004901885986, -0.025688044726848602,
                -0.162800133228302, 0.17779970169067383, -1.221199870109558,
                -0.022203965112566948, -1.367229700088501, -1.572600245475769,
                -0.9132998585700989, -0.81389981508255, -0.021099869161844254,
                -0.3522999882698059, -0.37209638953208923, 0.2160300612449646,
                -1.8671001195907593, -0.36069995164871216, -0.7875998616218567,
                -0.8053000569343567, 2.0, 1.0, 92.0, 97.0, 2.0, 1.0, 2.0, 1.0,
                2.0, 1.0, 2.0, 1.0
            ],
            'q99': [
                0.13070006012916566, 0.14439988136291504, 0.34759998321533203,
                0.27250003814697266, 0.1208365060389042, 0.012543297372758389,
                0.13228493928909302, 0.11719998717308044, 0.32760000228881836,
                0.2746999263763428, 0.09398054555058492, 0.18251116558909417,
                0.10790014266967773, 0.08918592654168608, 0.4172881069779397,
                0.24450016021728516, 0.6597001552581787, 0.22189998626708984,
                0.715996265411377, -0.3136299252510071, -0.4442429184913628,
                0.3393000364303589, 0.3890998363494873, 0.7334997653961182,
                0.4154999256134033, -0.1360963135957718, 0.7783298492431641,
                -1.1140998601913452, 0.4541001319885254, -0.14299994707107544,
                -0.04639989510178566, 57.0, 4.0, 97.0, 97.0, 57.0, 4.0, 97.0,
                4.0, 97.0, 4.0, 97.0, 4.0
            ],
            'count':
            418544
        },
        'action': {
            'mean': [
                0.01879958248456271, 0.031485158191750115, 0.07368794914814057,
                -0.0028184291499638585, -0.05094769448397053,
                -0.01966366534906301, -0.02781952276554756,
                -0.029985279712361083, 0.010903907964427336,
                0.018686267598934946, -0.008765613760300528,
                0.0589403524592044, -0.0417267438636165, 0.011222205547408125,
                0.002083319332670148, -0.003095515224305155,
                -0.013607547680973718, 0.035284611180845284,
                -0.00225087648638557, 0.029380998579454875,
                0.017783085184612624, 0.04295634897752905,
                -0.02501500577826991, -0.03065604371938277,
                0.03268137211302457, -0.03712941062459427,
                -0.01005799826363246, 0.0023408365519357395,
                0.01121333892349107, 0.007584350936355033,
                -0.010495338119159635, 7.384747223851716e-05,
                -2.221202711442865e-05, 0.9347245171092947, 0.9918086543820593,
                -0.00019284493553856322, 0.07828803535832911,
                3.333607285392755e-05, 0.9991646495154799,
                0.008132447195262646, 36.76136887228971,
                1.0940492611500424e-35, 95.36963454260484, 98.08544143507015,
                41.13597054847157, 1.0940492611500424e-35, 82.80273990398061,
                1.0940492611500424e-35, 82.80273990398061,
                1.0940492611500424e-35, 82.80273990398061,
                1.0940492611500424e-35
            ],
            'std': [
                0.10440701315834092, 0.0402029875309638, 0.0933800305736248,
                0.03757885029557996, 0.03334119120046179, 0.028094437409603757,
                0.10548569197189665, 0.04559907438171819, 0.11226583741102097,
                0.036567324971791826, 0.038193260777923184,
                0.025118322569778893, 0.05177016583370129,
                0.027123953977348302, 0.07050702615459783,
                0.046843018138980744, 0.054333986548949935, 0.2626104773207808,
                0.09208571532253064, 0.20026761204732677, 0.25484550718226784,
                0.2527027545679466, 0.20804369590145555, 0.11009673158916257,
                0.10615719337075386, 0.031268253766240116, 0.07065991081413243,
                0.054064533626289206, 0.03779192971388966,
                0.030026271330210626, 0.04835200516509691,
                0.0009232630227667171, 0.000621564165270335,
                0.024658758328447373, 0.01347995458427679, 0.00736814838262866,
                0.09975239727506895, 0.004305554195635772,
                0.0012784989908455433, 0.039795640096120494, 20.19800225888272,
                3.355337192912506e-32, 2.977141297154533, 0.2795374684004439,
                19.4036011317786, 3.355337192912506e-32, 35.46627714432199,
                3.355337192912506e-32, 35.46627714432199,
                3.355337192912506e-32, 35.46627714432199, 3.355337192912506e-32
            ],
            'min': [
                -0.6162942051887512, -0.21484142541885376, -0.5972588062286377,
                -0.30661895871162415, -0.24925139546394348,
                -0.13819850981235504, -0.6453017592430115, -0.2861673831939697,
                -0.782489538192749, -0.20442774891853333, -0.19455987215042114,
                -0.07232154905796051, -0.2772684097290039, -0.1146378368139267,
                -0.4535689949989319, -0.433417946100235, -0.45179688930511475,
                -1.2941067218780518, -0.5961405634880066, -1.1817448139190674,
                -1.2174170017242432, -1.5622127056121826, -1.2463995218276978,
                -0.7565920948982239, -0.6202925443649292, -0.30022746324539185,
                -0.6852419376373291, -0.4537838101387024, -1.0640697479248047,
                -0.5011018514633179, -0.4687747359275818,
                -0.011518796905875206, -0.007875248789787292,
                0.8940370082855225, 0.8342185020446777, -0.05620022118091583,
                -0.24666357040405273, -0.04920569807291031, 0.9801737666130066,
                -0.19804956018924713, 0.0, 0.0, 92.0, 98.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0
            ],
            'max': [
                0.8906533718109131, 0.24089300632476807, 0.6820911765098572,
                0.2557566165924072, 0.14124378561973572, 0.10757503658533096,
                0.7716604471206665, 0.16009223461151123, 0.5661941766738892,
                0.22574791312217712, 0.20955073833465576, 0.15735957026481628,
                0.26473093032836914, 0.20282840728759766, 0.4278891384601593,
                0.4444345533847809, 0.4347909092903137, 1.4488677978515625,
                0.6472907066345215, 1.141837239265442, 1.2274000644683838,
                1.4092843532562256, 1.3071191310882568, 0.8930418491363525,
                0.7644487619400024, 0.19302716851234436, 0.3673480153083801,
                0.4305163621902466, 0.6788275241851807, 0.3504282832145691,
                0.9978539943695068, 0.012269245460629463, 0.010671734809875488,
                1.078204870223999, 1.0, 0.05589602142572403, 0.551134467124939,
                0.05492148548364639, 1.0, 0.1653193086385727, 58.0,
                1.4508691105781216e-28, 98.0, 99.0, 58.0,
                1.4508691105781216e-28, 98.0, 1.4508691105781216e-28, 98.0,
                1.4508691105781216e-28, 98.0, 1.4508691105781216e-28
            ],
            'q01': [
                -0.23191335886716843, -0.06332188099622726,
                -0.20871078953146935, -0.10072427242994308,
                -0.1415131390094757, -0.07621277138590812, -0.2816150793433189,
                -0.1407223492860794, -0.3689217671751976, -0.07423130422830582,
                -0.10281371369957924, -0.0019709765911102295,
                -0.1733884531259537, -0.05124091446399689,
                -0.20340576767921448, -0.14121079444885254,
                -0.1895841658115387, -0.6232944732904434, -0.2614929085969925,
                -0.4483918058872223, -0.7151353979110717, -0.6661620211601258,
                -0.7207208275794983, -0.32497280836105347,
                -0.25276518523693087, -0.10889194905757904,
                -0.23525189235806465, -0.15126920223236084,
                -0.07737895846366882, -0.069888174533844, -0.12551138073205947,
                -0.002407002029940486, -0.0018575633876025677,
                0.9084766507148743, 0.9324967265129089, -0.020122891291975975,
                -0.11083248257637024, -0.011812308803200722,
                0.9937663674354553, -0.09316502511501312, 0.0, 0.0, 92.0, 98.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            'q99': [
                0.3245818054676062, 0.12473765030503275, 0.3505981370806711,
                0.08222569718956979, 0.025635869503021436, 0.04900759477168325,
                0.28776025801897054, 0.06395972520112991, 0.2564617997407914,
                0.1089927926659584, 0.07864519312977808, 0.11262276805937296,
                0.12208205595612548, 0.08439977467060089, 0.1841484785079961,
                0.12228039048612141, 0.12666195631027222, 0.8817144042253495,
                0.2613626149296765, 0.7181196212768555, 0.6071960371732719,
                0.9289278984069824, 0.567800105214119, 0.2653591355681423,
                0.33995115756988525, 0.047795194238424765, 0.16405910313129435,
                0.1410679817199707, 0.10788661301136027, 0.08360096991062194,
                0.13515613973140717, 0.003110955934971571,
                0.0018529229564592242, 1.0647358894348145, 0.9999969005584717,
                0.02221621200442314, 0.36070770025253296, 0.015861066058278084,
                0.9999992847442627, 0.1056414321064949, 58.0, 0.0, 98.0, 99.0,
                58.0, 0.0, 98.0, 0.0, 98.0, 0.0, 98.0, 0.0
            ],
            'count':
            20927200
        }
    }
}

_data_root = os.path.abspath(
    os.environ.get(
        'CANDY_DATA_ROOT',
        '/mnt/data/cpfs/limx_embmc/VLA_Data/Fixed-Feet-Mani/hf_cache/'
        'lerobot/lerobot'))
_candy_data_roots = [
    os.path.join(_data_root, name) for name in [
        '0611_2_candy_subtask_delta_base_relabel_2_v21',
        '0612_candy_subtask_delta_base_relabel_2_v21',
        '0616_candy_subtask_delta_base_relabel_2_v21',
        '0618_candy_subtask_delta_base_relabel_2_v21',
        '0622_candy_subtask_delta_base_relabel_2_v21',
        '0623_candy_subtask_delta_base_relabel_2_v21',
        '0624_candy_subtask_delta_base_relabel_2_v21',
        '0709_candy_subtask_delta_base_relabel_2_v21',
    ]
]

_state_dim = 43
_action_dim = 52
_model_action_dim = 64
_action_horizon = 50
_statistic_name = 'private'
_delta_action_mask = [True] * 31

_per_device_batch_size = 8
_grad_accumulation_steps = 2
_max_steps = 20_000

_task_prompts = {
    '1': ('pick up the white candy and place it in the left section of the '
          'snack tray with left arm'),
    '2': ('pick up the purple candy and place it in the right section of the '
          'snack tray with left arm'),
    '3': ('pick up the red candy and place it in the middle section of the '
          'snack tray with left arm'),
}

model = dict(
    type='PI05FlowMatching',
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
    projector=dict(type='LinearProjector', in_dim=1152, out_dim=2048),
    proj_width=1024,
    n_action_steps=_action_horizon,
    action_in_proj=dict(
        type='LinearProjector', in_dim=_model_action_dim, out_dim=1024),
    action_out_proj=dict(
        type='LinearProjector', in_dim=1024, out_dim=_model_action_dim),
    time_mlp_in=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    time_mlp_out=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    time_sampler='beta',
    time_beta_alpha=1.5,
    time_beta_beta=1.0,
    openpi_fp32_flow=True,
    max_action_dim=_model_action_dim,
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
    freeze_llm_backbone=False,
    freeze_vision_backbone=False,
    pretrained_name_or_path='./checkpoints/pi05_base/model.safetensors',
    name_mapping={
        'llm_backbone': 'paligemma_with_expert.paligemma.model.language_model',
        'vision_backbone.vision':
        'paligemma_with_expert.paligemma.model.vision_tower',
        'projector.projector':
        'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',
        'llm_expert': 'paligemma_with_expert.gemma_expert.model',
        'time_mlp_in.projector': 'time_mlp_in',
        'time_mlp_out.projector': 'time_mlp_out',
        # The base action projectors are 32-D. Omit both mappings so the
        # complete 64-D Candy projectors, including their biases, are newly
        # initialized instead of partially loading shape-compatible tensors.
        'llm_backbone.embed_tokens': 'paligemma_with_expert.paligemma.lm_head',
        'llm_expert.embed_tokens':
        'paligemma_with_expert.gemma_expert.lm_head',
    },
    params_to_change_dtype=[
        'llm_expert.llm.model.layers',
        'vlm_backbone.vlm.model.language_model.layers',
        'vlm_backbone.vlm.model.vision_tower',
        'vlm_backbone.vlm.model.multi_modal_projector',
    ],
    ori_action_dim=_action_dim,
    # Supervise all padded model dimensions, as in OpenPI.
    loss_action_dim=_model_action_dim,
)

inference_model = model.copy()

_train_transforms = [
    dict(
        type='ProcessParquetInputs',
        parquet_keys=[
            'observation.state', 'timestamp', 'actions', 'info', 'stats',
            'action_masks'
        ],
        video_keys=[
            'observation.images.head',
            'observation.images.left_wrist',
        ],
        name_mappings={
            'observation.state': ['states'],
            'actions': ['actions'],
        }),
    dict(type='RelativeActions', mask=_delta_action_mask),
    dict(
        type='NormalizeStatesAndActions',
        action_dim=None,
        state_dim=None,
        state_key='proprio',
        action_key='action',
        norm_type='quantile',
        output_dtype='float32'),
    dict(type='PreparePromptWithState'),
    dict(
        type='ProcessPrompts',
        max_len=200,
        tokenizer=dict(
            type='PretrainedTokenizer', model_path='checkpoints/pi05_base')),
    dict(type='PadStatesAndActions', model_action_dim=_model_action_dim),
    dict(type='ResizeImagesWithPad', height=224, width=224, backend='pil'),
    dict(type='SimpleNormalizeImages'),
    dict(type='OpenPIImageAugment', base_camera_indices=(0, )),
]

train_dataloader = dict(
    per_device_batch_size=_per_device_batch_size,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        reshuffle_each_epoch=True,
        dataset_statistics=_PI05_CANDY_STATS,
        statistic_name=_statistic_name,
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'action', 'timestamp'],
        datasets=dict(
            type='ParquetDataset',
            data_root_path=_candy_data_roots,
            action_key='action',
            transforms=_train_transforms,
            action_window_size=_action_horizon,
            window_start_idx=0,
            supervise_terminal_padding=True)))

runner = dict(
    type='FSDPTrainRunner',
    # 8 samples/GPU x 8 GPUs x 2 accumulation steps gives global batch 128;
    # 20,000 steps cover the 418,544-frame dataset about 6.12 times.
    max_steps=_max_steps,
    grad_accumulation_steps=_grad_accumulation_steps,
    ema_decay=0.99,
    seed=42,
    optimizer=dict(
        type='AdamW',
        lr=2.5e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        weight_decay_all_params=True,
        foreach=False,
        fused=True,
    ),
    max_grad_norm=1.0,
    sharding_strategy='global-shard-grad-op',
    fsdp_wrap_policy='execution-block',
    reduce_in_full_precision=True,
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'timestamp', 'images', 'img_masks', 'lang_tokens',
            'lang_masks', 'actions', 'action_masks'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    sampler=None,
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        schedule_style='openpi',
        warmup_steps=1000,
        decay_steps=30000,
        min_lr=2.5e-6),
    tokenizer=dict(
        type='PretrainedTokenizer', model_path='checkpoints/pi05_base'),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        window_size=100),
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=True,
    change_key_name=False)

inference = dict(
    type='OliInferenceRunner',
    seed=7,
    state_dim=_state_dim,
    action_chunk=_action_horizon,
    publish_rate=30,
    max_publish_step=0,
    interactive=True,
    default_prompt_id='1',
    default_execution_count=1,
    keep_params_fp32=True,
    mixed_precision_dtype='bf16',
    camera_names=['head', 'left_wrist'],
    task_descriptions=_task_prompts,
    dataset=dict(
        type='PrivateInferenceDataset',
        statistic_name=_statistic_name,
        img_keys=['head', 'left_wrist'],
        transforms=[
            dict(
                type='NormalizeStatesAndActions',
                action_dim=None,
                state_dim=None,
                state_key='proprio',
                action_key='action',
                norm_type='quantile',
                output_dtype='float32'),
            dict(type='PreparePromptWithState'),
            dict(
                type='ProcessPrompts',
                max_len=200,
                tokenizer=dict(type='PretrainedTokenizer')),
            dict(
                type='PadStatesAndActions',
                model_action_dim=_model_action_dim),
            dict(
                type='ResizeImagesWithPad',
                height=224,
                width=224,
                backend='pil'),
            dict(type='SimpleNormalizeImages'),
        ]),
    denormalize_action=dict(
        type='DenormalizeDeltaAction',
        statistic_name=_statistic_name,
        action_dim=_action_dim,
        norm_type='quantile',
        delta_action_mask=_delta_action_mask),
    operator=dict(
        type='OliOperator',
        control_backend='mros',
        hand_mode='finger',
        head_rgb_topic='/head/color/image_raw/compressed',
        left_wrist_rgb_topic=('/left_wrist_camera/color/image_raw/compressed'),
        joint_state_topic='/joint/state',
        finger_state_topic='/brainco1/hand/state',
        finger_cmd_topic='/brainco1/hand/cmd',
        finger_force_levels=(2.0, 2.0),
        teleop_wbt_topic='/teleop_cmd_WBT'))
