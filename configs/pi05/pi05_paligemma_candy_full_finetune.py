# Copyright 2026 Limx Dynamics
"""Standalone Boris-style PI0.5 Candy training and inference config.

This keeps the absolute-action, 32-step, 52-D-loss recipe from commit 3c4a92e
while using the current synchronous Oli runner/operator for deployment.
"""

import os

# Offline statistics for the absolute-action, 32-step Boris recipe.
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
                -0.16593376747366936, 0.06731787850898317,
                0.032734205740801846, 0.0819476938168803, -0.04690051838956029,
                -0.07627509974812081, -0.20862642128513484,
                -0.06836387652295346, -0.04745830411663582,
                0.13766230351088002, -0.04246043999596626, 0.15608557199229353,
                -0.06321735373549703, 0.013352108768516899, 0.1830255152101398,
                0.03270046499900987, 0.41601941166098577, -0.4774277314982457,
                0.33401353340819995, -0.7786349103373645, -0.9158302761089759,
                -0.4311277259929304, -0.08004565530715745, 0.3232993494829889,
                0.09489749876220334, -0.28639109834015075, 0.5030927116440708,
                -1.5170418395036762, 0.016899797966521996, -0.5031406434476745,
                -0.37132229909925396, 6.319942005832135e-05,
                -2.2728523519407997e-05, 0.9346772238142651,
                0.9916707226876642, -0.00011527933908301802,
                0.08024604326308918, -6.828599966137075e-05,
                0.9991558937758112, 0.008373980797576164, 37.39873156393256,
                1.7094519705469412e-35, 95.32348704676211, 98.08544143507015,
                41.94560518418348, 1.7094519705469412e-35, 84.6946351860696,
                1.7094519705469412e-35, 84.6946351860696,
                1.7094519705469412e-35, 84.6946351860696,
                1.7094519705469412e-35
            ],
            'std': [
                0.1656902248294647, 0.06043097500866077, 0.15871763570180122,
                0.07206933484948286, 0.056096775082872044,
                0.037117801347553284, 0.1633246208689796, 0.07434406604667874,
                0.1525005492347097, 0.07256140401257292, 0.05353464893791786,
                0.04352926294011982, 0.07136002383456894, 0.03999260879650067,
                0.11145643764369731, 0.08135250687740465, 0.11345426785382669,
                0.3239287242918284, 0.14210686555443372, 0.2398402772251969,
                0.27366658918553516, 0.2924490507270764, 0.2725474892616082,
                0.1595821239882591, 0.18034550463358007, 0.04850274766918864,
                0.1250593210195213, 0.16177287987223715, 0.1766623537669325,
                0.14734114711315582, 0.16434702862352032,
                0.0009255587955810431, 0.0006184511791860502,
                0.02467831283333291, 0.013638401008419588,
                0.007375629574076004, 0.09954562386292007,
                0.0042382566674122974, 0.0012899722529529742,
                0.039971801086837795, 19.57417290474705,
                4.1941713712105964e-32, 2.982508362381877, 0.2795374684067862,
                18.541095132848916, 4.1941713712105964e-32, 33.5614572563538,
                4.1941713712105964e-32, 33.5614572563538,
                4.1941713712105964e-32, 33.5614572563538,
                4.1941713712105964e-32
            ],
            'min': [
                -0.8924256563186646, -0.2061503678560257, -0.5120874047279358,
                -0.08699999749660492, -0.24533669650554657,
                -0.18944460153579712, -0.8983064889907837, -0.2951371669769287,
                -0.6981316804885864, -0.0766911506652832, -0.2199677675962448,
                0.03914107754826546, -0.3564435541629791, -0.13554328680038452,
                -0.12085013836622238, -0.2875482141971588,
                -0.030242495238780975, -1.8335720300674438,
                -0.08699999749660492, -1.5707963705062866, -1.813338279724121,
                -1.3100675344467163, -1.0540428161621094, -0.26207154989242554,
                -0.5323126912117004, -0.5215526819229126, -0.07577826827764511,
                -1.9524707794189453, -0.7850000262260437, -0.9562350511550903,
                -1.0829637050628662, -0.011518796905875206,
                -0.007875248789787292, 0.8940370082855225, 0.8342185020446777,
                -0.05620022118091583, -0.24666357040405273,
                -0.04920569807291031, 0.9801737666130066, -0.19804956018924713,
                0.0, 0.0, 92.0, 98.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            'max': [
                0.39339128136634827, 0.2929586172103882, 0.6768683791160583,
                0.48707762360572815, 0.161988765001297, 0.05870726332068443,
                0.3189982771873474, 0.19712494313716888, 0.5450605154037476,
                0.3972774147987366, 0.17296652495861053, 0.2786688506603241,
                0.17422601580619812, 0.19024710357189178, 0.6149630546569824,
                0.44001680612564087, 0.7850000262260437, 0.4317115843296051,
                1.0322153568267822, -0.04774332419037819, -0.10234005749225616,
                0.7849844694137573, 0.8011341094970703, 0.949849545955658,
                0.6284026503562927, -0.061088964343070984, 0.9557101130485535,
                -0.6341454982757568, 0.6147943735122681, 0.15925033390522003,
                0.3948856592178345, 0.012269245460629463, 0.010671734809875488,
                1.078204870223999, 1.0, 0.05589602142572403, 0.551134467124939,
                0.05492148548364639, 1.0, 0.1653193086385727, 58.0,
                1.4508691105781216e-28, 98.0, 99.0, 58.0,
                1.4508691105781216e-28, 98.0, 1.4508691105781216e-28, 98.0,
                1.4508691105781216e-28, 98.0, 1.4508691105781216e-28
            ],
            'q01': [
                -0.5931856632232666, -0.07767023891210556,
                -0.32756465673446655, -0.051300935447216034,
                -0.18027204275131226, -0.15032091736793518,
                -0.6527754068374634, -0.2298010140657425, -0.41203582286834717,
                -0.003256267635151744, -0.1577462904155254,
                0.06971298158168793, -0.26697981357574463,
                -0.07216550409793854, -0.026669228449463844,
                -0.15848477184772491, 0.1453673243522644, -1.1903762817382812,
                -0.0210605226457119, -1.3357057571411133, -1.583120346069336,
                -0.9350749254226685, -0.8804702758789062,
                -0.044217489659786224, -0.33649948239326477,
                -0.40561896562576294, 0.1955399513244629, -1.8347303867340088,
                -0.3425480902194974, -0.7899783849716187, -0.8264527916908264,
                -0.0024579386226832867, -0.0018460102146491408,
                0.9084151983261108, 0.931782066822052, -0.020045317709445953,
                -0.11013659834861755, -0.011873394250869751,
                0.9937247037887573, -0.09324877709150314, 0.0, 0.0, 92.0, 98.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            'q99': [
                0.16256889700889587, 0.19393233954906464, 0.4316217005252838,
                0.28578561544418335, 0.08758959919214249, 0.011919879354536533,
                0.11808160692453384, 0.10939281433820724, 0.34534308314323425,
                0.3253154158592224, 0.0896063894033432, 0.24860410392284393,
                0.07690957188606262, 0.11195296794176102, 0.4561415910720825,
                0.25171124935150146, 0.6613617539405823, 0.2248266488313675,
                0.7054608464241028, -0.2884508967399597, -0.39336174726486206,
                0.3759000897407532, 0.4260692000389099, 0.6968808174133301,
                0.4426094591617584, -0.17556899785995483, 0.7815229892730713,
                -1.1157784461975098, 0.4577501118183136, -0.1289283186197281,
                -0.05541282892227173, 0.003085657022893429,
                0.0018486010376363993, 1.0648120641708374, 0.9999969005584717,
                0.022331755608320236, 0.3626423478126526, 0.015622648410499096,
                0.9999992251396179, 0.10593197494745255, 58.0, 0.0, 98.0, 99.0,
                58.0, 0.0, 98.0, 0.0, 98.0, 0.0, 98.0, 0.0
            ],
            'count':
            13393408
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
_action_horizon = 32
_statistic_name = 'private'

_per_device_batch_size = 8
_grad_accumulation_steps = 2
_max_steps = 130_800

_init_checkpoint = os.path.abspath(
    os.environ.get('PI05_CANDY_INIT_CHECKPOINT',
                   './checkpoints/pi05_base_action64/model.safetensors'))

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
    pretrained_name_or_path=_init_checkpoint,
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
    params_to_change_dtype=[
        'llm_expert.llm.model.layers',
        'vlm_backbone.vlm.model.language_model.layers',
        'vlm_backbone.vlm.model.vision_tower',
        'vlm_backbone.vlm.model.multi_modal_projector',
    ],
    ori_action_dim=_action_dim,
    # Boris supervises only the 52 semantic dimensions; 52:64 is padding.
    loss_action_dim=_action_dim,
    zero_padded_action_dims=True,
    trim_action_prediction=True,
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
    dict(
        type='NormalizeStatesAndActions',
        action_dim=None,
        state_dim=None,
        state_key='proprio',
        action_key='action',
        norm_type='quantile',
        discrete_state_dims=list(range(31, 43)),
        discrete_action_dims=list(range(40, 52)),
        discrete_norm_type='min_max',
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
    # 130,800 steps cover the 418,544-frame dataset about 40 epochs.
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
    save_iter_interval=_max_steps // 5,
    max_keep_ckpts=3,
    enable_gradient_checkpointing=True,
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
                discrete_state_dims=list(range(31, 43)),
                discrete_action_dims=list(range(40, 52)),
                discrete_norm_type='min_max',
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
        type='DenormalizePrivateAction',
        statistic_name=_statistic_name,
        action_dim=_action_dim,
        norm_type='quantile',
        discrete_action_dims=list(range(40, 52)),
        discrete_norm_type='min_max'),
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
