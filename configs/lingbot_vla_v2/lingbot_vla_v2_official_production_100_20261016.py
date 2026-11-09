"""100-step replay of the official LingBot production sampler trace."""

_base_ = ["./lingbot_vla_v2_official_trace_parity_20261016.py"]

train_dataloader = dict(
    dataset=dict(
        manifest_path=(
            "/ceph/teamGolden/andyzeng/VLA/FluxVLA/configs/lingbot_vla_v2/"
            "official_captured_batch_manifest_20261020.json"
        ),
        require_flow=True,
        replay_order="official_step_rank",
    )
)

runner = dict(
    max_steps=100,
    fsdp_api_version=2,
    # The official production trace uses AdamW with a sqrt(32/4) LR multiplier
    # for routed expert tensors.  Keep this exact contract for parity; Muon is
    # available as a separate FluxVLA optimizer but is not this baseline.
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        foreach=True,
        fused=False,
    ),
    # Official enable_fp32 applies an FP32 policy only to action/basic FSDP
    # units. Qwen VLM layers and the root use BF16 forward parameters with
    # FP32 reductions, and the native training loop does not add a blanket
    # autocast around the model. This layered policy is required for parity:
    # root projections emit BF16 while FixAdaRMSNorm gamma/beta stay FP32.
    enable_mixed_precision_training=True,
    enable_forward_autocast=False,
    batch_dtype="bf16",
    reduce_in_full_precision=True,
    keep_params_fp32=False,
)

# Match the actual official trace run: Qwen VLM/vision attention uses
# flash-attention-2, while the LingBot custom action block was run with eager
# attention (verified in official_prod_action_trace_20261019/console.log).
model = dict(
    # ``train.enable_fp32`` in the official runner controls FSDP parameter
    # storage/reduction.  It is independent from the model-level
    # ``action_fp32`` input contract; the official production config records
    # the latter as false.  Keep it explicit here because the dense base
    # recipe defaults to true for a different alignment experiment.
    action_fp32=False,
    attention_implementation="eager",
    vit_attn_implementation="flash_attention_2",
    attention=dict(
        implementation="eager",
        vit_implementation="flash_attention_2",
    ),
)
