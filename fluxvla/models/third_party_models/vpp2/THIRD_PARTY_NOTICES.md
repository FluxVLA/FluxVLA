# Third-party notices

## Source and local changes

The numerical modules are extracted from
[VPP2](https://github.com/roboterax/video-prediction-policy-2) at revision
`3365872910244f14df83003062f7c563f14c6d9e` (the XPolicyLab evaluation pin).
Logging imports use the standard library; numerical implementations and parameter
names are retained. The FluxVLA head ports cached-video Euler inference and
reuses FluxVLA DreamZero VAE and T5 modules. Experimental routes are not exposed.
Native Python configs own the model and inference settings; FluxThemis owns
simulator assets and evaluation orchestration.

## Licenses

Complete license texts and copyright notices are retained in [LICENSE](LICENSE).
The identical Apache-2.0 text is included once for all listed Apache components.

- VPP2 code derived from FastWAM retains the MIT notice, copyright 2026
  The FastWAM Authors.
- Wan components derive from [Wan2.1](https://github.com/Wan-Video/Wan2.1) and
  [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), under
  Apache-2.0.
- [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) reference materials
  retain the MIT notice. The simulator and its assets are not bundled here.
- The Apache-2.0 notice for the
  [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) reference evaluation
  interface is retained. XPolicyLab is not a runtime dependency of the native
  FluxVLA integration.

Model weights and datasets are distributed separately under their own terms;
these code licenses do not assign licenses to those artifacts.
