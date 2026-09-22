# Experimental EDM Heun sampler for packed Boltz-2

Use `--diffusion_solver heun --step_scale 1.0 --no_contact_guidance` with `--batch_size 4 --batch_layout packed`. `--sampling_steps` and `--sampling_steps_affinity` independently select the step counts for structure and affinity. The default solver remains Euler with the existing Boltz-2 step scale of 1.5.

Heun makes an Euler proposal, evaluates the denoiser at the next positive noise level, and averages the two slopes. The zero-noise terminal step uses Euler only, giving 2N-1 denoiser evaluations per sample for N steps. See the [EDM reference algorithm](https://github.com/NVlabs/edm/blob/main/generate.py).

This implementation retains the Boltz stochastic schedule, churn/noise settings, per-record RNG, native-padding centering, mixed precision policy, and packed conditioning. It uses the standard Heun unit step multiplier; consequently a comparison with default Euler changes that multiplier as well as integration order and (when requested) the step count. This is an adaptation of the EDM integrator to Boltz, not a replacement with all of NVIDIA's image-sampling defaults.

The corrector receives no additional random rotation or noise. With reverse-diffusion alignment enabled, its denoised coordinates are aligned into the proposal frame before averaging slopes. Potential guidance is not supported with Heun and is rejected explicitly.

The experiment compares 50 and 100 steps in both stages, seeds 42 and 43, using the saved 87-case inputs and original checkpoint files. Structure kernels are enabled and affinity kernels disabled through the experiment harness, matching the saved Euler baselines. A four-case functional smoke test precedes the full runs. Reduced steps do not guarantee preserved prediction accuracy; benchmark results must be evaluated before adopting Heun.

The benchmark YAMLs have no contact or pocket constraints. Contact guidance is explicitly disabled for Heun; it contributes no constraint forces to these baseline inputs.

For installation, branch lineage, commands and publication validation, see [Native batching and EDM Heun branches](native_batching.md).
