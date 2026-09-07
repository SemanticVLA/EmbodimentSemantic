# Automatic Recovery Supervision for VLA Adaptation from a Zero-Shot Start

This plan builds **automatic adaptation from a zero-shot start**, with the arrow controller supplying the corrections that RoboTTT obtains from humans.

Two facts shape the plan:

- Automatically generated trajectories are still supervision. Attempt 1 can be zero-shot; attempts after training measure adaptation. Those must be reported separately.
- **Exact RoboTTT reuse is currently blocked by missing artifacts.** The paper describes an Eagle-based GR00T N1.7; NVIDIA's public N1.7 now uses Cosmos-Reason2. The official sources checked do not provide a RoboTTT checkpoint or implementation exposing the missing training details. Using the public model without documenting this difference would violate the exact-reuse requirement. [Paper Appendix A](https://arxiv.org/html/2607.15275v1#A1), [public model card](https://huggingface.co/nvidia/GR00T-N1.7-3B).

Published settings below are identified as published; new experimental decisions are specified explicitly. Availability findings reflect the source inspection on 2026-09-07.

## 1. The research question and paper motivation

The question is:

> Can a robot turn its own failed attempts into training data, using an automatic geometric controller for corrections, and subsequently complete new episodes without that controller?

RoboTTT supplies a mechanism for learning from failure-and-correction histories. Its DAgger experiment still depends on human corrective actions. Our controller could remove the human from that data-collection loop.

However, simply replacing a human teacher with a controller is not a sufficient methodological contribution. DAgger already addresses training on states induced by the learner, and VLA-OPD already studies expert supervision on learner-generated trajectories. [DAgger](https://proceedings.mlr.press/v15/ross11a.html), [VLA-OPD](https://arxiv.org/abs/2603.26666).

The proposed contribution is more specific:

> An automatic controller supplies successful corrections from states produced by a failing VLA. Training preserves the failed history as context, supervises only the corrections, and produces measurable improvements in subsequent VLA-only execution.

The experiments must establish three claims separately:

1. **Automatic supervision works:** the controller produces usable corrections from actual VLA failure states.
2. **Failure context matters:** retaining the failed prefix improves learning beyond training on the same corrective actions alone.
3. **The VLA acquires the improvement:** success increases when the controller, arrows, bounding boxes, and evaluator feedback are unavailable to the VLA.

The paper's central figure should therefore show **VLA-only success against accumulated automatic intervention cost**, with matched correction-only and shuffled-history baselines.

## 2. Define zero-shot before selecting a checkpoint

Use three distinct data partitions:

- **Source:** any data used to pretrain the VLA, train its TTT layers, or establish its robot action interface.
- **Adaptation:** target episodes on which the VLA attempts the task and can receive automatic corrections.
- **Query:** episodes used only to measure performance. They never supply training data.

The zero-shot checkpoint must satisfy all of these conditions:

- No adaptation or query demonstrations were used in its task-specific training.
- Its TTT layers were already trained on source trajectories; they are not randomly inserted immediately before evaluation.
- Its robot state encoder, action decoder, and normalization support the evaluation robot.
- Available training provenance supports the claimed task exclusion.

Existing LIBERO-finetuned SmolVLA, Pi0.5, and OpenVLA checkpoints can validate the software integration. They cannot establish strict LIBERO task zero-shot performance.

If foundation pretraining cannot be audited, the permitted claim is **“no target-specific fine-tuning before deployment.”** We must not upgrade that to proven task zero-shot.

Freeze the controller, model, prompts, thresholds, and training settings before generating the scored target splits. Existing controller tuning is part of its prior knowledge and must be disclosed.

## 3. Establish the exact RoboTTT implementation before training

Create a fidelity manifest mapping each requirement to:

```text
paper section → source symbol → configuration value → checkpoint tensor → verification test
```

The primary experiment requires the paper's implementation and a source-trained checkpoint before target-task post-training. The manifest must resolve:

- Exact backbone and checkpoint revision.
- Fast-MLP dimensions, biases, and normalization.
- Q/K/V projection dimensions.
- Learned inner-learning-rate parameterization.
- Action horizon H, chunk stride, and denoising steps k.
- Fast-state handling across denoising iterations.
- TBPTT segment length and detach locations.
- Sequence cropping and intervention-boundary masking.
- Masked-loss reduction.
- Optimizer betas, epsilon, clipping, and scheduler boundaries.
- Pretraining mixture and context curriculum.

The checked paper does not specify all of these. The primary training launcher must reject an unresolved manifest. It must not silently substitute a public same-name checkpoint or guessed settings.

The simulator coordinator, recorder, and controller handoff can be implemented while this artifact dependency remains unresolved.

## 4. Preserve the two separate learning mechanisms

Let $\psi$ denote the ordinary VLA parameters, $\phi$ the TTT projections and other learned configuration, $W_0$ the learned fast-weight initialization, and $W_t$ the episode's working fast weights.

The published inner update is:

$$
\mathcal L_{\mathrm{FW},t}
=
\left\|f_{W_{t-1}}(K_t)-V_t\right\|_2^2,
$$

$$
W_t
=
W_{t-1}
-
\eta_t\nabla_W\mathcal L_{\mathrm{FW},t},
\qquad
O_t=f_{W_t}(Q_t).
$$

This is the update used during execution. The teacher's action loss must not be substituted for this key–value loss.

The published action-training objective uses:

$$
A_t^{\tau_t}
=
\tau_t A_t+(1-\tau_t)\epsilon_t,
$$

$$
\ell_t
=
\left\|
v_{\psi,\phi}
\left(\Phi_t,A_t^{\tau_t},q_t;W_{t-1}\right)
-
(A_t-\epsilon_t)
\right\|_2^2.
$$

For controller-generated correction sequences, apply the paper's supervision asymmetry:

$$
m_t =
\begin{cases}
0 & \text{VLA-generated context},\\
1 & \text{accepted controller correction}.
\end{cases}
$$

The correction objective is the masked sequence flow-matching loss. Its exact reduction and padding behavior must come from the reference implementation.

Use independent noise levels per action chunk:

$$
\tau_t=0.999(1-u_t),
\qquad
u_t\sim\operatorname{Beta}(1.5,1).
$$

The parameter-update contract is:

| Parameter group | Source sequence pretraining | Correction training | Robot execution |
|---|---|---|---|
| Original VLA parameters $\psi$ | Frozen | Trainable | Frozen |
| Added TTT projections and learned configuration $\phi$ | Trainable | Trainable | Frozen |
| Learned initialization $W_0$ | Meta-trained | Trainable through the permitted unroll | Used to initialize $W$ |
| Episode fast weights $W_t$ | Inner updates | Inner updates | Inner updates only |

These are the published stage-level freezing rules. Exact tensor membership must be checked against the reference code. [RoboTTT method and training details](https://arxiv.org/html/2607.15275v1#S3).

Implementation requirements:

- Preserve the two-layer GeLU fast model, register-token pathway, and gated residual.
- Differentiate through inner updates during outer training.
- Carry fast-weight values across TBPTT boundaries while detaching their gradients.
- Reset episode fast weights to the current learned $W_0$ at every new evaluation episode.
- Never use the existing action-only or LoRA training recipe as an unmarked replacement.

A critical gradient test is required: when the first TBPTT segment contains only masked failure context, $W_0$ may receive no correction gradient through that segment. We must reproduce and report that behavior rather than silently changing the unroll to force a desired gradient.

## 5. Build one coordinator that owns the failed state

Implement the experiment under a new `automatic_ttt` package in the LIBERO subsystem.

The current [VLA rollout](/C:/Users/hassa/OneDrive/Desktop/EmbodimentSemantic/vla_benchmarking/libero/evaluation/libero_policy_rollout.py:212) closes its environment when it finishes. The reusable controller setup is inside the [controller runner](/C:/Users/hassa/OneDrive/Desktop/EmbodimentSemantic/vla_benchmarking/libero/arrow_grasp_controller/controller/runner.py:1987). Running their separate commands would not preserve the failure state.

The new coordinator must execute this sequence:

1. Construct and settle one environment.
2. Initialize the VLA's episode state.
3. Allow **220 accepted VLA control actions**, matching the current shared evaluation budget.
4. Stop early if the official task succeeds.
5. If the budget expires without success and the simulator remains valid, discard the unexecuted remainder of the VLA's action chunk.
6. Preserve the simulator instance and its physical state.
7. Invoke the recovery teacher with a maximum of **1,200 additional actions**, including preparation and retries.
8. Record the outcome and close the environment once.

Configure collection episodes to support at least the combined 1,420-action budget. The 220-action handoff is a coordinator event, not an environment termination.

For the first version, failure means **“no task success within 220 actions.”** There is no confidence-based or visually guessed early-failure detector. A genuinely terminated, truncated, or invalid simulator does not receive further actions.

The official evaluator decides success and whether collection needs a teacher. Its values never enter the VLA's observations.

## 6. Make the controller a valid takeover teacher

The existing controller was designed around its own prepared start state. Its initial gripper opening is unsafe as a general takeover operation: the VLA may already be holding the bowl.

Implement an explicit takeover classifier with these outcomes:

- **Correct source held:** use a placement-only continuation. Preserve the grasp, estimate the destination from fresh RGB-D, and execute validated transfer, release, and retreat.
- **No object held:** prepare the gripper and invoke the canonical grasp-and-place routine after validating clearance from the current pose.
- **Wrong object held:** record a teacher failure in version 1; do not open the gripper at an arbitrary location.
- **Unobservable target, unsafe path, invalid state, or uncertain holding state:** record a teacher failure without executing a guessed recovery.

For the simulation study, held-object classification can use simulator contacts. Record that explicitly as additional teacher privilege. It must not be described as a capability already present in the canonical controller.

The teacher interface should be:

```python
recover(
    live_env,
    task_spec,
    recorder,
    action_budget=1200,
) -> RecoveryResult
```

It must never reset, teleport objects, restore an earlier scene, or close the environment.

Give the takeover-capable controller a new version identity. The historical controller success rate does not establish its recovery rate on VLA-induced states.

## 7. Record actual transitions and construct valid training sequences

Record every accepted environment action as:

```text
episode_id, task_id, layout_hash, simulator_step
observation_before
executed_action
observation_after
actor: vla | teacher
teacher_phase, teacher_attempt_id
terminated, truncated, task_success
model_hash, controller_hash, processor_hash
```

The observation contains synchronized clean external-camera RGB, wrist RGB, and raw robot state. Store depth, arrows, boxes, contact-state labels, and evaluator information in teacher-only sidecars.

The recorder sits at `env.step`, so it records actions that actually executed. It must not infer actions from waypoints or reuse unexecuted VLA proposals.

Export through the selected model's verified processor:

- Validate the seven-dimensional LIBERO OSC action convention, scale, frame, and gripper sign.
- Preserve raw state and apply the model's declared state encoding.
- Apply image orientation and resizing exactly once.
- Reconstruct action chunks from contiguous executed transitions.
- Mask padding and actor boundaries.
- Never concatenate a fresh reset trajectory onto a failed prefix as if execution were continuous.

The current [episode trace](/C:/Users/hassa/OneDrive/Desktop/EmbodimentSemantic/vla_benchmarking/libero/evaluation/arrow_episode_trace.py:93) supplies useful persistence patterns, but it is not already a complete synchronized training dataset.

For the primary protocol, accept supervision only from intervention episodes where the teacher reaches official task success. Within those episodes, teacher actions receive correction masks and VLA actions remain context.

The RoboTTT DAgger Distillation experiment used a pooled **100 trajectories**:
50 collected with RoboTTT as the base policy and 50 with GR00T N1.7. This is a
trajectory count, not a fixed number of individual correction actions. Our
automatic-teacher analogue therefore targets **100 accepted Arrow-corrected
trajectories per adaptation round**; the 20-attempts-per-task schedule below is
the collection budget, and unrecovered attempts are not silently replaced.

Every candidate is passed through the package's strict demonstration validator
before replay. It requires a contiguous VLA prefix followed by captured Arrow
corrections on the same live environment, exact observation continuity at the
handoff and between steps, complete action chunks, valid action ranges, explicit
held/unheld source state, and an evaluator-confirmed successful final teacher
transition. Invalid or unsuccessful traces stay in the raw archive but cannot
become positive training demonstrations.

Keep unsuccessful teacher episodes in the raw archive and collection denominator. Do not label them successful or collect additional episodes merely to replace them.

This success filter is our explicit automatic-teacher quality rule; it is not attributed to RoboTTT.

## 8. Specify the automatic adaptation experiment

Use LIBERO first. The initial integration canary is task 0, seed 1000, with a separate unscored artifact identity. It verifies execution and recording, not learning.

The proposed scored protocol is:

- **Adaptation tasks:** IDs 0, 2, 4, 6, and 8.
- **Transfer tasks:** IDs 1, 3, 5, 7, and 9.
- **Independent adaptation runs:** three, with RNG seeds 17, 29, and 43.
- **Adaptation rounds:** three.
- **Collection per round:** 20 attempted episodes per adaptation task—100 attempts per round.
- **Query set:** 50 fixed episodes per task, evaluated at checkpoints 0, 1, 2, and 3.
- **Checkpoint selection:** always report the predetermined round checkpoints; no query-based best-checkpoint selection.

Generate and seal disjoint adaptation/query layouts. Check their initial-state and layout hashes, not just seed numbers: different seeds can select the same underlying initialization.

The implementation now treats an observation hash as insufficient for this lock:
the concrete host must provide a simulator-state digest or replay key plus the
student-observation digest. A scored protocol lock therefore covers all 1,500
query reset identities (10 tasks × 3 adaptation seeds × 50 episodes). The task-0,
seed-1000 canary is explicitly `pilot`/unscored and cannot be used to produce a
headline improvement claim.

The transfer tasks never contribute controller trajectories. Their results measure transfer across held-out LIBERO task configurations; these tasks share a bowl-placement skill family, so this is not evidence for acquiring arbitrary unseen manipulation skills.

Each adaptation round performs:

```text
collect fixed 100 attempts
    → recover failed attempts automatically
    → append eligible intervention sequences to cumulative replay
    → train using the correction-masked RoboTTT objective
    → save a new checkpoint
    → evaluate VLA alone
```

No existing target-task demonstrations enter replay.

The proposed update budget is **20,000 outer optimizer steps per round**, using 1K maximum training context and effective sequence batch size 8. Use the published post-training AdamW peak learning rate $5\times10^{-5}$, weight decay $10^{-5}$, and cosine schedule; obtain the remaining optimizer/scheduler settings from the fidelity manifest.

Twenty thousand steps **per automatic round** is our proposed deployment protocol. The paper reports that duration for task post-training; it does not establish this repeated online schedule.

Use cumulative replay over accepted intervention episodes. Resolve window sampling through the fidelity manifest and log how often a supervised correction retains its failure boundary. If no eligible correction exists, skip the update and retain the previous checkpoint.

During each rollout, slow parameters stay fixed. Persistent slow-parameter updates happen only between rounds. After loading a new checkpoint, initialize a fresh fast state.

## 9. Run comparisons that can support the claimed mechanism

Use two experiments.

For the **controlled learning comparison**, give all trained arms the same pooled intervention dataset, initialization, corrective targets, optimizer schedule, and random seeds:

| Arm | Training difference |
|---|---|
| No correction training | Source-trained RoboTTT checkpoint; ordinary fast updates during execution |
| Correction-only | Process the same sequences, but reset temporal memory at the failure-to-correction boundary |
| Full failure context | Retain the actual failed prefix and supervise controller corrections |
| Shuffled failure context | Permute synchronized observation/action bundles within the failed prefix; preserve its length and the exact correction suffix |

The correction-only arm must process the prefix before discarding its state, so its compute and token exposure remain comparable. Preserve correction-position indexing across arms.

Evaluate the full-context checkpoint again with temporal memory reset at each policy timestep. This is an inference ablation, not a separately trained model.

Add a parameter-matched GDN arm using the same correction data to test whether the effect requires TTT rather than recurrent memory generally.

For the **automatic system experiment**, let each trainable method collect its own failures and corrections under the same attempted-episode and teacher-action budgets. Differences in collected data are part of the system result and must not be presented as an isolated effect of the loss function.

Also report:

- Teacher success directly from initial states.
- Teacher recovery success from VLA failure states.
- Hybrid VLA-plus-teacher success without training.
- VLA-only success after training.

These outcomes must remain separate.

## 10. Define improvement before seeing results

The primary metric is **official VLA-only task success within 220 actions**. During query evaluation:

- Teacher invocation is disabled.
- No slow-parameter update is allowed.
- Fast state resets at every episode boundary.
- Student input excludes teacher sidecars.
- Query trajectories never enter replay.

Report adaptation-task and transfer-task averages separately, with equal task weighting.

The preregistered practical target is:

- At least **10 percentage points** improvement over the no-correction-training arm on adaptation tasks.
- A paired 95% confidence interval for that improvement whose lower bound exceeds zero.
- Full failure context outperforming both correction-only and shuffled-prefix controls with paired uncertainty reported.
- Transfer measured independently; do not imply transfer if its interval includes zero.

Compute paired differences using the same query layouts. Use hierarchical resampling over tasks, adaptation runs, and episode identities, preserving pairing across methods.

Also report controller calls, teacher actions, failed interventions, training GPU-hours, elapsed adaptation time, and inference latency. Lower training loss or a successful teacher intervention does not count as VLA improvement.

The fixed main query matrix contains **6,000 rollouts per arm**: 10 tasks × 50 episodes × 4 checkpoints × 3 adaptation runs. Estimate its cost from the unscored integration pilot before launching it.

## 11. Verification and implementation order

Implementation ownership follows this sequence:

1. **Research/fidelity:** resolve the reference model and exact training manifest.
2. **Backend/controller:** extract the live-environment recovery service, implement takeover classification, and build the atomic recorder.
3. **ML:** implement the model adapter, sequence exporter, masked trainer, and checkpoint handling.
4. **DevOps:** pin the runtime and artifacts; benchmark collection, training, and query costs.
5. **Tester:** independently exercise lifecycle, action, gradient, masking, and split contracts.
6. **Reviewer:** assess fidelity, leakage, teacher privilege, and whether the experimental comparisons support the claims.

Recording and controller extraction can proceed while reference artifacts are being resolved. Scientific training waits for the fidelity requirements.

Required tests include:

- Exactly one environment reset and close; no reset at takeover.
- Correct handling of an already-held source.
- No execution after genuine termination.
- Observation/action alignment for every accepted step.
- Correct cancellation of unexecuted chunk tails.
- Correct masks at takeover, padding, and episode boundaries.
- Fast-state numerical parity with the reference implementation.
- Slow-parameter hashes unchanged during inference.
- Correct gradient flow through inner updates and TBPTT boundaries.
- No teacher-only fields in student batches.
- No overlap between adaptation and query state identities.
- No optimizer step when an update batch contains no supervised actions.

## 12. The evidence needed for an ICLR submission

The submission should contain:

- A method diagram showing both fast updates during execution and persistent correction training between rounds.
- VLA-only adaptation curves, teacher cost, and transfer results.
- The matched correction-only, shuffled-history, memory-reset, and GDN comparisons.
- Recovery-state analysis, including held-object and unrecoverable failures.
- A released intervention dataset with actor masks and provenance.
- Reproducible model, controller, preprocessing, split, and training manifests.

LIBERO establishes the first result. A broader manipulation claim requires an external replication. RoboCasa is the natural second environment, but it currently lacks a VLA adapter and uses a different moving-base/action interface. Its adapter, takeover teacher, and recorder must therefore pass their own validation before applying this protocol to the existing PickPlace-21 manifest.

If the external replication is not completed, scope the paper's conclusion to the demonstrated LIBERO skill family.

The implementation succeeds scientifically only if the trained VLA improves without its teacher, and the controlled experiments show why preserving failure context helps. Exact reuse of RoboTTT remains a prerequisite for the primary method claim—not a label applied after an approximate implementation.
