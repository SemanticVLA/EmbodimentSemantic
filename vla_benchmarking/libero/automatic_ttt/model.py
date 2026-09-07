"""A dependency-light, paper-shaped TTT-KVB sequence module.

The VLA backbone remains an injected dependency.  ``build_ttt_kvb_module``
only supplies the RoboTTT fast sequence component; a model-specific adapter is
responsible for connecting its tokens and flow-matching action head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple


@dataclass(frozen=True)
class TTTKVBConfig:
    model_dim: int
    fast_hidden_dim: int
    output_dim: int | None = None
    register_tokens_per_timestep: int = 16
    gate_initial_alpha: float = 0.001
    # RoboTTT reports a learned step size on top of a constant base learning
    # rate of 0.1.  The exact parameterization is still artifact-gated, but
    # this default preserves the published base value rather than silently
    # substituting a conventional 1e-3 optimizer rate.
    inner_learning_rate: float = 0.1
    learn_inner_learning_rate: bool = True
    fast_activation: str = "gelu"

    def __post_init__(self) -> None:
        if self.model_dim <= 0 or self.fast_hidden_dim <= 0:
            raise ValueError("model_dim and fast_hidden_dim must be positive")
        if self.output_dim is not None and self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.register_tokens_per_timestep != 16:
            raise ValueError("RoboTTT uses exactly 16 learned register tokens per timestep")
        if self.fast_activation != "gelu":
            raise ValueError("RoboTTT fast network is a two-layer GeLU MLP")


class FastState(NamedTuple):
    w1: Any
    b1: Any
    w2: Any
    b2: Any

    def detached(self) -> "FastState":
        # A detached state is still the leaf optimized by the next segment's
        # inner update.  ``detach()`` alone would make autograd.grad fail at
        # every TBPTT boundary.
        return FastState(*(value.detach().requires_grad_(True) for value in self))


def build_ttt_kvb_module(config: TTTKVBConfig) -> Any:
    """Build the torch module, importing torch only when it is requested."""
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("TTT-KVB training requires optional dependency 'torch'") from exc

    output_dim = config.output_dim or config.model_dim

    class TTTKVBModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.q_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
            self.k_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
            self.v_proj = nn.Linear(config.model_dim, output_dim, bias=False)
            self.w0_1 = nn.Parameter(torch.empty(config.fast_hidden_dim, config.model_dim))
            self.b0_1 = nn.Parameter(torch.zeros(config.fast_hidden_dim))
            self.w0_2 = nn.Parameter(torch.empty(output_dim, config.fast_hidden_dim))
            self.b0_2 = nn.Parameter(torch.zeros(output_dim))
            nn.init.xavier_uniform_(self.w0_1)
            nn.init.xavier_uniform_(self.w0_2)
            self.register_tokens = nn.Parameter(
                torch.zeros(config.register_tokens_per_timestep, config.model_dim)
            )
            nn.init.normal_(self.register_tokens, std=0.02)
            self.alpha = nn.Parameter(torch.tensor(float(config.gate_initial_alpha)).atanh())
            # Softplus parameterization keeps eta positive.  The exact
            # parameterization is an artifact-gated field in fidelity.py.
            inverse = torch.log(torch.expm1(torch.tensor(float(config.inner_learning_rate))))
            if config.learn_inner_learning_rate:
                self.log_inner_learning_rate = nn.Parameter(inverse)
            else:
                self.register_buffer("log_inner_learning_rate", inverse)

        @property
        def inner_learning_rate(self) -> Any:
            return F.softplus(self.log_inner_learning_rate)

        def reset_fast_state(self, batch_size: int | None = None) -> FastState:
            # Do not detach W0: the first segment can meta-learn the initial
            # state during pretraining/posttraining as described by RoboTTT.
            if batch_size is None:
                return FastState(self.w0_1, self.b0_1, self.w0_2, self.b0_2)
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")
            return FastState(*(
                parameter.unsqueeze(0).expand(batch_size, *parameter.shape)
                for parameter in (self.w0_1, self.b0_1, self.w0_2, self.b0_2)
            ))

        @staticmethod
        def _apply(state: FastState, x: Any) -> Any:
            if state.w1.ndim == 2:
                hidden = F.gelu(F.linear(x, state.w1, state.b1))
                return F.linear(hidden, state.w2, state.b2)
            # Each leading batch item has its own fast weights.  Using
            # batched einsums prevents one failed episode's K/V from updating
            # another episode's state.
            bias1 = state.b1.view(state.b1.shape[0], *([1] * (x.ndim - 2)), state.b1.shape[-1])
            hidden = F.gelu(torch.einsum("b...d,bhd->b...h", x, state.w1) + bias1)
            bias2 = state.b2.view(state.b2.shape[0], *([1] * (hidden.ndim - 2)), state.b2.shape[-1])
            return torch.einsum("b...h,boh->b...o", hidden, state.w2) + bias2

        def ttt_step(
            self,
            keys: Any,
            values: Any,
            queries: Any,
            state: FastState | None = None,
            *,
            create_graph: bool = True,
        ) -> tuple[Any, FastState, Any]:
            """Adapt on K,V, then apply the updated state to Q.

            This ordering is intentionally observable and tested.  ``keys``,
            ``values`` and ``queries`` may have arbitrary leading dimensions;
            the final dimension must match the configured projections.
            """
            if getattr(keys, "ndim", 0) < 2:
                raise ValueError("TTT K/V/Q tensors require a batch dimension")
            batch_size = int(keys.shape[0])
            if state is None:
                state = self.reset_fast_state(batch_size=batch_size)
            elif state.w1.ndim == 2:
                state = self.reset_fast_state(batch_size=batch_size)
            elif int(state.w1.shape[0]) != batch_size:
                raise ValueError("fast-state batch dimension does not match K/V/Q batch")
            params = tuple(state)
            prediction = self._apply(state, keys)
            squared = (prediction - values).pow(2)
            # Reduce only non-batch dimensions, then differentiate the sum.
            # Consequently each episode receives its own update and duplicating
            # a batch does not change the update magnitude for an example.
            per_example_loss = squared.reshape(batch_size, -1).mean(dim=1)
            update_loss = per_example_loss.sum()
            inner_loss = per_example_loss.mean()
            grads = torch.autograd.grad(update_loss, params, create_graph=create_graph, allow_unused=False)
            eta = self.inner_learning_rate
            updated = FastState(*(parameter - eta * grad for parameter, grad in zip(params, grads)))
            output = self._apply(updated, queries)
            return output, updated, inner_loss

        def forward(self, x: Any, state: FastState | None = None) -> tuple[Any, FastState, Any]:
            """Run one TTT step and apply the learned residual gate.

            The 16 learned registers are prepended to each timestep's token
            sequence.  They participate in the K/V binding update, while the
            returned residual is sliced back to the caller's original tokens;
            this keeps the action-head shape unchanged and gives registers a
            real cross-timestep information path.
            """
            if x.ndim < 2:
                raise ValueError("TTT input must have at least batch and feature dimensions")
            self._automatic_ttt_forward_observed = True
            squeezed = x.ndim == 2
            token_input = x.unsqueeze(-2) if squeezed else x
            prefix_shape = token_input.shape[:-2]
            registers = self.register_tokens.view(*([1] * len(prefix_shape)), config.register_tokens_per_timestep, config.model_dim)
            registers = registers.expand(*prefix_shape, config.register_tokens_per_timestep, config.model_dim)
            augmented = torch.cat((registers, token_input), dim=-2)
            keys = self.k_proj(augmented)
            values = self.v_proj(augmented)
            queries = self.q_proj(augmented)
            residual, state, inner_loss = self.ttt_step(keys, values, queries, state)
            residual = residual[..., config.register_tokens_per_timestep:, :]
            base = token_input[..., :output_dim]
            output = base + torch.tanh(self.alpha) * residual
            if squeezed:
                output = output.squeeze(-2)
            return output, state, inner_loss

        def detach_fast_state(self, state: FastState) -> FastState:
            return state.detached()

        def runtime_receipt(self) -> dict[str, Any]:
            """Describe what this module actually implements.

            This is deliberately a *single-layer-kernel* receipt.  It cannot
            pass the exact RoboTTT gate until an adapter wires equivalent
            modules into all 16 contextual DiT layers and supplies the VLA's
            published preprocessing/action-head metadata.
            """
            return {
                "scope": "algorithmic_component.single_layer_static_registers",
                "ttt_layer_count": 1,
                "fields": {
                    "fast_mlp_hidden_dim": config.fast_hidden_dim,
                    "qkv_dimensions": {
                        "q": config.model_dim,
                        "k": config.model_dim,
                        "v": output_dim,
                    },
                    "qkv_normalization": "none",
                    "inner_learning_rate_parameterization": "softplus(log_inner_learning_rate)",
                    "tbptt_segment_length": None,
                    "action_horizon": None,
                    "denoising_steps": None,
                    "token_packing": "register_prefix_local_kernel",
                },
                "optimizer_metadata": {},
            }

    return TTTKVBModule()


def model_parameter_metadata(module: Any) -> dict[str, Any]:
    """Return stable metadata used by experiment manifests and audits."""
    config = getattr(module, "config", None)
    return {
        "module_class": type(module).__name__,
        "config": vars(config) if config is not None else None,
        "slow_parameter_names": [name for name, _ in module.named_parameters()],
        "fast_state_parameters": ["w0_1", "b0_1", "w0_2", "b0_2"],
        "update_order": "inner K,V update then Q application",
        "gate": "tanh(alpha)",
        "runtime_receipt": module.runtime_receipt() if hasattr(module, "runtime_receipt") else None,
    }
