# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

"""CHaRS's OT-clustering intervention hooks, vendored for CAT's arm comparison.

`GaussianOTClustHook` and `GaussianOTPCAHook` are copied verbatim (class bodies
unchanged) from CHaRS ("Concept Heterogeneity-aware Representation Steering",
`/home/dixon/projects/CHaRS/tox_and_t2i/chars/hooks/transport.py:534-1114`), a
fork of this same `act` package that generalizes `LinearOTHook`'s single-Gaussian
closed-form OT to a k-means-clustered, Sinkhorn-Knopp-coupled OT plan (optionally
projected onto its top-`k_pcs` principal components for the PCA variant).

Not part of upstream `ml-act` -- kept in a separate file rather than merged into
`transport.py` so the diff against upstream `act` stays isolated to two new
files (`this module` + `act/utils/nonparam.py`, this file's only new
dependency). Both classes already subclass this package's own
`InterventionHook` and share its `fit(responses, labels)` / `forward(module,
input, output)` contract with every other hook here (`LinearOTHook`,
`GaussianOTHook`, ...), so they need no adapter to be driven by
`TorchHookConceptTransportPipeline` (see
`Concept_Activation_Transport/src/experiments/common/chars_transport.py`).

Imports are trimmed to only what these two classes use (the source file also
defines other hook classes that import `LinearProj`/`solve_ot_1d`/`nn`/
`DataLoader`/`Dataset`, none of which `GaussianOTClustHook`/`GaussianOTPCAHook`
reference).
"""

import copy
import logging
import typing as t

import einops
import numpy as np
import torch

from act.hooks.intervention_hook import InterventionHook
from act.utils.nonparam import (
    calculate_cost_matrix,
    calculate_kernel_matrix,
    calculate_kernel_matrix_adaptive,
    compute_cluster_weights,
    get_clusters,
    get_steering_direction2_torch,
    pca_get_steering_direction2_torch,
    sinkhorn_knopp_multi,
)
from act.utils.quantiles import compute_quantiles


class GaussianOTClustHook(InterventionHook):
    def __init__(
        self,
        module_name: str,
        device: str = None,
        intervention_position: str = "all",
        dtype: torch.dtype = torch.float32,
        strength: float = 1.0,
        std_eps: float = 1e-4,
        quantiles_src: str = "q_all",
        hook_onlymean: bool = True,
        k = 4,
        lmda = "adaptive",
        sim = "adaptive_gaussian",
        **kwargs,
    ):
        """
        Gaussian Optimal Transport hook. Assumes the output of each neuron is independent and Gaussianly distributed.
        Transports from N(mu1, std1) to N(mu2, std2) if hook_forward is True, and N(mu2, std2) to N(mu1, std1) otherwise.

        Args:
            module_name (str): Module name on which the hook is applied.
            device (torch.device): Torch device.
            dtype (str): The torch dtype for this hook.
            intervention_position (str): The position of the token to intervene upon (all, last)
            **kwargs: Any extra arguments, for compatibility with other hooks.
        """
        super().__init__(
            module_name=module_name,
            device=device,
            intervention_position=intervention_position,
            dtype=dtype,
        )
        # print('init hook start')

        self.strength = float(strength)
        self.onlymean = bool(int(hook_onlymean))
        self.std_eps = float(std_eps)
        self.quantiles_src = str(quantiles_src)
        self.k = int(k)
        self.lmda = str(lmda)
        self.sim = str(sim)

        # Register buffer placeholders
        for buffer_name in ["mu1", "mu2", "std1", "std2", "diff", "src_clusters", "tgt_clusters", "P_star"]:
            self.register_buffer(buffer_name, torch.empty(0))

        # Register non-buffer placeholders
        self.quantiles_dict_src = None

    def __str__(self):
        txt = (
            f"GaussianOTClust("
            f"module_name={self.module_name}, "
            f"quantiles_src={self.quantiles_src}, "
            f"strength={self.strength:0.2f}, "
            f"onlymean={self.hook_onlymean}"
            f")"
        )
        return txt

    def state_dict(self, *args, **kwargs) -> t.Dict:
        d = super().state_dict()
        # Merge non-buffers into dict
        d.update(
            {
                "quantiles_dict_src": self.quantiles_dict_src,
            }
        )
        # Merge hook_args required for incremental learning
        d.update(
            {
                "quantiles_src": self.quantiles_src,
            }
        )
        d.update(
            {
                "diff": self.diff
            }
        )
        d.update(
            {
                "src_clusters": self.src_clusters
            }
        )
        d.update(
            {
                "tgt_clusters": self.tgt_clusters
            }
        )
        d.update(
            {
                "P_star": self.P_star
            }
        )
        return d

    def load_state_dict(
        self,
        state_dict: t.Mapping[str, t.Any],
        state_path="",
        strict: bool = True,
        assign: bool = False,
    ) -> None:
        # Fill in registered buffers

        all_state_path = state_path.parent

        for buffer_name, _ in self.named_buffers():

            setattr(
                self,
                buffer_name,
                state_dict[buffer_name].to(self.device).to(self.dtype),
            )
        # breakpoint()

        self.quantiles_dict_src = {
            k: [v[0].to(self.device), v[1].to(self.device)]
            for k, v in state_dict["quantiles_dict_src"].items()
        }

        # Non tensor parameters. If these exist in state_dict, they will override the ones passed through constructor.
        # This is useful for hooks learnt for some specific args (eg. incremental hooks).
        for hook_arg in [attr for attr in dir(self) if attr.startswith("hook_")]:
            if hook_arg in state_dict:
                # We know the type, __init__() cas been called.
                vartype = type(getattr(self, hook_arg))
                varval = vartype(state_dict[hook_arg])
                logging.warning(
                    f"Overriding {hook_arg} to {varval} in {self.__class__.__name__}."
                )
                setattr(self, hook_arg, varval)
        self._post_load()

    def fit(self, responses: torch.Tensor, labels=torch.Tensor, **kwargs) -> None:

        # Typecasting to float64 to avoid overflow in the computation of the mean/std.
        z_f64 = responses.to(torch.float64)
        labels = labels.to(torch.bool)
        self.mu1 = torch.mean(z_f64[labels], dim=0).to(self.dtype)
        self.std1 = torch.std(z_f64[labels], dim=0).to(self.dtype)
        self.mu2 = torch.mean(z_f64[~labels], dim=0).to(self.dtype)
        self.std2 = torch.std(z_f64[~labels], dim=0).to(self.dtype)
        self.diff = self.mu2 - self.mu1

        # Clustering
        # TODO: Sanity check responses, sanity check labels
        src = z_f64[labels][None, :, :]
        tgt = z_f64[~labels][None, :, :]
        # TODO: Get src and tgt in the form (layer, ex_pt, batch, token, dim)
        src_clusters, src_labels, src_inertia = get_clusters(src, self.k)
        tgt_clusters, tgt_labels, tgt_inertia = get_clusters(tgt, self.k)
        src_cluster_weights = compute_cluster_weights(src_labels, self.k)
        tgt_cluster_weights = compute_cluster_weights(tgt_labels, self.k)
        cost_matrix = calculate_cost_matrix(src_clusters, tgt_clusters)
        if self.lmda == "adaptive":
            kernel_matrix = calculate_kernel_matrix_adaptive(cost_matrix)
        else:
            if self.lmda == "sqrtd":
                eps = np.sqrt(z_f64.shape[-1])
            else:
                eps = self.lmda
            kernel_matrix = calculate_kernel_matrix(cost_matrix, eps=eps)
        if self.k > 1:
            self.P_star = sinkhorn_knopp_multi(src_cluster_weights, tgt_cluster_weights, kernel_matrix, self.k, max_iter=1000, tau=1e-6)
        else:
            self.P_star = torch.Tensor([[1.0]])

        # Remove Floating Dimension
        src_clusters = src_clusters.squeeze(0)
        tgt_clusters = tgt_clusters.squeeze(0)
        if self.k > 1:
            self.P_star = self.P_star.squeeze(0)
        self.src_clusters = torch.Tensor(src_clusters).to(self.dtype)
        self.tgt_clusters = torch.Tensor(tgt_clusters).to(self.dtype)
        assert len(self.src_clusters.shape) == 2, "Problem with Source Clusters!"
        assert len(self.tgt_clusters.shape) == 2, "Problem with Target Clusters!"
        assert len(self.P_star.shape) == 2, "Problem with P_star!"

        self.quantiles_dict_src = compute_quantiles(z_f64[labels])
        self._post_load()

    def _post_load(self) -> None:
        """
        This method should be called after loading the states of the hook.
        So calls must be placed at the end of .fit() and at the end of .load_state_dict().
        """
        super()._post_load()

        self.mask = (
            torch.ones_like(
                self.mu1, dtype=torch.bool
            )  # just here to be able to comment/uncomment masks below easily]
            # TODO: We've had this all the time, remove?
            & (self.std1 > self.std_eps)
            & (self.std2 > self.std_eps)
        )

        # Pre-computing things beforehand
        self.mu1_m = self.mu1[self.mask]
        self.mu2_m = self.mu2[self.mask]
        self.std1_m = self.std1[self.mask]
        self.std2_m = self.std2[self.mask]

        self.std1_2_m = self.std2_m / self.std1_m

        # TODO: Move this to some post_load method to avoid doing it every time.
        if self.quantiles_src == "q_all":
            self.quantiles_src = [-1e6, 1e6]
        else:
            self.quantiles_src = copy.deepcopy(
                self.quantiles_dict_src[self.quantiles_src]
            )
            self.quantiles_src[0] = self.quantiles_src[0][self.mask].view(1, -1)
            self.quantiles_src[1] = self.quantiles_src[1][self.mask].view(1, -1)

    def forward(self, module, input, output) -> t.Any:
        output_shape = output.shape
        if len(output_shape) == 3:
            output = output.reshape(-1, output_shape[2])
            self.mu1_m = self.mu1_m.view(1, -1)
            self.mu2_m = self.mu2_m.view(1, -1)
            self.std1_m = self.std1_m.view(1, -1)
            self.std2_m = self.std2_m.view(1, -1)
            self.std1_2_m = self.std1_2_m.view(1, -1)
            z_unit = output[:, self.mask]  # B,  U
            z_mask = z_unit
        elif len(output_shape) == 4:
            # Handles image model case
            self.mu1_m = self.mu1_m.view(1, -1, 1, 1)
            self.mu2_m = self.mu2_m.view(1, -1, 1, 1)
            self.std1_m = self.std1_m.view(1, -1, 1, 1)
            self.std2_m = self.std2_m.view(1, -1, 1, 1)
            self.std1_2_m = self.std1_2_m.view(1, -1, 1, 1)
            z_unit = output[:, self.mask]  # B,  U
            z_mask = z_unit.mean(dim=(2, 3), keepdim=False)  # B, C
        else:
            raise NotImplementedError(f"Can't handle tensors with dim=2.")

        # Select outputs that fall in statistics of pooled
        pool_mask = (
            (self.quantiles_src[0] < z_mask) & (z_mask < self.quantiles_src[1])
        ).to(z_unit.dtype)

        # Transport the outputs
        if self.onlymean:
            feature_direction = get_steering_direction2_torch(self.src_clusters, self.tgt_clusters, self.P_star, output, self.sim)
            z_ot = z_unit + 1.0*feature_direction
        else:
            z_ot = self.std1_2_m * (z_unit - self.mu1_m) + self.mu2_m

        # Apply transport with specific strength
        z_ot = self.strength * z_ot + (1 - self.strength) * z_unit

        if len(output_shape) == 4:
            pool_mask = pool_mask[..., None, None]

        # Only apply masked outputs
        output[:, self.mask] = pool_mask * z_ot + (1 - pool_mask) * z_unit
        output = output.view(*output_shape)
        return output

class GaussianOTPCAHook(InterventionHook):
    def __init__(
        self,
        module_name: str,
        device: str = None,
        intervention_position: str = "all",
        dtype: torch.dtype = torch.float32,
        strength: float = 1.0,
        std_eps: float = 1e-4,
        quantiles_src: str = "q_all",
        hook_onlymean: bool = True,
        k = 4,
        k_pcs = 3,
        lmda = "adaptive",
        sim = "adaptive_gaussian",
        **kwargs,
    ):
        """
        Gaussian Optimal Transport hook. Assumes the output of each neuron is independent and Gaussianly distributed.
        Transports from N(mu1, std1) to N(mu2, std2) if hook_forward is True, and N(mu2, std2) to N(mu1, std1) otherwise.

        Args:
            module_name (str): Module name on which the hook is applied.
            device (torch.device): Torch device.
            dtype (str): The torch dtype for this hook.
            intervention_position (str): The position of the token to intervene upon (all, last)
            **kwargs: Any extra arguments, for compatibility with other hooks.
        """
        super().__init__(
            module_name=module_name,
            device=device,
            intervention_position=intervention_position,
            dtype=dtype,
        )

        self.strength = float(strength)
        self.onlymean = bool(int(hook_onlymean))
        self.std_eps = float(std_eps)
        self.quantiles_src = str(quantiles_src)
        self.k = int(k)
        self.k_pcs = int(k_pcs)
        self.lmda = str(lmda)
        self.sim = str(sim)

        # Register buffer placeholders
        for buffer_name in ["mu1", "mu2", "std1", "std2", "diff", "src_clusters", "tgt_clusters", "P_star", "v_bar", "pc_scores", "top_k_pc"]:
            self.register_buffer(buffer_name, torch.empty(0))

        # Register non-buffer placeholders
        self.quantiles_dict_src = None

    def __str__(self):
        txt = (
            f"GaussianOTPCA("
            f"module_name={self.module_name}, "
            f"quantiles_src={self.quantiles_src}, "
            f"strength={self.strength:0.2f}, "
            f"onlymean={self.hook_onlymean}"
            f")"
        )
        return txt

    def state_dict(self, *args, **kwargs) -> t.Dict:
        d = super().state_dict()
        # Merge non-buffers into dict
        d.update(
            {
                "quantiles_dict_src": self.quantiles_dict_src,
            }
        )
        # Merge hook_args required for incremental learning
        d.update(
            {
                "quantiles_src": self.quantiles_src,
            }
        )
        d.update(
            {
                "diff": self.diff
            }
        )
        d.update(
            {
                "src_clusters": self.src_clusters
            }
        )
        d.update(
            {
                "tgt_clusters": self.tgt_clusters
            }
        )
        d.update(
            {
                "P_star": self.P_star
            }
        )
        d.update(
            {
                "v_bar": self.v_bar
            }
        )
        d.update(
            {
                "pc_scores": self.pc_scores
            }
        )
        d.update(
            {
                "top_k_pc": self.top_k_pc
            }
        )
        return d

    def load_state_dict(
        self,
        state_dict: t.Mapping[str, t.Any],
        state_path="",
        strict: bool = True,
        assign: bool = False,
    ) -> None:
        # Fill in registered buffers

        all_state_path = state_path.parent

        for buffer_name, _ in self.named_buffers():

            setattr(
                self,
                buffer_name,
                state_dict[buffer_name].to(self.device).to(self.dtype),
            )
        # breakpoint()

        self.quantiles_dict_src = {
            k: [v[0].to(self.device), v[1].to(self.device)]
            for k, v in state_dict["quantiles_dict_src"].items()
        }

        # Non tensor parameters. If these exist in state_dict, they will override the ones passed through constructor.
        # This is useful for hooks learnt for some specific args (eg. incremental hooks).
        for hook_arg in [attr for attr in dir(self) if attr.startswith("hook_")]:
            if hook_arg in state_dict:
                # We know the type, __init__() cas been called.
                vartype = type(getattr(self, hook_arg))
                varval = vartype(state_dict[hook_arg])
                logging.warning(
                    f"Overriding {hook_arg} to {varval} in {self.__class__.__name__}."
                )
                setattr(self, hook_arg, varval)
        self._post_load()

    def fit(self, responses: torch.Tensor, labels=torch.Tensor, **kwargs) -> None:

        # Typecasting to float64 to avoid overflow in the computation of the mean/std.
        z_f64 = responses.to(torch.float64)
        labels = labels.to(torch.bool)
        self.mu1 = torch.mean(z_f64[labels], dim=0).to(self.dtype)
        self.std1 = torch.std(z_f64[labels], dim=0).to(self.dtype)
        self.mu2 = torch.mean(z_f64[~labels], dim=0).to(self.dtype)
        self.std2 = torch.std(z_f64[~labels], dim=0).to(self.dtype)
        self.diff = self.mu2 - self.mu1

        # Clustering
        # TODO: Sanity check responses, sanity check labels
        src = z_f64[labels][None, :, :]
        tgt = z_f64[~labels][None, :, :]
        # TODO: Get src and tgt in the form (layer, ex_pt, batch, token, dim)
        src_clusters, src_labels, src_inertia = get_clusters(src, self.k)
        tgt_clusters, tgt_labels, tgt_inertia = get_clusters(tgt, self.k)
        src_cluster_weights = compute_cluster_weights(src_labels, self.k)
        tgt_cluster_weights = compute_cluster_weights(tgt_labels, self.k)
        cost_matrix = calculate_cost_matrix(src_clusters, tgt_clusters)
        if self.lmda == "adaptive":
            kernel_matrix = calculate_kernel_matrix_adaptive(cost_matrix)
        else:
            if self.lmda == "sqrtd":
                eps = np.sqrt(z_f64.shape[-1])
            else:
                eps = self.lmda
            kernel_matrix = calculate_kernel_matrix(cost_matrix, eps=eps)
        if self.k > 1:
            self.P_star = sinkhorn_knopp_multi(src_cluster_weights, tgt_cluster_weights, kernel_matrix, self.k, max_iter=1000, tau=1e-6)
        else:
            self.P_star = torch.Tensor([[1.0]])

        src_clusters = src_clusters.squeeze(0)
        tgt_clusters = tgt_clusters.squeeze(0)
        if self.k > 1:
            self.P_star = self.P_star.squeeze(0)

        P_star = self.P_star.float().numpy()

        # Computation of Principal Components and Score
        v = tgt_clusters[None, :, :] - src_clusters[:, None, :]
        i_test, j_test = 2, 3
        correct_test = tgt_clusters[j_test] - src_clusters[i_test]
        computed_test = v[i_test, j_test]
        assert np.isclose(correct_test, computed_test).all()

        scaled_v = v * P_star[:, :, None]
        v_bar = np.sum(scaled_v, axis = (0, 1))


        # Centered v
        centered_v = v - v_bar[None, None, :]

        # Computation of Weighted Covariance Matrix
        centered_v_reshaped = centered_v.reshape(-1, v_bar.shape[-1])
        P_star_reshaped = P_star.reshape(-1, 1)
        sigma_total_ex_pt = centered_v_reshaped.transpose(1, 0) @ (P_star_reshaped * centered_v_reshaped)

        # EigenDecomposition
        ex_pt_eigdecomp_ex_pt = np.linalg.eigh(sigma_total_ex_pt)

        # Principal_components;
        eigenvecs_ex_pt = ex_pt_eigdecomp_ex_pt.eigenvectors[:, ::-1]
        top_k_pc_ex_pt = eigenvecs_ex_pt[:, :self.k_pcs].transpose().copy()

        # Compute pc scores:
        pc_scores_ex_pt = einops.einsum(top_k_pc_ex_pt, centered_v, "k d, i j d -> k i j")


        # Remove Floating Dimension

        self.src_clusters = torch.Tensor(src_clusters).to(self.dtype)
        self.tgt_clusters = torch.Tensor(tgt_clusters).to(self.dtype)
        self.v_bar = torch.Tensor(v_bar).to(self.dtype)
        self.top_k_pc = torch.Tensor(top_k_pc_ex_pt).to(self.dtype)
        self.pc_scores = torch.Tensor(pc_scores_ex_pt).to(self.dtype)

        assert len(self.src_clusters.shape) == 2, "Problem with Source Clusters!"
        assert len(self.tgt_clusters.shape) == 2, "Problem with Target Clusters!"
        assert len(self.P_star.shape) == 2, "Problem with P_star!"
        assert len(self.v_bar.shape) == 1, "Problem with V bar!"
        assert len(self.top_k_pc.shape) == 2, "Problem with Top K PC!"
        assert len(self.pc_scores.shape) == 3, "Problem with PC Scores!"

        self.quantiles_dict_src = compute_quantiles(z_f64[labels])
        self._post_load()

    def _post_load(self) -> None:
        """
        This method should be called after loading the states of the hook.
        So calls must be placed at the end of .fit() and at the end of .load_state_dict().
        """
        super()._post_load()

        self.mask = (
            torch.ones_like(
                self.mu1, dtype=torch.bool
            )  # just here to be able to comment/uncomment masks below easily]
            # TODO: We've had this all the time, remove?
            & (self.std1 > self.std_eps)
            & (self.std2 > self.std_eps)
        )

        # Pre-computing things beforehand
        self.mu1_m = self.mu1[self.mask]
        self.mu2_m = self.mu2[self.mask]
        self.std1_m = self.std1[self.mask]
        self.std2_m = self.std2[self.mask]

        self.std1_2_m = self.std2_m / self.std1_m

        # TODO: Move this to some post_load method to avoid doing it every time.
        if self.quantiles_src == "q_all":
            self.quantiles_src = [-1e6, 1e6]
        else:
            self.quantiles_src = copy.deepcopy(
                self.quantiles_dict_src[self.quantiles_src]
            )
            self.quantiles_src[0] = self.quantiles_src[0][self.mask].view(1, -1)
            self.quantiles_src[1] = self.quantiles_src[1][self.mask].view(1, -1)

    def forward(self, module, input, output) -> t.Any:
        output_shape = output.shape
        if len(output_shape) == 3:
            output = output.reshape(-1, output_shape[2])
            self.mu1_m = self.mu1_m.view(1, -1)
            self.mu2_m = self.mu2_m.view(1, -1)
            self.std1_m = self.std1_m.view(1, -1)
            self.std2_m = self.std2_m.view(1, -1)
            self.std1_2_m = self.std1_2_m.view(1, -1)
            z_unit = output[:, self.mask]  # B,  U
            z_mask = z_unit
        elif len(output_shape) == 4:
            # Handles image model case
            self.mu1_m = self.mu1_m.view(1, -1, 1, 1)
            self.mu2_m = self.mu2_m.view(1, -1, 1, 1)
            self.std1_m = self.std1_m.view(1, -1, 1, 1)
            self.std2_m = self.std2_m.view(1, -1, 1, 1)
            self.std1_2_m = self.std1_2_m.view(1, -1, 1, 1)
            z_unit = output[:, self.mask]  # B,  U
            z_mask = z_unit.mean(dim=(2, 3), keepdim=False)  # B, C
        else:
            raise NotImplementedError(f"Can't handle tensors with dim=2.")

        # Select outputs that fall in statistics of pooled
        pool_mask = (
            (self.quantiles_src[0] < z_mask) & (z_mask < self.quantiles_src[1])
        ).to(z_unit.dtype)

        # Transport the outputs
        if self.onlymean:
            feature_direction = pca_get_steering_direction2_torch(self.src_clusters, self.tgt_clusters, self.P_star, self.v_bar, self.pc_scores, self.top_k_pc, output, self.sim)
            z_ot = z_unit + 1.0*feature_direction
        else:
            z_ot = self.std1_2_m * (z_unit - self.mu1_m) + self.mu2_m

        # Apply transport with specific strength
        z_ot = self.strength * z_ot + (1 - self.strength) * z_unit

        if len(output_shape) == 4:
            pool_mask = pool_mask[..., None, None]

        # Only apply masked outputs
        output[:, self.mask] = pool_mask * z_ot + (1 - pool_mask) * z_unit
        output = output.view(*output_shape)
        return output
