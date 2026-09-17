"""
Physics-informed inverse network for two-plane rotor unbalance.

WHAT THIS IS, PRECISELY. Not a PDE-residual PINN and not an MLP with a physics
penalty bolted on. The governing physics here is a *parametric linear operator* --
the unbalance-to-response influence model -- so this is a differentiable-physics
inverse model: a modally-parameterized anisotropic influence operator whose
closed-form least-squares inverse is differentiated end-to-end.

WHY THIS ARCHITECTURE. `scripts/representation_test.py` and
`scripts/angle_novelty_control.py` show that the deployed Extra-Trees estimator
loses accuracy on unseen angles (predictive R2 -1.88 vs +0.75 for a size-matched
random holdout, phase 69 deg vs 3 deg). The Cartesian input-output formulation
does not encode rotational equivariance, whereas the operator below does:

    V_s(w) = sum_j [ Hf_sj(w) * U_j  +  Hb_sj(w) * conj(U_j) ]

Hf (forward whirl) is exactly equivariant under a global rotation of U; Hb
(backward whirl) breaks equivariance in the physically correct way, which is how an
anisotropic rotor actually behaves. Set `anisotropic=False` to get the isotropic
ablation (Hb == 0), which is exactly equivariant.

MODAL PARAMETERIZATION, AND WHY IT MATTERS. An unconstrained per-speed anisotropic
fit was tried during this project and scored WORSE than the isotropic ICM (R2pred
0.62 -> 0.29 under angle holdout), because it adds free complex coefficients per
speed against only four distinct high-U angles. Here each mode contributes a
rank-one residue phi_sm * chi_jm shared across every speed and every angle:

    H_sj(w) = sum_m  phi_sm * chi_jm * g_m(w),    g_m(w) = D_m(w_ref) / D_m(w)
    D_m(w)  = w_m^2 - w^2 + 2i * zeta_m * w_m * w

Parameter budget: 26*M real numbers (2 poles + 8 sensor + 4 disk residue reals per
sign per mode). M=2 -> 52 parameters against 170 (condition, speed) pairs x 4
complex sensors = 1360 real equations. The rank-one factorization is both fewer
parameters than free residues and the physically meaningful constraint.

The `g_m(w) = D_m(w_ref)/D_m(w)` form is a reparameterization, not a different
model: it makes every residue dimensionless and O(1) at the reference speed, which
is what keeps the fit conditioned.

IDENTIFIABILITY WARNING -- READ BEFORE CLAIMING ANYTHING. This campaign dwells at
only two sub-critical speeds (w = 105 and 126 rad/s) against w_n ~ 173 rad/s. The
frequency lever arm is far too short to identify the poles from the response, so
`w_m` and `zeta_m` are held near their measured run-up values by `lambda_prior` and
are effectively PRIOR-DOMINATED, not estimated. Likewise Hb is excited only by
angular variation, and only four distinct angles exist above 24 g.mm. Run
`scripts/pinn_gate.py --identifiability` and report which parameters are informed by
data and which are not. Do not present prior-dominated parameters as measurements.

LOSSES.
    L_sup    supervised, on labeled rows: || U_hat - U ||^2 in normalized units,
             complex, so magnitude and phase are fit jointly with no wrap-around.
    L_phys   || A(psi) u_hat - b ||^2, the least-squares residual of the operator
             fit. NEEDS NO LABELS -- it measures whether the operator can explain
             the measured response at all. Computed on the training fold by default;
             pass `extra_unlabeled` to add genuinely unlabeled acquisitions (the 51
             blind runs currently do no work in training). No blind LABEL is ever
             touched, but the gate defaults to `extra_unlabeled=None` so the
             head-to-head comparison uses exactly the same data as the baselines.
    L_prior  anchors poles to the measured run-up identification (see above).

OPTIONAL EQUIVARIANT RESIDUAL (`residual=True`). A small network on
rotation-INVARIANT features emitting a per-disk log-magnitude scale and a phase
offset -- the parameterization validated in the representation test. It corrects
what the modal operator cannot express while preserving equivariance by
construction. Off by default: the first question is what physics structure alone
buys, and adding the residual before answering that confounds the two.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:                                  # pragma: no cover
    raise ImportError("estimation.pinn needs pytorch: pip install torch") from exc

from processing.features import SENSORS

__all__ = ["PINNRegressor", "pinn_design_matrix", "N_SENSORS", "N_DISKS"]

N_SENSORS = len(SENSORS)          # 4 probes
N_DISKS = 2                       # 2 balancing planes

# measured run-up identification: Nc ~ 1648 rpm with an X/Y split near
# 1750 rpm, zeta ~ 0.04. Used as the pole PRIOR, not as an initial guess to be
# freely optimized away -- see the identifiability warning above.
RPM_TO_RADS = 2.0 * np.pi / 60.0
# The measured first bending pair, split by support anisotropy.
W_PRIOR_RPM = np.array([1648.0, 1750.0])
W_PRIOR = W_PRIOR_RPM * RPM_TO_RADS
ZETA_PRIOR = np.array([0.04, 0.04])


def pole_priors(n_modes: int):
    """Distinct pole priors for `n_modes`, in rad/s, with per-mode prior weights.

    Tiling the measured pair (an earlier `np.resize`) gave DUPLICATE poles for
    n_modes > 2, which makes the modal basis degenerate: two modes with identical
    denominators are indistinguishable, their residues are unidentifiable, and the
    fit can diverge. Modes beyond the measured pair are therefore placed at
    geometrically increasing frequencies and act as a RESIDUAL-FLEXIBILITY term,
    the standard modal-truncation device for out-of-band contributions.

    Those extra modes are NOT identified by this campaign, which dwells only at two
    sub-critical speeds, so they receive a weak prior weight and must be described
    as residual terms rather than as measured modes.
    """
    w, z, wt = [], [], []
    for m in range(n_modes):
        if m < len(W_PRIOR_RPM):
            w.append(W_PRIOR_RPM[m]); z.append(ZETA_PRIOR[m]); wt.append(1.0)
        else:
            k = m - len(W_PRIOR_RPM)
            w.append(W_PRIOR_RPM[0] * (3.0 + 2.0 * k))   # 3x, 5x, 7x the first
            z.append(0.05); wt.append(0.05)              # weakly anchored
    return (np.array(w) * RPM_TO_RADS, np.array(z), np.array(wt))


def pinn_design_matrix(df, invariant_cols: list[str] | None = None) -> np.ndarray:
    """
    Pack the operator's inputs into one array so the PINN can be driven through the
    same sklearn-style `fit(X, Y)` / `predict(X)` interface as the baselines, and
    therefore be scored by the SAME factor-aware protocols without special-casing.

    Layout:  [0:4] Re V_s   [4:8] Im V_s   [8] rpm   [9:] invariant features
    """
    cols = [df[f"{p}_resp_re"].to_numpy(float) for p in SENSORS]
    cols += [df[f"{p}_resp_im"].to_numpy(float) for p in SENSORS]
    cols += [df["rpm"].to_numpy(float)]
    if invariant_cols:
        cols += [np.nan_to_num(df[c].to_numpy(float)) for c in invariant_cols]
    return np.column_stack(cols)


class _Operator(nn.Module):
    """The modally-parameterized influence operator and its differentiable inverse."""

    def __init__(self, n_modes: int, w_ref: float, anisotropic: bool,
                 n_invariant: int = 0, residual: bool = False,
                 residual_width: int = 16, seed: int = 0):
        super().__init__()
        M = self.M = n_modes
        self.anisotropic = anisotropic
        self.w_ref = w_ref

        # ---- poles: w_m = w_prior * exp(dlogw), zeta_m = zeta_prior * exp(dlogz)
        # parameterized as deviations so lambda_prior can pull them to zero and so
        # positivity is structural.
        wp, zp, wt = pole_priors(M)
        self.register_buffer("w_prior", torch.tensor(wp, dtype=torch.float64))
        self.register_buffer("zeta_prior", torch.tensor(zp, dtype=torch.float64))
        self.register_buffer("prior_weight", torch.tensor(wt, dtype=torch.float64))
        self.dlogw = nn.Parameter(torch.zeros(M, dtype=torch.float64))
        self.dlogz = nn.Parameter(torch.zeros(M, dtype=torch.float64))

        # ---- rank-one modal residues, real/imag stored separately
        g = torch.Generator().manual_seed(seed)
        def p(*shape, scale):
            return nn.Parameter(scale * torch.randn(*shape, generator=g, dtype=torch.float64))
        s = 1.0 / np.sqrt(M)
        self.phi_f_re = p(N_SENSORS, M, scale=s); self.phi_f_im = p(N_SENSORS, M, scale=s)
        self.chi_f_re = p(N_DISKS, M, scale=s);   self.chi_f_im = p(N_DISKS, M, scale=s)
        # backward residues start near zero -> the model begins at the well-behaved
        # isotropic solution and adds anisotropy only if the data pay for it
        b = 0.01 * s
        self.phi_b_re = p(N_SENSORS, M, scale=b); self.phi_b_im = p(N_SENSORS, M, scale=b)
        self.chi_b_re = p(N_DISKS, M, scale=b);   self.chi_b_im = p(N_DISKS, M, scale=b)

        # ---- optional equivariant residual head
        self.residual = residual and n_invariant > 0
        if self.residual:
            self.net = nn.Sequential(
                nn.Linear(n_invariant, residual_width), nn.Tanh(),
                nn.Linear(residual_width, 2 * N_DISKS),          # [log-scale, d-phase] per disk
            ).double()
            # `nn.Linear` uses the global torch RNG by default. Reinitialise every
            # layer from the local restart generator so repeated runs are exactly
            # reproducible without changing global RNG state.
            with torch.no_grad():
                for layer in self.net:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, generator=g)
                        layer.bias.zero_()
                self.net[-1].weight.mul_(0.01)                   # start near identity

    # ------------------------------------------------------------------ physics
    def poles(self):
        return self.w_prior * torch.exp(self.dlogw), self.zeta_prior * torch.exp(self.dlogz)

    def _g(self, w):
        """(N, M) dimensionless modal speed factor D_m(w_ref)/D_m(w)."""
        wm, zm = self.poles()
        def D(x):
            x = x.reshape(-1, 1)
            return (wm ** 2 - x ** 2) + 1j * (2.0 * zm * wm * x)
        d_ref = D(torch.full_like(w[:1], self.w_ref))       # (1, M)
        return d_ref / D(w)

    def H(self, w):
        """Forward/backward influence tensors, each (N, S, J) complex."""
        g = self._g(w)                                       # (N, M)
        phi_f = self.phi_f_re + 1j * self.phi_f_im           # (S, M)
        chi_f = self.chi_f_re + 1j * self.chi_f_im           # (J, M)
        Hf = torch.einsum("sm,jm,nm->nsj", phi_f, chi_f, g)
        if self.anisotropic:
            phi_b = self.phi_b_re + 1j * self.phi_b_im
            chi_b = self.chi_b_re + 1j * self.chi_b_im
            Hb = torch.einsum("sm,jm,nm->nsj", phi_b, chi_b, g)
        else:
            Hb = torch.zeros_like(Hf)
        return Hf, Hb

    def real_system(self, w):
        """
        Assemble the real (N, 2S, 2J) system  A u = b  with u = [Re U_j, Im U_j].

        From V_s = sum_j [Hf_sj U_j + Hb_sj conj(U_j)], writing U_j = x_j + i y_j:
            Re V_s = sum_j (Re Hf + Re Hb) x_j + (Im Hb - Im Hf) y_j
            Im V_s = sum_j (Im Hf + Im Hb) x_j + (Re Hf - Re Hb) y_j
        2S = 8 real equations for 2J = 4 unknowns -> over-determined, which is why
        this is better conditioned than the 4-equation complex-plane form.
        """
        Hf, Hb = self.H(w)
        fr, fi, br, bi = Hf.real, Hf.imag, Hb.real, Hb.imag
        N = w.shape[0]
        A = torch.zeros(N, 2 * N_SENSORS, 2 * N_DISKS, dtype=torch.float64)
        A[:, 0::2, 0::2] = fr + br
        A[:, 0::2, 1::2] = bi - fi
        A[:, 1::2, 0::2] = fi + bi
        A[:, 1::2, 1::2] = fr - br
        return A

    def invert(self, w, b, ridge: float = 1e-8):
        """Differentiable ridge-regularized least-squares inverse. Returns (u, A)."""
        A = self.real_system(w)
        At = A.transpose(1, 2)
        AtA = At @ A
        eye = torch.eye(2 * N_DISKS, dtype=torch.float64).expand_as(AtA)
        # scale the ridge to the problem so it regularizes without biasing
        lam = ridge * torch.diagonal(AtA, dim1=1, dim2=2).mean(dim=1).clamp(min=1e-12)
        u = torch.linalg.solve(AtA + lam.reshape(-1, 1, 1) * eye,
                               (At @ b.unsqueeze(-1))).squeeze(-1)
        return u, A

    def forward(self, w, b, inv=None, ridge: float = 1e-8):
        u, A = self.invert(w, b, ridge)
        if self.residual and inv is not None:
            h = self.net(inv)
            for j in range(N_DISKS):
                x, y = u[:, 2 * j].clone(), u[:, 2 * j + 1].clone()
                scale = torch.exp(h[:, j].clamp(-2.0, 2.0))       # magnitude correction
                dphi = h[:, N_DISKS + j]                          # phase offset (rad)
                c, s = torch.cos(dphi), torch.sin(dphi)
                u = u.clone()
                u[:, 2 * j] = scale * (c * x - s * y)
                u[:, 2 * j + 1] = scale * (s * x + c * y)
        resid = (A @ u.unsqueeze(-1)).squeeze(-1) - b             # physics residual
        return u, resid


class PINNRegressor:
    """
    sklearn-style wrapper so the PINN is scored by the same protocols as the
    baselines. `X` comes from `pinn_design_matrix`; `Y` is [U1_re, U1_im, U2_re, U2_im].

    Inputs and targets are standardized internally (the operator absorbs the scale),
    which is what keeps the least-squares solve conditioned.
    """

    def __init__(self, n_modes: int = 2, anisotropic: bool = True,
                 residual: bool = False, n_invariant: int = 0,
                 epochs: int = 3000, lr: float = 0.02,
                 lambda_phys: float = 1.0, lambda_prior: float = 10.0,
                 lambda_res: float = 1e-3, ridge: float = 1e-8,
                 n_restarts: int = 3,
                 extra_unlabeled: np.ndarray | None = None, verbose: bool = False):
        self.__dict__.update(locals()); del self.self

    # ------------------------------------------------------------------ helpers
    def _split(self, X):
        V = X[:, :2 * N_SENSORS]
        w = X[:, 2 * N_SENSORS] * RPM_TO_RADS
        inv = X[:, 2 * N_SENSORS + 1:] if self.n_invariant else None
        return V, w, inv

    def _b(self, V):
        """Interleave to [Re V_1, Im V_1, Re V_2, Im V_2, ...] matching real_system."""
        b = np.empty_like(V)
        b[:, 0::2] = V[:, :N_SENSORS]
        b[:, 1::2] = V[:, N_SENSORS:]
        return b

    # ---------------------------------------------------------------------- fit
    def fit(self, X, Y):
        V, w, inv = self._split(np.asarray(X, float))
        Y = np.asarray(Y, float)
        b = self._b(V)
        self.sV_ = float(np.sqrt(np.mean(b ** 2))) or 1.0
        self.sU_ = float(np.sqrt(np.mean(Y ** 2))) or 1.0
        self.w_ref_ = float(np.mean(w))
        if inv is not None:
            self.mu_ = inv.mean(0); self.sd_ = inv.std(0) + 1e-9

        t = lambda a: torch.tensor(a, dtype=torch.float64)
        bt, wt, yt = t(b / self.sV_), t(w), t(Y / self.sU_)
        it = t((inv - self.mu_) / self.sd_) if inv is not None else None

        # unlabeled responses for the physics term (no labels used)
        xb = xw = xi = None
        if self.extra_unlabeled is not None and len(self.extra_unlabeled):
            Vx, wx, ivx = self._split(np.asarray(self.extra_unlabeled, float))
            xb, xw = t(self._b(Vx) / self.sV_), t(wx)
            xi = t((ivx - self.mu_) / self.sd_) if ivx is not None else None

        # Several random restarts, keeping the one with the lowest TRAINING loss.
        # Gradient descent on this operator occasionally lands in a bad local
        # minimum, which showed up as one mode count failing while its neighbours
        # succeeded. Selection uses the training objective only, never test data,
        # so it removes an optimizer artifact without leaking.
        best_state, best_loss, best_final = None, np.inf, None
        for restart in range(max(1, self.n_restarts)):
            state, loss_val, final = self._train_once(restart, bt, wt, yt, it, xb, xw, xi)
            if loss_val < best_loss:
                best_state, best_loss, best_final = state, loss_val, final
            if self.verbose:
                print(f"    restart {restart}: loss={loss_val:.5f}")
        self.op_.load_state_dict(best_state)
        self.final_ = best_final
        self.final_["restarts"] = int(max(1, self.n_restarts))
        self.final_["best_train_loss"] = float(best_loss)
        return self

    def _train_once(self, seed, bt, wt, yt, it, xb, xw, xi):
        self.op_ = _Operator(self.n_modes, self.w_ref_, self.anisotropic,
                             n_invariant=self.n_invariant, residual=self.residual,
                             seed=seed)
        opt = torch.optim.Adam(self.op_.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        for ep in range(self.epochs):
            opt.zero_grad()
            u, resid = self.op_(wt, bt, it, self.ridge)
            l_sup = ((u - yt) ** 2).mean()
            l_phys = (resid ** 2).mean()
            if xb is not None:
                _, rx = self.op_(xw, xb, xi, self.ridge)
                l_phys = l_phys + (rx ** 2).mean()
            pw = self.op_.prior_weight
            l_prior = ((pw * self.op_.dlogw ** 2).sum()
                       + (pw * self.op_.dlogz ** 2).sum())
            loss = l_sup + self.lambda_phys * l_phys + self.lambda_prior * l_prior
            if self.op_.residual:
                loss = loss + self.lambda_res * sum(
                    (p ** 2).sum() for p in self.op_.net.parameters())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.op_.parameters(), 10.0)
            opt.step(); sched.step()
            if self.verbose and (ep % 500 == 0 or ep == self.epochs - 1):
                print(f"    ep {ep:5d}  sup={l_sup.item():.4f} phys={l_phys.item():.4f} "
                      f"prior={l_prior.item():.2e}")
        final = {k: float(v.detach()) for k, v in
                 (("sup", l_sup), ("phys", l_phys), ("prior", l_prior))}
        return ({k: v.detach().clone() for k, v in self.op_.state_dict().items()},
                float(loss.detach()), final)

    # ------------------------------------------------------------------ predict
    def predict(self, X):
        V, w, inv = self._split(np.asarray(X, float))
        t = lambda a: torch.tensor(a, dtype=torch.float64)
        it = t((inv - self.mu_) / self.sd_) if inv is not None else None
        with torch.no_grad():
            u, _ = self.op_(t(w), t(self._b(V) / self.sV_), it, self.ridge)
        return u.numpy() * self.sU_

    # ------------------------------------------------------ forward prediction
    def predict_response(self, X, U_true) -> np.ndarray:
        """
        FORWARD direction: given a KNOWN unbalance, predict the 1X response.

        This is the capability that separates a twin from a regression. The tree
        baselines cannot do it at all: they only invert. Validating it on held-out
        conditions answers the defining question, whether the identified operator
        predicts measurements it was not fitted to.

        `U_true` is (n, 4) as [U1_re, U1_im, U2_re, U2_im] in g.mm. Returns the
        predicted response interleaved as [Re V_1, Im V_1, Re V_2, Im V_2, ...],
        matching `measured_response`, in the original response units.

        Note this uses the label as an INPUT, not as a target, so it is a genuine
        forward test rather than a restatement of the inverse fit.
        """
        V, w, _ = self._split(np.asarray(X, float))
        u = np.asarray(U_true, float) / self.sU_
        t = lambda a: torch.tensor(a, dtype=torch.float64)
        with torch.no_grad():
            A = self.op_.real_system(t(w))
            pred = (A @ t(u).unsqueeze(-1)).squeeze(-1)
        return pred.numpy() * self.sV_

    def measured_response(self, X) -> np.ndarray:
        """The measured response for the same rows, in the same interleaved order."""
        V, _, _ = self._split(np.asarray(X, float))
        return self._b(V)

    # ------------------------------------------------------- reporting helpers
    def n_parameters(self) -> dict:
        """Parameter counts, distinguishing ACTIVE from inert.

        The backward-whirl residues exist as tensors even when `anisotropic=False`,
        but they are multiplied by zero and receive no gradient, so they must not be
        counted when reporting the isotropic model's complexity in the manuscript.
        """
        def n(pred):
            return sum(p.numel() for k, p in self.op_.named_parameters() if pred(k))
        phys_all = n(lambda k: not k.startswith("net."))
        backward = n(lambda k: ("_b_" in k))
        res = n(lambda k: k.startswith("net."))
        active = phys_all - (0 if self.op_.anisotropic else backward)
        return {"physics_active": active, "physics_inert": phys_all - active,
                "residual": res, "total_active": active + res}

    def identified_poles(self) -> dict:
        """Fitted poles in rpm, next to the prior they were anchored to.

        A small deviation does NOT mean the pole was measured -- with two
        sub-critical speeds it mostly means `lambda_prior` held it. Report the
        deviation, not a claim of identification.
        """
        with torch.no_grad():
            wm, zm = self.op_.poles()
        return {"w_rpm": (wm.numpy() / RPM_TO_RADS).round(1).tolist(),
                "w_prior_rpm": (self.op_.w_prior.numpy() / RPM_TO_RADS).round(1).tolist(),
                "zeta": zm.numpy().round(4).tolist(),
                "zeta_prior": self.op_.zeta_prior.numpy().round(4).tolist(),
                "caveat": "prior-anchored; 2 sub-critical speeds cannot identify poles"}

    def anisotropy_strength(self) -> float:
        """‖Hb‖/‖Hf‖ at the reference speed — how much anisotropy the fit actually used."""
        with torch.no_grad():
            Hf, Hb = self.op_.H(torch.tensor([self.w_ref_], dtype=torch.float64))
            return float(Hb.abs().sum() / Hf.abs().sum().clamp(min=1e-12))
