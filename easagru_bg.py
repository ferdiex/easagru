"""
easagru_bg.py - EASA basal ganglia core for easagru.

A port of the EASA core (github.com/ferdiex/easa, bg_core.py + bg_motor.py),
itself a 1:1 Python port of the C++ controller of Prescott, Montes Gonzalez,
Gurney, Humphries & Redgrave (Neural Networks, 2006).

What is identical to EASA: every neuron model, every constant, every
connection (cortex / TRN / thalamus loop; STN, D1, D2, GPe, GPi nuclei;
recurrent striatal and TRN inhibition), the gating of motor activities
(gate_signals) and their integration (integrate_activity).
test_step4_bg.py checks numerically that, with EASA's own weights, this
module produces the same outputs as the reference copy in easa_ref/.

What is generalised (the task is different, so these are parameters):
  - number of channels and of sensory inputs,
  - the per-channel weight matrices CORTEX_WEIGHTS and SENSORY_WEIGHTS,
  - the motor vector (here: left/right wheel activities and the signal),
  - dopamine D1/D2 (EASA exposes them too).
"""
import numpy as np

# ---- constants, identical to EASA bg_core.py (names kept) -------------------
K = 5.0
TIMESTEP = 0.2
MAXTIME = 0.2

CORTEX_M, CORTEX_T = 1.0, 0.0
THALAMUS_M, THALAMUS_T = 0.62, -0.8
TRN_M, TRN_T = 0.5, 0.0
W_CTXTH, W_BGTH, W_THCTX = 1.0, -1.0, 1.0
W_CTXTRN, W_THTRN, W_TRNTH = 1.0, 1.0, 0.13

SWITCHING_CONSTANT = 2.5
SIGNAL_M, SIGNAL_T = 1.0, 0.0
ACTIVITY_M, ACTIVITY_T = 1.0, 0.0
THALAMIC_TONIC = 1.0

STN_M, STN_T = 0.35, -0.25
W_CTX, W_BSTN, W_GPE_STN = 1.0, 0.1, -1.0
MODULATION, W_EXT, W_BSTR = 1.0, 1.0, 0.1
D1_M, D1_T, W_HSD1 = 0.35, 0.2, 1.0
D2_M, D2_T, W_HSD2 = 0.35, 0.2, 1.0
W_STR, W_STN_GPE, W_STN_GPI = -1.0, 0.8, 0.8
GPE_M, GPE_T, W_GPE_GPI = 1.0, -0.2, -0.4
GPI_M, GPI_T = 1.0, -0.2


def piecewise_linear(a, m, t):
    """leaky.h squashing, vectorised: 0 below t, linear, saturates at 1."""
    return np.clip(m * (np.asarray(a, dtype=float) - t), 0.0, 1.0)


def leak(a, u):
    """leaky.h integration step."""
    return a + (-K * (a - u)) * TIMESTEP


class BasalGanglia:
    """basal_ganglia.h::bg generalised to n channels / m sensors (vectorised)."""

    def __init__(self, cortex_weights, sensory_weights, dopamine_d1, dopamine_d2):
        self.cw = np.asarray(cortex_weights, dtype=float)
        self.sw = np.asarray(sensory_weights, dtype=float)
        self.n = len(self.cw)
        self.d1_da, self.d2_da = dopamine_d1, dopamine_d2
        self.reset()

    def reset(self):
        z = np.zeros(self.n)
        self.a_stn, self.a_d1, self.a_d2, self.a_gpe, self.a_gpi = z.copy(), z.copy(), z.copy(), z.copy(), z.copy()
        self.stn_out, self.d1_out, self.d2_out, self.gpe_out = z.copy(), z.copy(), z.copy(), z.copy()
        self.dot = z.copy()

    def output(self, cortex, sensors, bhfeedback):
        cortex, sensors, bh = (np.asarray(v, dtype=float) for v in (cortex, sensors, bhfeedback))
        rec_d1 = -(self.d1_out.sum() - self.d1_out) * W_HSD1   # uses last step's outputs
        rec_d2 = -(self.d2_out.sum() - self.d2_out) * W_HSD2
        self.dot = self.sw @ sensors

        self.a_stn = leak(self.a_stn, cortex * (W_CTX + self.cw) + self.dot + bh * W_BSTN + self.gpe_out * W_GPE_STN)
        self.stn_out = piecewise_linear(self.a_stn, STN_M, STN_T)
        total_stn = self.stn_out.sum()

        drive = cortex * (W_EXT + self.cw) + self.dot + bh * W_BSTR
        self.a_d1 = leak(self.a_d1, (MODULATION + self.d1_da) * drive + rec_d1)
        self.d1_out = piecewise_linear(self.a_d1, D1_M, D1_T)
        self.a_d2 = leak(self.a_d2, (MODULATION - self.d2_da) * drive + rec_d2)
        self.d2_out = piecewise_linear(self.a_d2, D2_M, D2_T)

        self.a_gpe = leak(self.a_gpe, self.d2_out * W_STR + total_stn * W_STN_GPE)
        self.gpe_out = piecewise_linear(self.a_gpe, GPE_M, GPE_T)
        self.a_gpi = leak(self.a_gpi, self.d1_out * W_STR + total_stn * W_STN_GPI + self.gpe_out * W_GPE_GPI)
        return piecewise_linear(self.a_gpi, GPI_M, GPI_T)


class BGController:
    """bg_controller.c main loop (thalamocortical part), generalised."""

    def __init__(self, cortex_weights, sensory_weights, dopamine_d1=0.2, dopamine_d2=0.2):
        self.bg = BasalGanglia(cortex_weights, sensory_weights, dopamine_d1, dopamine_d2)
        self.n = self.bg.n
        self.reset()

    def reset(self):
        self.bg.reset()
        z = np.zeros(self.n)
        self.a_ctx, self.a_trn, self.a_th = z.copy(), z.copy(), z.copy()
        self.thalamus = z.copy()

    def step(self, sensors, bhfeedback=None):
        bh = np.zeros(self.n) if bhfeedback is None else bhfeedback
        gpi = np.zeros(self.n)
        for _ in range(max(1, int(round(MAXTIME / TIMESTEP)))):
            self.a_ctx = leak(self.a_ctx, self.thalamus * W_THCTX)
            ctx = piecewise_linear(self.a_ctx, CORTEX_M, CORTEX_T)
            self.a_trn = leak(self.a_trn, ctx * W_CTXTRN + self.thalamus * W_THTRN)
            trn = piecewise_linear(self.a_trn, TRN_M, TRN_T)
            gpi = self.bg.output(ctx, sensors, bh)
            rec_trn = -(trn.sum() - trn) * W_TRNTH
            self.a_th = leak(self.a_th, ctx * W_CTXTH + gpi * W_BGTH + rec_trn)
            self.thalamus = piecewise_linear(self.a_th, THALAMUS_M, THALAMUS_T)
        return gpi


def gate_signals(gpi, activities):
    """motor.h gate: a channel's motor activity passes only if its GPi output is low."""
    return piecewise_linear((1.0 - SWITCHING_CONSTANT * np.asarray(gpi)[:, None]) * activities,
                            SIGNAL_M, SIGNAL_T)


def integrate_activity(gated):
    """motor.h: sum the gated activities of all channels, squash to [0, 1]."""
    return piecewise_linear(gated.sum(axis=0), ACTIVITY_M, ACTIVITY_T)
