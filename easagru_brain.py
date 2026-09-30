"""
easagru_brain.py - the controllers that drive one e-puck.

Two controllers share the same Res-GRU:
  - GRUOnlyController: the Res-GRU drives the wheels and the signal directly.
  - EASAController:    the EASA basal ganglia arbitrate between channels;
                       the Res-GRU is one of them (the learned, "cortical"
                       channel) and bids for control.

EASA variants (experimental conditions, see CONDITIONS at the end):
  "eat"   explore, avoid, eat, gru   - innate channels know the task
                                        (stop on food). Run easa_s1 showed
                                        that evolution then silences the
                                        GRU (~1% control, no signalling).
  "safe"  explore, avoid, gru        - innate channels only keep the robot
                                        safe; everything about the task
                                        (stopping on food, signalling,
                                        following the signal) must be
                                        learned by the GRU. The BG gets no
                                        food detector: only obstacle, bid,
                                        fear, hunger. Satiation (hunger
                                        falls on food) lowers the explore
                                        drive, which is the only way the
                                        food reaches the BG.

Channels of EASAController (index: name - what it does when selected):
  0 explore   - wander: forward with a slowly changing random turn
  1 avoid     - turn away from the side with the stronger front IR reading
  2 eat       - on food: stop (and stay). It does NOT emit.
  3 gru       - the Res-GRU's own wheel and signal commands

Communication is NOT innate: only the Res-GRU can emit the signal and only
the Res-GRU sees it. The BG does not receive the signal as a detector, so it
cannot hard-wire a response to it. A first version had innate "emit on food"
and "approach the signal" channels; the smoke run showed that they solved the
task with the GRU in control ~1% of the time, i.e. signalling was designed,
not evolved. They were removed.

BG sensory vector (5 entries, EASA conventions: detectors +1 present / -1
absent, drives in [0, 1]):
  0 obstacle   - front IR above threshold
  1 on_food    - mean ground reading below threshold
  2 gru_bid    - the Res-GRU's bid for control, sigmoid output in [0, 1]
  3 fear       - decays over time (EASA motivational system)
  4 hunger     - grows over time, falls while eating (EASA motivational system)

The weight matrices below are HAND-SET starting values ("BG by hand"): the
experiment later tunes them (CMA-ES). They are not claimed to be optimal.
"""
import numpy as np

from easagru_bg import BGController, gate_signals, integrate_activity

# ------------------------------------------------------------------ channels
ALL_CHANNELS = ["explore", "avoid", "eat", "gru"]   # union, used for logs/plots
CHANNEL_NAMES = ALL_CHANNELS                           # kept for log/plot imports

VARIANTS = {
    # channels, sensor names, SENSORY_WEIGHTS (rows = channels), CORTEX_WEIGHTS
    "eat": {
        "channels": ["explore", "avoid", "eat", "gru"],
        "sensors": ["obstacle", "on_food", "gru_bid", "fear", "hunger"],
        #          obst  food  bid   fear  hunger
        "sw": [[-0.3, -0.3, -0.3, 0.0, 0.6],    # explore
               [0.9, -0.3, 0.0, 0.5, 0.0],      # avoid
               [0.0, 0.9, 0.0, 0.0, 0.5],       # eat
               [-0.3, 0.0, 1.2, 0.0, 0.0]],     # gru (can win on food to signal)
        "cw": [0.3, 0.3, 0.5, 0.3],
    },
    "safe": {
        "channels": ["explore", "avoid", "gru"],
        "sensors": ["obstacle", "gru_bid", "fear", "hunger"],
        #          obst  bid   fear  hunger
        "sw": [[-0.3, -0.3, 0.0, 0.6],          # explore
               [0.9, 0.0, 0.5, 0.0],            # avoid
               [-0.3, 1.2, 0.0, 0.0]],          # gru
        "cw": [0.3, 0.3, 0.3],
    },
}
# Backward-compatible names (variant "eat")
SENSORY_WEIGHTS = np.array(VARIANTS["eat"]["sw"])
CORTEX_WEIGHTS = np.array(VARIANTS["eat"]["cw"])

# Motor vector (EASA style: separate backward / forward activities)
LEFT_B, LEFT_F, RIGHT_B, RIGHT_F, EMIT = range(5)
MOTOR_VECTOR = 5

IR_OBSTACLE = 300.0        # raw IR, ~2 cm
FOOD_THRESHOLD = 500.0     # mean ground reading
FEAR_DECAY = 0.999         # EASA bg_sensors.py, per control step
HUNGER_GROWTH = 1.005      # EASA bg_sensors.py, per control step
SATIATION = 0.995          # our addition: hunger falls while on food

# Initial (fear, hunger) per robot. The two robots start slightly different
# (0.1 apart) so their motivations are distinguishable and not in lockstep.
# Hunger cannot start at 0: EASA hunger grows multiplicatively (x1.005 per
# step), so 0 would stay 0 forever.
MOTIVATION_INIT = [(1.0, 0.1), (0.9, 0.2)]


def activity_from_command(left, right, emit):
    """(left, right in [-1, 1] of max speed, emit bool) -> motor activity vector."""
    a = np.zeros(MOTOR_VECTOR)
    a[LEFT_B], a[LEFT_F] = max(0.0, -left), max(0.0, left)
    a[RIGHT_B], a[RIGHT_F] = max(0.0, -right), max(0.0, right)
    a[EMIT] = 1.0 if emit else 0.0
    return a


def command_from_activity(m):
    return m[LEFT_F] - m[LEFT_B], m[RIGHT_F] - m[RIGHT_B], bool(m[EMIT] > 0.5)


# ------------------------------------------------------------------- Res-GRU
N_IN, N_HID, N_OUT = 17, 16, 4   # outputs: left, right, emit logit, bid logit


def encode_observation(obs, fear, hunger):
    """Normalise one robot's observation into the Res-GRU input vector (17)."""
    ir = np.clip((np.log(obs["ir"]) - np.log(67.19)) / (np.log(2000.0) - np.log(67.19)), 0.0, 1.0)
    ground = np.clip((856.0 - obs["ground"]) / (856.0 - 171.0), 0.0, 1.0)
    s = obs["signal"]
    rec = 1.0 if s["received"] else 0.0
    sig = [rec, rec * np.sin(s["bearing"]), rec * np.cos(s["bearing"]),
           rec * min(1.0, s["strength"] / 100.0)]  # strength 100 = 10 cm
    return np.concatenate([ir, ground, sig, [fear, hunger]])


def res_gru_shapes(n_in):
    return [("w_gru", (n_in + N_HID, 3 * N_HID)), ("b_gru", (3 * N_HID,)),
            ("w_out", (N_HID, N_OUT)), ("b_out", (N_OUT,)), ("w_res", (n_in, N_OUT))]


def res_gru_params(n_in):
    return sum(int(np.prod(s)) for _, s in res_gru_shapes(n_in))


class ResGRU:
    """GRU (16 units) with a residual input->output path, as in the earlier
    GRU-Residual work, rewritten here with continuous outputs. n_in is 17
    by default (19 for the gru_compass condition)."""

    SHAPES = res_gru_shapes(N_IN)
    N_PARAMS = res_gru_params(N_IN)

    def __init__(self, genome=None, n_in=N_IN):
        self.n_in = n_in
        self.SHAPES = res_gru_shapes(n_in)
        self.N_PARAMS = res_gru_params(n_in)
        self.set_genome(np.zeros(self.N_PARAMS) if genome is None else genome)
        self.reset()

    def set_genome(self, genome):
        genome = np.asarray(genome, dtype=float)
        assert genome.size == self.N_PARAMS, f"genome has {genome.size} params, expected {self.N_PARAMS}"
        i = 0
        for name, shape in self.SHAPES:
            n = int(np.prod(shape))
            setattr(self, name, genome[i:i + n].reshape(shape))
            i += n

    def reset(self):
        self.h = np.zeros(N_HID)

    def forward(self, x):
        H = N_HID
        g = np.concatenate([x, self.h]) @ self.w_gru + self.b_gru
        z = 1.0 / (1.0 + np.exp(-g[:H]))
        r = 1.0 / (1.0 + np.exp(-g[H:2 * H]))
        h_tilde = np.tanh(np.concatenate([x, r * self.h]) @ self.w_gru[:, 2 * H:] + self.b_gru[2 * H:])
        self.h = (1 - z) * self.h + z * h_tilde
        y = self.h @ self.w_out + self.b_out + x @ self.w_res
        return {"left": float(np.tanh(y[0])), "right": float(np.tanh(y[1])),
                "emit": bool(y[2] > 0.0), "bid": float(1.0 / (1.0 + np.exp(-y[3])))}


def random_genome(rng, scale=0.5):
    return rng.normal(0.0, scale, ResGRU.N_PARAMS)


# --------------------------------------------------------- innate behaviours
class Innate:
    """Movement generators for the three innate channels."""

    def __init__(self, rng):
        self.rng = rng
        self.turn, self.timer = 0.0, 0

    def reset(self):
        self.turn, self.timer = 0.0, 0

    def explore(self, obs):
        self.timer -= 1
        if self.timer <= 0:
            self.turn = self.rng.uniform(-0.5, 0.5)
            self.timer = int(self.rng.integers(16, 48))   # 1-3 s
        return 1.0 - self.turn, 1.0 + self.turn, False

    @staticmethod
    def avoid(obs):
        ir = obs["ir"]
        if max(ir[0], ir[1]) > max(ir[6], ir[7]):
            return -0.5, 0.5, False    # obstacle on the right: spin left
        return 0.5, -0.5, False

    @staticmethod
    def eat(obs):
        return 0.0, 0.0, False


class Motivation:
    """EASA fear / hunger dynamics (bg_sensors.py), with satiation on food."""

    def __init__(self, fear_init=1.0, hunger_init=0.1):
        self.init = (fear_init, hunger_init)
        self.reset()

    def reset(self):
        self.fear, self.hunger = self.init

    def step(self, on_food):
        self.fear *= FEAR_DECAY
        self.hunger = float(np.clip(self.hunger * (SATIATION if on_food else HUNGER_GROWTH), 0.0, 1.0))


# ---------------------------------------------------------------- controllers
class GRUOnlyController:
    """Condition 'GRU only': the network drives wheels and signal directly."""

    def __init__(self, genome, rng=None, robot_index=0):
        self.net = ResGRU(genome)
        self.motivation = Motivation(*MOTIVATION_INIT[robot_index])

    def reset(self):
        self.net.reset()
        self.motivation.reset()

    def act(self, obs):
        self.motivation.step(obs["on_food"])
        out = self.net.forward(encode_observation(obs, self.motivation.fear, self.motivation.hunger))
        return out["left"], out["right"], out["emit"], {"gru_bid": out["bid"], "fear": self.motivation.fear,
                                                        "hunger": self.motivation.hunger}


class EASAController:
    """Condition 'GRU + EASA': the basal ganglia select among the channels of
    the chosen variant ("eat" or "safe"); the Res-GRU is one of them."""

    def __init__(self, genome, rng=None, robot_index=0, variant="eat", dopamine_d1=0.2,
                 dopamine_d2=0.2, sensory_weights=None, cortex_weights=None):
        v = VARIANTS[variant]
        self.variant = variant
        self.channels = v["channels"]
        self.sensor_names = v["sensors"]
        self.gru_index = self.channels.index("gru")
        self.rng = rng if rng is not None else np.random.default_rng()
        self.net = ResGRU(genome)
        self.innate = Innate(self.rng)
        self.motivation = Motivation(*MOTIVATION_INIT[robot_index])
        sw = np.array(v["sw"]) if sensory_weights is None else sensory_weights
        cw = np.array(v["cw"]) if cortex_weights is None else cortex_weights
        self.bg = BGController(cw, sw, dopamine_d1, dopamine_d2)
        self.bhfeedback = np.zeros(len(self.channels))

    def reset(self):
        self.net.reset()
        self.innate.reset()
        self.motivation.reset()
        self.bg.reset()
        self.bhfeedback = np.zeros(len(self.channels))

    def _sensors(self, obs, bid):
        m = self.motivation
        values = {
            "obstacle": 1.0 if max(obs["ir"][[0, 1, 6, 7]]) > IR_OBSTACLE else -1.0,
            "on_food": 1.0 if obs["on_food"] else -1.0,
            "gru_bid": bid,
            "fear": m.fear,
            "hunger": m.hunger,
        }
        return np.array([values[n] for n in self.sensor_names])

    def act(self, obs):
        self.motivation.step(obs["on_food"])
        m = self.motivation
        gru = self.net.forward(encode_observation(obs, m.fear, m.hunger))
        sensors = self._sensors(obs, gru["bid"])
        gpi = self.bg.step(sensors, self.bhfeedback)

        commands = {
            "explore": self.innate.explore(obs),
            "avoid": self.innate.avoid(obs),
            "eat": self.innate.eat(obs),
            "gru": (gru["left"], gru["right"], gru["emit"]),
        }
        activities = np.vstack([activity_from_command(*commands[c]) for c in self.channels])
        gated = gate_signals(gpi, activities)
        self.bhfeedback = gated.max(axis=1)          # behaviour feedback: channel is acting
        left, right, emit = command_from_activity(integrate_activity(gated))

        selected = gpi < 1.0 / 2.5                   # gate open: its activity can pass
        # also expressed over ALL_CHANNELS (NaN / False where the variant has no such channel)
        gpi_all = np.full(len(ALL_CHANNELS), np.nan)
        th_all = np.full(len(ALL_CHANNELS), np.nan)
        sel_all = np.zeros(len(ALL_CHANNELS), dtype=bool)
        for i, c in enumerate(self.channels):
            k = ALL_CHANNELS.index(c)
            gpi_all[k], th_all[k], sel_all[k] = gpi[i], self.bg.thalamus[i], selected[i]
        diag = {"gpi": gpi_all, "thalamus": th_all, "selected": sel_all,
                "gru_selected": bool(selected[self.gru_index]),
                "sensors": sensors, "gru_bid": gru["bid"], "fear": m.fear, "hunger": m.hunger}
        return left, right, emit, diag


# ------------------------------------------------- evolvable basal ganglia
# Condition "easa_evo": the "safe" EASA variant whose priorities evolve with
# the Res-GRU. The genome is [Res-GRU params | BG genes]. BG genes (17):
#   12 sensory weights (3 channels x 4 sensors), decoded w = 1.5 * tanh(g)
#    3 cortex weights (persistence),              decoded c = sigmoid(g)
#    2 dopamine levels D1, D2,                    decoded d = sigmoid(g)
# The bounded decodings keep every value in a range where the EASA dynamics
# stay well behaved. Initial genes decode to the hand-set "safe" values
# (dopamine 0.2) plus small noise, so generation 0 starts like easa_safe.
N_BG_GENES = 12 + 3 + 2
BG_W_SCALE = 1.5


def _logit(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def bg_prior_genes():
    v = VARIANTS["safe"]
    sw = np.arctanh(np.clip(np.array(v["sw"]).ravel() / BG_W_SCALE, -0.999, 0.999))
    cw = _logit(np.array(v["cw"]))
    da = _logit(np.array([0.2, 0.2]))
    return np.concatenate([sw, cw, da])


def decode_bg_genes(genes):
    genes = np.asarray(genes, dtype=float)
    sw = BG_W_SCALE * np.tanh(genes[:12]).reshape(3, 4)
    cw = 1.0 / (1.0 + np.exp(-genes[12:15]))
    d1, d2 = 1.0 / (1.0 + np.exp(-genes[15:17]))
    return sw, cw, float(d1), float(d2)


# ------------------------------------------------------------ compass test
# Condition "gru_compass": the GRU alone plus the privileged inputs of the
# earlier pipeline (bearing to the food, x[8]; proximity to the food, x[9]).
# Test of the hypothesis that such a compass makes the signal redundant.
N_IN_COMPASS = N_IN + 2


class GRUCompassController(GRUOnlyController):
    def __init__(self, genome, rng=None, robot_index=0):
        self.net = ResGRU(genome, n_in=N_IN_COMPASS)
        self.motivation = Motivation(*MOTIVATION_INIT[robot_index])

    def act(self, obs):
        self.motivation.step(obs["on_food"])
        x = encode_observation(obs, self.motivation.fear, self.motivation.hunger)
        compass = [obs["food_bearing"] / np.pi, float(np.exp(-obs["food_distance"] / 0.5))]
        out = self.net.forward(np.concatenate([x, compass]))
        return out["left"], out["right"], out["emit"], {"gru_bid": out["bid"], "fear": self.motivation.fear,
                                                        "hunger": self.motivation.hunger}


# ---------------------------------------------------------------- conditions
CONDITIONS = ("easa", "easa_safe", "easa_evo", "gru", "gru_compass")


def genome_size(condition):
    if condition == "gru_compass":
        return res_gru_params(N_IN_COMPASS)
    return ResGRU.N_PARAMS + (N_BG_GENES if condition == "easa_evo" else 0)


def initial_genome(condition, rng, init_sigma=0.5, bg_sigma=0.1):
    g = rng.normal(0.0, init_sigma, genome_size(condition) - (N_BG_GENES if condition == "easa_evo" else 0))
    if condition == "easa_evo":
        g = np.concatenate([g, bg_prior_genes() + rng.normal(0.0, bg_sigma, N_BG_GENES)])
    return g


def make_controller(condition, genome, rng=None, robot_index=0):
    """easa = EASA variant 'eat' (run easa_s1), easa_safe = variant 'safe',
    easa_evo = variant 'safe' with evolved BG priorities, gru = GRU only."""
    if condition == "easa":
        return EASAController(genome, rng, robot_index, variant="eat")
    if condition == "easa_safe":
        return EASAController(genome, rng, robot_index, variant="safe")
    if condition == "easa_evo":
        genome = np.asarray(genome, dtype=float)
        sw, cw, d1, d2 = decode_bg_genes(genome[ResGRU.N_PARAMS:])
        return EASAController(genome[:ResGRU.N_PARAMS], rng, robot_index, variant="safe",
                              dopamine_d1=d1, dopamine_d2=d2, sensory_weights=sw, cortex_weights=cw)
    if condition == "gru":
        return GRUOnlyController(genome, rng, robot_index)
    if condition == "gru_compass":
        return GRUCompassController(genome, rng, robot_index)
    raise ValueError(f"unknown condition {condition!r}; choose from {CONDITIONS}")
