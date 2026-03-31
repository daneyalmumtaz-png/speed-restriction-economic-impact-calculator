# ftia_jt.py
# Speed Restriction Journey-Time calculator calibrated to FTIA (Sm3) logic.

from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

DT = 0.1               # time step [s]
DECEL = 0.6            # constant braking [m/s^2]

# --- Sm3 (Pendolino) acceleration table from FTIA source (km/h, m/s^2) ---
_SM3_TABLE_KMH = [
    (0, .404),(2,.404),(4,.404),(6,.404),(8,.404),(10,.404),(12,.403),(14,.403),
    (16,.403),(18,.403),(20,.403),(22,.403),(24,.402),(26,.402),(28,.402),(30,.402),
    (32,.401),(34,.401),(36,.401),(38,.401),(40,.401),(42,.400),(44,.400),(46,.400),
    (48,.399),(50,.399),(52,.399),(54,.398),(56,.398),(58,.398),(60,.397),(62,.397),
    (64,.397),(66,.396),(68,.396),(70,.395),(72,.395),(74,.394),(76,.394),(78,.393),
    (80,.393),(82,.393),(84,.392),(86,.392),(88,.381),(90,.370),(92,.359),(94,.356),
    (96,.342),(98,.333),(100,.325),(102,.317),(104,.308),(106,.302),(108,.297),(110,.288),
    (112,.280),(114,.274),(116,.268),(118,.262),(120,.256),(122,.250),(124,.244),(126,.239),
    (128,.232),(130,.227),(132,.223),(134,.217),(136,.211),(138,.208),(140,.202),(142,.198),
    (144,.192),(146,.189),(148,.186),(150,.180),(152,.176),(154,.173),(156,.169),(158,.163),
    (160,.160),(162,.156),(164,.153),(166,.149),(168,.146),(170,.142),(172,.139),(174,.135),
    (176,.131),(178,.128),(180,.124),(182,.121),(184,.117),(186,.116),(188,.113),(190,.109),
    (192,.105),(194,.102),(196,.101),(198,.097),(200,.093),(202,.089),(204,.088),(206,.085),
    (208,.081),(210,.080),(212,.076),(214,.072),(216,.071),(218,.067)
]

_SM3_TABLE = [(kmh/3.6, a) for kmh, a in _SM3_TABLE_KMH]  # convert km/h→m/s


def sm3_accel(v: float) -> float:
    """
    FTIA selection rule: use the last table entry whose speed (m/s) is <= current v.
    If v below first, return first a; if above last, return last a.
    """
    a = _SM3_TABLE[0][1]
    for v_th, a_th in _SM3_TABLE:
        if v_th <= v:
            a = a_th
        else:
            break
    return a


@dataclass
class PhaseResult:
    time: float
    dist: float
    v_end: float


def _brake_to(v0: float, v_target: float) -> PhaseResult:
    """Brake at DECEL until speed <= v_target (stepwise, Euler)."""
    t = 0.0
    s = 0.0
    v = v0
    while v > v_target + 1e-9:
        s += v*DT + 0.5*(-DECEL)*(DT*DT)
        v = max(0.0, v - DECEL*DT)
        t += DT
        if v <= v_target:
            break
    return PhaseResult(time=t, dist=s, v_end=v)


def _plateau_time(v_post_brake: float, v_r: float, lead_m: float, L_m: float, train_len_m: float) -> PhaseResult:
    """
    FTIA plateau implementation: treat as pure time windows k + w,
    using v_r only to set the *duration*, not to snap speed.
    Distance increments with current v (no artificial clamping to v_r).
    """
    if v_r <= 0:
        t_total = 0.0
    else:
        k = lead_m / v_r
        w = (L_m + train_len_m) / v_r
        t_total = k + w

    n_steps = int(round(t_total / DT))
    t = 0.0
    s = 0.0
    v = v_post_brake
    for _ in range(n_steps):
        s += v*DT                       # no acceleration/deceleration on plateau
        t += DT
    # fine remainder
    rem = max(0.0, t_total - n_steps*DT)
    s += v*rem
    t += rem
    return PhaseResult(time=t, dist=s, v_end=v)


def _accelerate_to(v0: float, v_target: float) -> PhaseResult:
    """Accelerate with Sm3 table stepwise until v >= v_target."""
    t = 0.0
    s = 0.0
    v = v0
    while v < v_target - 1e-9:
        a = sm3_accel(v)
        s += v*DT + 0.5*a*(DT*DT)
        v = v + a*DT
        t += DT
        if v >= v_target:
            break
    return PhaseResult(time=t, dist=s, v_end=v)


@dataclass
class RestrictedRun:
    t_brake: float
    s_brake: float
    t_plateau: float
    s_plateau: float
    t_accel: float
    s_accel: float
    t_total: float
    s_total: float


def restricted_time(v0_kmh: float, vt_kmh: float, vr_kmh: float,
                    L_m: float, train_len_m: float, lead_m: float) -> RestrictedRun:
    """FTIA-like restricted run composed of brake → plateau → accelerate."""
    v0 = v0_kmh/3.6
    vt = vt_kmh/3.6
    vr = vr_kmh/3.6

    # 1) Brake to vr
    br = _brake_to(v0, vr)

    # 2) Plateau (time windows)
    pl = _plateau_time(br.v_end, vr, lead_m, L_m, train_len_m)

    # 3) Accelerate to vt
    ac = _accelerate_to(pl.v_end, vt)

    t_tot = br.time + pl.time + ac.time
    s_tot = br.dist + pl.dist + ac.dist

    return RestrictedRun(
        t_brake=br.time, s_brake=br.dist,
        t_plateau=pl.time, s_plateau=pl.dist,
        t_accel=ac.time, s_accel=ac.dist,
        t_total=t_tot, s_total=s_tot
    )


# ---------- Baseline over the same distance (minimal-time heuristic) ----------

def _time_brake_segment(v_start: float, v_end: float) -> Tuple[float, float]:
    """Exact braking time & distance at constant decel."""
    t = (v_start - v_end) / DECEL
    s = (v_start*v_start - v_end*v_end) / (2*DECEL)
    return max(0.0, t), max(0.0, s)


def baseline_time_over_distance(v0_kmh: float, vt_kmh: float, D: float) -> float:
    """
    Minimal-time baseline to cover distance D with boundary speeds v0 and vt.
    Heuristic aligned to FTIA practicality:
      - If v0 <= vt: accelerate with Sm3 up to vt (or fractionally), then cruise.
      - If v0  > vt: cruise at v0 then brake to vt; if D is too short, brake immediately
        and end above vt (physically minimal time).
    """
    v0 = v0_kmh/3.6
    vt = vt_kmh/3.6

    if D <= 1e-9:
        return 0.0

    # Case A: Up-rate (v0 <= vt)
    if v0 <= vt:
        # accelerate with Sm3 until vt (or until we run out of D)
        t = 0.0
        s = 0.0
        v = v0
        while v < vt - 1e-9 and s < D - 1e-6:
            a = sm3_accel(v)
            ds = v*DT + 0.5*a*(DT*DT)
            dv = a*DT
            if s + ds >= D:   # finish fractionally inside this step
                # linearized fraction
                rem = (D - s) / max(1e-12, v)  # conservative finish
                t += rem
                return t
            s += ds
            v += dv
            t += DT

        # If still distance left after reaching vt, cruise at vt
        if s < D:
            t += (D - s) / max(1e-12, vt)
        return t

    # Case B: Down-rate (v0 > vt)
    s_brake_needed = max(0.0, (v0*v0 - vt*vt)/(2*DECEL))
    if D >= s_brake_needed:
        # cruise then brake
        t_cruise = (D - s_brake_needed) / max(1e-12, v0)
        t_brake, _ = _time_brake_segment(v0, vt)
        return t_cruise + t_brake
    else:
        # too short to reach vt at end; brake immediately and finish above vt
        v_end_sq = max(0.0, v0*v0 - 2*DECEL*D)
        v_end = v_end_sq**0.5
        t_brake_only = (v0 - v_end)/DECEL
        return t_brake_only


@dataclass
class ImpactResult:
    phases: Dict[str, float]
    distances: Dict[str, float]
    total_restricted_s: float
    total_distance_m: float
    baseline_s: float
    impact_s: float


def compute_impact(v0_kmh: float, vt_kmh: float,
                   vr_kmh: float, L_m: float,
                   train_len_m: float = 160.0, lead_m: float = 0.0) -> ImpactResult:
    rr = restricted_time(v0_kmh, vt_kmh, vr_kmh, L_m, train_len_m, lead_m)
    t_base = baseline_time_over_distance(v0_kmh, vt_kmh, rr.s_total)
    impact = rr.t_total - t_base
    return ImpactResult(
        phases={
            "brake_s": rr.t_brake,
            "plateau_s": rr.t_plateau,
            "accel_s": rr.t_accel
        },
        distances={
            "brake_m": rr.s_brake,
            "plateau_m": rr.s_plateau,
            "accel_m": rr.s_accel
        },
        total_restricted_s=rr.t_total,
        total_distance_m=rr.s_total,
        baseline_s=t_base,
        impact_s=impact
    )


# -------- Convenience: batch runner and pretty printer --------

def run_scenario(v0_kmh: float, vt_kmh: float, vr_kmh: float,
                 L_m: float, train_len_m: float = 160.0, lead_m: float = 0.0) -> Dict[str, Any]:
    res = compute_impact(v0_kmh, vt_kmh, vr_kmh, L_m, train_len_m, lead_m)
    out = {
        "v0_kmh": v0_kmh,
        "vt_kmh": vt_kmh,
        "vr_kmh": vr_kmh,
        "L_m": L_m,
        "train_len_m": train_len_m,
        "lead_m": lead_m,
        "t_brake_s": round(res.phases["brake_s"], 3),
        "t_plateau_s": round(res.phases["plateau_s"], 3),
        "t_accel_s": round(res.phases["accel_s"], 3),
        "t_restricted_s": round(res.total_restricted_s, 3),
        "D_m": round(res.total_distance_m, 3),
        "t_baseline_s": round(res.baseline_s, 3),
        "impact_s": round(res.impact_s, 3)
    }
    return out


if __name__ == "__main__":
    # Example: 200→170 with an 80 km/h restriction for 1000 m, Pendolino, no lead.
    example = run_scenario(v0_kmh=200, vt_kmh=170, vr_kmh=80, L_m=1000, train_len_m=160, lead_m=0)
    from pprint import pprint
    pprint(example)
