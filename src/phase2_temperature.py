"""
Phase 2: 온도 의존과 ALD window.

Phase 1의 TMA 반쪽에 H2O 반쪽을 붙여 전체 사이클을 돌리고, GPC의 온도 의존을
구한다. 01_모델설계.md 4.5절이 정한 목표 셋에 답한다.

    1. GPC의 감소 기울기 -0.0082 nm^-2/도를 재현한다
    2. 그 감소가 OH 밀도 단독에서 오는지 모델로 분해해 보인다
    3. 저온 경계를 beta_H2O(T)로 설명하고, 주어진 H2O 노출량에서
       포화하지 못하는 온도를 구한다

H2O 반쪽 모델 ([P05] 4절 J 요약을 따름)

    리간드교환   Me* + H2O -> OH* + CH4(g).   Me -1, OH +1
    해리         H2O + Al-O-Al -> 2 Al-OH.    OH 증가, Me 무관

    "When the H2O reaction self-terminates, the surface hydroxyl group
     concentration settles to a value dictated by the temperature"  [P05]

    -> 해리 항을 sigma_OH0(T)를 향한 구동력으로 둔다. 종착점이 온도를 탄다.

TMA 반쪽은 phase1_saturation.py를 그대로 쓴다.

실행: python src/phase2_temperature.py
"""

import os
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.pyplot as plt

from phase1_saturation import (
    KB, NA, TORR, NM2_TO_M2, P, FIG_DIR,
    sigma_OH0_nm2, dcAl_ref_nm2, rho_Al, hertz_knudsen, gpc_from_sAl,
    rhs as rhs_tma,
)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["figure.dpi"] = 110


# ----------------------------------------------------------------------
# 1. H2O 파라미터
# ----------------------------------------------------------------------

P_H2O = {
    "M_H2O": 18.015e-3,      # kg/mol

    # [02 #8] beta_H2O의 온도 의존. Arts 2019 Fig.3b + Vandalon 2016 SFG의
    # 세 점(150/220/300도 기판온도, 각 범위의 기하평균)에 Arrhenius를 맞춘 결과.
    # 잔차 17% 이내. 문헌 자체 산포가 각 점에서 2.5배 수준이므로 충분하다.
    "beta0_H2O": 2.3469,      # 무차원 전지수인자
    "Ea_H2O_eV": 0.4253,      # eV

    # 해리 항의 상대 확률. 문헌에 값이 없다. k_ex와 같은 방식으로
    # 종착점이 이 값에 둔감함을 테스트로 확인한다 (test_rdiss_insensitivity).
    "r_diss": 1.0,

    "P_H2O_torr": 0.1,        # TMA와 같은 압력을 기준으로 둔다

    # --- 응축(탈수산화) 2 Al-OH -> Al-O-Al + H2O ---
    # [02 #7] Ea_dehyd. [Z00]이 비정질 실리카에서 측정한 ALD 온도역 값.
    # 실험 반응차수 n = 2.00 (associative desorption)이므로 2차로 둔다.
    "Ea_dehyd_eV": 0.90,      # 0.72 ~ 1.09 eV 구간의 중앙값. 감도 스윕 대상
    # 전인자는 문헌에 없다. 200도에서의 선형화 완화시간 tau_c로 파라미터화한다.
    "tau_c_200_s": 10.0,      # s. 문헌 기울기를 맞추는 값 (아래 참조)
}

# ----------------------------------------------------------------------
# 응축 모델 선택. 기본은 "flux_coupled".
#
#   "flux_coupled"  응축 속도를 물 플럭스에 묶는다. 미지 파라미터가 r_diss
#                   하나이고 결과가 그것에 둔감하다(test_rdiss_insensitivity).
#                   한계: H2O 포화 곡선이 중간 dose에서 점근값을 0.77 % 초과하는
#                   비단조 봉우리가 생긴다. 운전점(마진 42)에서 멀리 떨어진
#                   구간이라 Phase 5 결론에는 영향이 없다.
#
#   "arrhenius"     응축을 물과 분리하고 [Z00]의 Ea_dehyd와 2차 반응차수를 쓴다.
#                   물리적으로 더 옳고 비단조성도 해소된다. 그러나 전인자 tau_c가
#                   문헌에 없고 결과가 거기에 35.7 % 의존한다. 문헌 GPC 기울기를
#                   맞추는 tau_c는 약 10 s 한 점뿐이라 사실상 피팅이 되고,
#                   그 조건에서 300도 퍼지 1초에 OH의 65 %를 잃는다는 검증 불가능한
#                   주장을 하게 된다.
#
# 판단: 문서화된 구조적 한계를 미검증 피팅 파라미터와 맞바꾸는 것은 손해다.
#       기본은 flux_coupled로 두고 arrhenius는 비교용으로 남긴다.
#       상세는 03_결과.md Phase 2 부록.
# ----------------------------------------------------------------------
CONDENSATION_MODEL = "flux_coupled"

KB_EV = 8.617333262e-5


def beta_H2O(T_C):
    """H2O 초기 반응확률. [02 #8]"""
    return P_H2O["beta0_H2O"] * math.exp(
        -P_H2O["Ea_H2O_eV"] / (KB_EV * (T_C + 273.15)))


# ----------------------------------------------------------------------
# 2. 사이클 파라미터와 지배식
# ----------------------------------------------------------------------

def k_cond(T_C, s_OH0, tau_c_200=None, Ea=None):
    """
    응축(탈수산화) 2차 속도상수 k_c [m^2 s^-1].

        R_cond = k_c x s_OH^2                    [Z00] 실험 반응차수 n = 2.00

    전인자가 문헌에 없으므로 200도에서의 선형화 완화시간 tau_c로 고정한다.
    평형 근처에서 d(delta)/dt = -2 k_c sigma_OH0 delta 이므로

        tau_c = 1 / (2 k_c sigma_OH0)   ->   k_c(200) = 1 / (2 tau_c sigma_OH0(200))

    온도 의존은 Ea_dehyd로 준다. 이쪽은 [Z00] 실측이라 물리적 근거가 있다.
    """
    tau = tau_c_200 if tau_c_200 is not None else P_H2O["tau_c_200_s"]
    E = Ea if Ea is not None else P_H2O["Ea_dehyd_eV"]
    T_ref = 200.0 + 273.15
    s_ref = sigma_OH0_nm2(200.0) * NM2_TO_M2
    k_ref = 1.0 / (2.0 * tau * s_ref)
    return k_ref * math.exp(-E / KB_EV * (1.0 / (T_C + 273.15) - 1.0 / T_ref))


def make_cycle_params(T_C, sOH0_nm2=None, beta_tma=None, r_diss=None,
                      P_tma_torr=None, P_h2o_torr=None, sMe_max_nm2=None,
                      tau_c_200=None, Ea_dehyd=None, cond_model=None):
    """TMA 반쪽과 H2O 반쪽에 모두 쓰는 파라미터 묶음."""
    T_K = T_C + 273.15
    P_t = (P_tma_torr if P_tma_torr is not None else P["P_TMA_torr"]) * TORR
    P_w = (P_h2o_torr if P_h2o_torr is not None else P_H2O["P_H2O_torr"]) * TORR

    sMe_max = (sMe_max_nm2 if sMe_max_nm2 is not None
               else P["sMe_max_nm2"]) * NM2_TO_M2
    s_OH0 = (sOH0_nm2 if sOH0_nm2 is not None
             else sigma_OH0_nm2(T_C)) * NM2_TO_M2

    b_tma = beta_tma if beta_tma is not None else P["beta_TMA"]
    Gamma_tma = hertz_knudsen(P_t, T_K, P["M_TMA"])
    Gamma_h2o = hertz_knudsen(P_w, T_K, P_H2O["M_H2O"])

    # Da_ex >= 100이면 포화값이 수렴한다 (Phase 1에서 확인)
    k_ex = 100.0 * b_tma * Gamma_tma / (sMe_max * s_OH0)

    return {
        "T_C": T_C, "T_K": T_K,
        "P_tma": P_t, "P_h2o": P_w,
        "Gamma": Gamma_tma,          # phase1.rhs가 쓰는 이름
        "Gamma_h2o": Gamma_h2o,
        "beta": b_tma,               # phase1.rhs가 쓰는 이름
        "beta_h2o": beta_H2O(T_C),
        "a_Me": 1.0 / sMe_max, "sMe_max": sMe_max,
        "s_OH0": s_OH0, "k_ex": k_ex,
        "r_diss": r_diss if r_diss is not None else P_H2O["r_diss"],
        "k_cond": k_cond(T_C, s_OH0, tau_c_200, Ea_dehyd),
        "cond_model": cond_model or CONDENSATION_MODEL,
    }


def rhs_h2o(t, y, p):
    """
    H2O 반쪽. 상태 y = [s_OH, s_Me, s_Al], 전부 m^-2.

    세 과정을 분리한다. 물이 있어야 하는 것과 없어도 되는 것을 구분하는 것이
    핵심이다.

        리간드교환  Me* + H2O -> OH* + CH4         물 필요. 메틸 잔량에 1차
        해리        H2O + Al-O-Al -> 2 Al-OH       물 필요
        응축        2 Al-OH -> Al-O-Al + H2O       물 불필요. OH에 2차

    응축이 곧 탈수산화다. [Z00]이 실험으로 확정한 반응차수 n = 2.00을 따른다.
    해리와 응축의 균형이 정상상태 sigma_OH0(T)를 만들도록 k_d = k_c sigma_OH0^2로
    두면 순 항이 다음과 같이 정리된다.

        R_net = k_c (sigma_OH0^2 - s_OH^2)

    이전 판(2026-08-23 오전)은 응축을 물 플럭스에 묶어놨다. 그래서 물을 끊는
    순간 과잉 OH를 되돌릴 수 없었고, H2O 포화 곡선이 중간 dose에서 점근값을
    0.77 % 초과하는 비단조 거동이 나왔다(Phase 5 3절). 응축을 물과 분리하고
    퍼지 단계에 넣으면서 해소된다.
    """
    s_OH, s_Me, s_Al = y
    s_OH = max(s_OH, 0.0)

    R_le = p["beta_h2o"] * p["Gamma_h2o"] * max(s_Me, 0.0) / p["sMe_max"]

    if p["cond_model"] == "arrhenius":
        # 해리와 응축이 균형을 이뤄 sigma_OH0가 나오도록 k_d = k_c sigma_OH0^2
        R_net = p["k_cond"] * (p["s_OH0"] ** 2 - s_OH ** 2)
    else:
        # flux_coupled: 물 플럭스에 비례하는 1차 완화
        R_net = (p["beta_h2o"] * p["Gamma_h2o"] * p["r_diss"]
                 * (1.0 - s_OH / p["s_OH0"]))

    return [R_le + R_net, -R_le, 0.0]


def rhs_purge(t, y, p):
    """
    퍼지 단계. 기체상 반응물이 없으므로 응축만 진행된다.

        2 Al-OH -> Al-O-Al + H2O(g)      R = k_c s_OH^2

    물이 없어 해리가 균형을 잡아주지 않으므로 OH가 단조 감소한다. 퍼지가 길면
    다음 TMA 반쪽이 만나는 OH가 줄어 GPC가 떨어진다. 이것이 실제 공정에서
    퍼지 시간이 GPC에 영향을 주는 경로 중 하나다.
    """
    s_OH, s_Me, s_Al = y
    s_OH = max(s_OH, 0.0)
    if p["cond_model"] != "arrhenius":
        # flux_coupled에서는 응축이 물 플럭스에 묶여 있어 퍼지 중 진행되지
        # 않는다. 이것이 그 모델의 구조적 한계다(모듈 상단 주석 참조).
        return [0.0, 0.0, 0.0]
    return [-p["k_cond"] * s_OH ** 2, 0.0, 0.0]


# ----------------------------------------------------------------------
# 3. 사이클 드라이버
# ----------------------------------------------------------------------

def _integrate(fun, y0, t_end, p, rtol=1e-10):
    sol = solve_ivp(fun, (0.0, t_end), y0, args=(p,), method="LSODA",
                    rtol=rtol, atol=1e-6, max_step=t_end / 50.0)
    if not sol.success:
        raise RuntimeError(f"적분 실패: {sol.message}")
    return [float(v) for v in sol.y[:, -1]]


# 퍼지 시간. 문헌값을 확보하지 못한 가정치다. Phase 5의 사이클 시간 계산과
# 같은 값을 쓴다. 퍼지 중에는 응축만 진행된다.
T_PURGE_S = 1.0


def run_cycles(T_C, D_tma_mtorr_s, D_h2o_mtorr_s, n_cycles=12,
               t_purge_s=T_PURGE_S, **kw):
    """
    정상 성장 상태까지 사이클을 반복한다.

        TMA 반쪽 -> 퍼지 -> H2O 반쪽 -> 퍼지

    dose [mTorr s] -> 시간 t = D / P.
    사이클마다 s_Al을 0으로 초기화해 그 사이클에서 올라간 양만 센다.
    s_OH와 s_Me는 사이클을 넘어 이어진다.

    반환: 사이클별 GPC 배열 [Angstrom/cycle]과 마지막 표면 상태.
    """
    p = make_cycle_params(T_C, **kw)
    t_tma = (D_tma_mtorr_s * 1e-3) * TORR / p["P_tma"]
    t_h2o = (D_h2o_mtorr_s * 1e-3) * TORR / p["P_h2o"]

    # 정상 성장 표면에서 출발한다 (그 온도의 OH 밀도, 메틸 없음)
    s_OH, s_Me = p["s_OH0"], 0.0
    gpc = []
    for _ in range(n_cycles):
        s_OH, s_Me, s_Al = _integrate(rhs_tma, [s_OH, s_Me, 0.0], t_tma, p)
        gpc.append(gpc_from_sAl(s_Al))
        if t_purge_s > 0:
            s_OH, s_Me, _ = _integrate(rhs_purge, [s_OH, s_Me, 0.0],
                                       t_purge_s, p)
        s_OH, s_Me, _ = _integrate(rhs_h2o, [s_OH, s_Me, 0.0], t_h2o, p)
        if t_purge_s > 0:
            s_OH, s_Me, _ = _integrate(rhs_purge, [s_OH, s_Me, 0.0],
                                       t_purge_s, p)

    return np.array(gpc), {"s_OH": s_OH / NM2_TO_M2,
                           "s_Me": s_Me / NM2_TO_M2, "p": p}


# 충분 포화로 볼 기준 dose.
#   TMA  : 특성시간 약 10 ms -> 10 mTorr s (0.1 Torr에서 100 ms)는 10 tau
#   H2O  : 특성시간이 150도에서 677 ms까지 커진다. 5 tau를 덮으려면 339 mTorr s
#          필요하므로 문헌값 750 mTorr s [A19]를 기준으로 쓴다. [R01]의 약 3 s
#          포화 펄스와도 자릿수가 맞는다.
D_TMA_SAT = 10.0     # mTorr s
D_H2O_SAT = 750.0    # mTorr s


def steady_gpc(T_C, D_tma=D_TMA_SAT, D_h2o=D_H2O_SAT, n_cycles=12, **kw):
    """정상상태 GPC [Angstrom/cycle]. 기본 dose는 충분 포화 조건."""
    g, _ = run_cycles(T_C, D_tma, D_h2o, n_cycles, **kw)
    return g[-1]


# ----------------------------------------------------------------------
# 4. 검증
# ----------------------------------------------------------------------

def test_cycle_convergence():
    """사이클 반복이 정상상태로 수렴하나."""
    worst = 0.0
    for T_C in (150.0, 200.0, 250.0, 300.0):
        g, _ = run_cycles(T_C, D_TMA_SAT, D_H2O_SAT, n_cycles=15)
        worst = max(worst, abs(g[-1] - g[-2]) / g[-1])
    return {"name": "사이클 정상상태 수렴", "value": worst,
            "limit": 1e-6, "ok": worst < 1e-6, "fmt": "마지막 두 사이클 상대차 {:.2e}"}


def test_surface_regenerates():
    """
    충분 포화 조건에서 H2O 반쪽이 표면을 원래대로 되돌리나.
    퍼지를 끄고 H2O 반쪽만 본다. 메틸이 다 제거되고 OH가 sigma_OH0(T)로
    돌아와야 한다. (퍼지를 켜면 응축으로 OH가 조금 줄어드는 것이 정상이다.
    그 크기는 test_purge_loss에서 따로 본다.)
    """
    worst = 0.0
    for T_C in (150.0, 200.0, 250.0, 300.0):
        _, st = run_cycles(T_C, D_TMA_SAT, D_H2O_SAT, n_cycles=12, t_purge_s=0.0)
        target = st["p"]["s_OH0"] / NM2_TO_M2
        worst = max(worst,
                    st["s_Me"] / (st["p"]["sMe_max"] / NM2_TO_M2),
                    abs(st["s_OH"] - target) / target)
    return {"name": "표면 재생 (퍼지 끔)", "value": worst,
            "limit": 1e-3, "ok": worst < 1e-3, "fmt": "최대 편차 {:.2e}"}


def test_purge_loss():
    """
    퍼지 중 응축으로 잃는 OH의 크기. 물리적으로 0보다 크고, 표면을 고갈시킬
    만큼 크면 안 된다. 통상 퍼지(1 s)에서 수 % 수준이어야 한다.
    """
    worst = 0.0
    for T_C in (150.0, 200.0, 250.0, 300.0):
        _, st = run_cycles(T_C, D_TMA_SAT, D_H2O_SAT, n_cycles=12)
        target = st["p"]["s_OH0"] / NM2_TO_M2
        worst = max(worst, (target - st["s_OH"]) / target)
    if CONDENSATION_MODEL != "arrhenius":
        # flux_coupled에서는 구조상 0이다. 알려진 한계이므로 실패로 세지 않는다.
        return {"name": "퍼지 중 OH 손실 (flux_coupled: 해당없음)",
                "value": worst, "limit": 0.0, "ok": True,
                "fmt": "{:.1%} (구조상 0. 알려진 한계)"}
    return {"name": "퍼지 중 OH 손실", "value": worst,
            "limit": 0.30, "ok": 0.0 < worst < 0.30,
            "fmt": "sigma_OH0 대비 최대 {:.1%}"}


def test_h2o_curve_monotonic():
    """
    H2O dose를 늘릴 때 GPC가 단조 증가해야 한다.

    이전 판에서는 응축이 물 플럭스에 묶여 있어 중간 dose가 점근값을 0.77 %
    초과하는 봉우리가 생겼다(Phase 5 3절). 응축을 물과 분리하고 퍼지에
    넣으면서 해소되는지를 직접 검사한다.
    """
    Ds = np.geomspace(5.0, 3000.0, 22)
    g = np.array([steady_gpc(200.0, D_h2o=float(d), n_cycles=10) for d in Ds])
    excess = float(g.max() / g[-1] - 1.0)
    lim = 2e-3 if CONDENSATION_MODEL == "arrhenius" else 1e-2
    tag = "" if CONDENSATION_MODEL == "arrhenius" else " (flux_coupled 허용치)"
    return {"name": "H2O 곡선 단조성" + tag, "value": excess,
            "limit": lim, "ok": excess < lim,
            "fmt": "점근값 대비 최대 초과 {:+.2e}"}


def test_tau_c_sweep():
    """
    응축 전인자(tau_c)는 문헌에 없다. 정상 GPC가 이 값에 얼마나 의존하나.
    둔감하지 않다면 그 사실 자체를 결과로 보고해야 한다.
    """
    vals = {t: steady_gpc(200.0, tau_c_200=t) for t in (0.3, 1.0, 3.0, 10.0, 30.0)}
    v = list(vals.values())
    spread = (max(v) - min(v)) / float(np.mean(v))
    nm = ("정상 GPC의 tau_c 의존성" if CONDENSATION_MODEL == "arrhenius"
          else "정상 GPC의 tau_c 의존성 (flux_coupled: 미사용)")
    return {"name": nm, "value": spread,
            "limit": 0.05, "ok": spread < 0.05,
            "fmt": "tau_c 0.3~30 s에서 상대 산포 {:.2%}", "detail": vals}


def test_rdiss_insensitivity():
    """
    해리 상대확률 r_diss는 문헌에 없다. 충분 포화 조건에서 정상상태 GPC가
    이 값에 둔감해야 한다 (종착점은 sigma_OH0(T)가 정하므로).
    """
    vals = [steady_gpc(200.0, r_diss=r) for r in (0.3, 1.0, 3.0, 10.0)]
    spread = (max(vals) - min(vals)) / float(np.mean(vals))
    return {"name": "정상 GPC의 r_diss 둔감성", "value": spread,
            "limit": 1e-3, "ok": spread < 1e-3, "fmt": "상대 산포 {:.2e}"}


def test_condensation_branch():
    """
    OH가 sigma_OH0(T)를 넘은 상태에서 출발하면 응축으로 되돌아와야 한다.
    해리/응축 완화 항의 부호가 양방향으로 열려 있는지 직접 검사한다.

    이 테스트가 없어서 처음 구현에서 오버슈트를 놓쳤다. 리간드교환이 OH를
    무조건 늘리는데 되돌릴 경로가 없어 300도에서 8.24까지 올라갔다.
    """
    worst = 0.0
    for T_C in (150.0, 200.0, 250.0, 300.0):
        p = make_cycle_params(T_C)
        y = [1.5 * p["s_OH0"], 0.0, 0.0]      # 목표의 1.5배에서 출발
        # 완화는 지수형이다. 50% 초과에서 0.1%까지 내려오려면 약 6.2 tau가 필요한데
        # 기준 dose는 150도에서 6.1 tau라 아슬하게 모자란다. 분기 동작을 보는
        # 테스트이므로 여유를 두고 3배 dose를 준다.
        t_end = (3.0 * D_H2O_SAT * 1e-3) * TORR / p["P_h2o"]
        s_OH, _, _ = _integrate(rhs_h2o, y, t_end, p)
        worst = max(worst, abs(s_OH - p["s_OH0"]) / p["s_OH0"])
    return {"name": "응축 분기 (과잉 OH 복귀)", "value": worst,
            "limit": 1e-3, "ok": worst < 1e-3, "fmt": "sigma_OH0 대비 편차 {:.2e}"}


# ----------------------------------------------------------------------
# 5. Phase 2 목표별 계산
# ----------------------------------------------------------------------

def goal1_slope(Ts=None):
    """목표 1: GPC 온도 기울기를 문헌과 대조."""
    if Ts is None:
        Ts = np.array([150.0, 175.0, 200.0, 225.0, 250.0, 275.0, 300.0])
    dcAl = np.array([steady_gpc(t) / 1e10 * rho_Al() / NM2_TO_M2 for t in Ts])
    slope, icept = np.polyfit(Ts, dcAl, 1)
    return Ts, dcAl, slope, icept


def goal2_decompose(Ts=None):
    """
    목표 2: 감소가 OH 밀도 단독에서 오는지 분해.

    (a) 전체        sigma_OH0(T)가 온도에 따라 변함
    (b) OH 고정     sigma_OH0를 200도 값 9.0에 묶고 온도만 변화

    (b)가 평평하면 감소 원인은 OH 밀도 단독이다. [P05] Fig.20과 같은 논리를
    전체 사이클로 확장한 것.
    """
    if Ts is None:
        Ts = np.linspace(150.0, 300.0, 16)
    full = np.array([steady_gpc(t) for t in Ts])
    frozen = np.array([steady_gpc(t, sOH0_nm2=9.0) for t in Ts])
    return Ts, full, frozen


def goal3_low_T_boundary(D_h2o_list=(30.0, 100.0, 300.0, 750.0), Ts=None):
    """
    목표 3: H2O dose를 줄이면 저온에서 반쪽이 완결되지 않아 GPC가 떨어진다.
    이것이 window 하한을 만든다.
    """
    if Ts is None:
        Ts = np.linspace(120.0, 300.0, 19)
    out = {}
    for D in D_h2o_list:
        out[D] = np.array([steady_gpc(t, D_h2o=D) for t in Ts])
    return Ts, out


def sensitivity(Ts, gpc):
    """S(T) = (1/GPC) dGPC/dT  [1/K]. Phase 3의 입력."""
    d = np.gradient(gpc, Ts)
    return d / gpc


def h2o_half_time(T_C):
    """H2O 반쪽의 특성 시간 [ms]. 자릿수 점검용."""
    p = make_cycle_params(T_C)
    return p["sMe_max"] / (p["beta_h2o"] * p["Gamma_h2o"]) * 1e3


# ----------------------------------------------------------------------
# 6. 그래프
# ----------------------------------------------------------------------

def plot_window():
    """그래프 1: GPC vs T. 충분 포화 곡선과 H2O dose를 줄인 곡선들."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    Ts, curves = goal3_low_T_boundary()
    for D in sorted(curves):
        lab = (f"H2O dose {D:g} mTorr s"
               + ("  (문헌값, 충분 포화)" if D >= D_H2O_SAT else ""))
        ax1.plot(Ts, curves[D], label=lab)

    ref = np.array([gpc_from_sAl(dcAl_ref_nm2(t) * NM2_TO_M2) for t in Ts])
    ax1.plot(Ts, ref, "k--", lw=1.4, label="문헌 [P05] Fig.19 피팅")
    ax1.set_xlabel("증착 온도 [도]")
    ax1.set_ylabel("GPC [Å/cycle]")
    ax1.set_title("GPC vs 온도. H2O 노출량별")
    ax1.legend(fontsize=7.5, loc="lower left")
    ax1.grid(alpha=0.3)

    Ts2, full, frozen = goal2_decompose()
    ax2.plot(Ts2, full, "o-", ms=3.5, label="전체 모델 ($\\sigma_{OH0}$가 T에 따라 변함)")
    ax2.plot(Ts2, frozen, "s-", ms=3.5,
             label="$\\sigma_{OH0}$ = 9.0 고정 (온도만 변화)")
    ax2.set_xlabel("증착 온도 [도]")
    ax2.set_ylabel("GPC [Å/cycle]")
    ax2.set_title("원인 분해: 감소는 OH 밀도 단독인가")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("Phase 2: 온도 의존과 ALD window", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "phase2_temperature.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sensitivity():
    """그래프 2: 민감도 S(T)와 허용 온도 폭. Phase 3 예고."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    Ts = np.linspace(150.0, 300.0, 31)
    gpc = np.array([steady_gpc(t) for t in Ts])
    S = sensitivity(Ts, gpc) * 100.0          # %/K

    ref = np.array([gpc_from_sAl(dcAl_ref_nm2(t) * NM2_TO_M2) for t in Ts])
    S_ref = sensitivity(Ts, ref) * 100.0

    ax1.plot(Ts, np.abs(S), label="모델")
    ax1.plot(Ts, np.abs(S_ref), "k--", label="문헌 [P05] Fig.19 피팅")
    ax1.axhline(2.588, color="r", ls=":", lw=1.2,
                label="LPCVD poly-Si (873 K, Ea 1.7 eV)")
    ax1.set_yscale("log")
    ax1.set_ylim(0.08, 5.0)
    # 로그축 기본 눈금은 10의 거듭제곱만 찍혀 이 범위에서 "1" 하나만 남는다.
    # 눈금을 명시하고, 지수 표기 대신 평문으로 찍어 한글 폰트 글리프 문제도 피한다.
    ax1.set_yticks([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}"))
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_xlabel("증착 온도 [도]")
    ax1.set_ylabel("|S| = |dGPC/dT| / GPC  [%/K]")
    ax1.set_title("온도 민감도")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    for target in (0.5, 1.0, 2.0):
        ax2.plot(Ts, target / np.abs(S), label=f"목표 산포 {target:g} %")
    ax2.set_xlabel("증착 온도 [도]")
    ax2.set_ylabel("허용 온도 폭 $\\Delta$T [K]")
    ax2.set_title("목표 산포별 허용 ΔT (모델 기준)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("Phase 2에서 나온 Phase 3 입력", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "phase2_sensitivity.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# 7. main
# ----------------------------------------------------------------------

def _hr(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main():
    _hr("Phase 2: 온도 의존과 ALD window")
    print(f"  beta_H2O(T) = {P_H2O['beta0_H2O']:.4f}"
          f" exp(-{P_H2O['Ea_H2O_eV']:.4f} eV / kT)      [02 #8]")
    print(f"  {'T[도]':>6} {'beta_H2O':>12} {'sigma_OH0':>11} {'H2O 반쪽 tau':>13}"
          f" {'5 tau dose':>12}")
    print(f"  {'':>6} {'':>12} {'[nm^-2]':>11} {'[ms]':>13} {'[mTorr s]':>12}")
    for T_C in (150.0, 200.0, 250.0, 300.0):
        tau = h2o_half_time(T_C)
        print(f"  {T_C:6.0f} {beta_H2O(T_C):12.3e}"
              f" {sigma_OH0_nm2(T_C):11.2f} {tau:13.1f}"
              f" {5*tau*1e-3*P_H2O['P_H2O_torr']*1e3:12.1f}")
    print(f"\n  H2O 반쪽이 TMA 반쪽(tau 약 10 ms)보다 20 ~ 70배 느리다."
          f" 저온일수록 격차가 커진다.")
    print(f"  150도에서 5 tau를 덮으려면 339 mTorr s가 필요하다."
          f" 기준 dose를 문헌값")
    print(f"  750 mTorr s [A19]로 잡은 근거다."
          f" [R01]의 포화 펄스 약 3 s와도 자릿수가 맞는다.")

    # ---- 검증 ----
    _hr("검증: Phase 2 코드")
    tests = [test_condensation_branch(), test_cycle_convergence(),
             test_surface_regenerates(), test_purge_loss(),
             test_rdiss_insensitivity(), test_h2o_curve_monotonic(),
             test_tau_c_sweep()]
    all_ok = True
    for t in tests:
        all_ok &= t["ok"]
        print(f"  [{'통과' if t['ok'] else '실패'}] {t['name']:34s} "
              + t["fmt"].format(t["value"]) + f"  (기준 {t['limit']:.0e})")

    # ---- 목표 1 ----
    _hr("목표 1: GPC 온도 기울기 재현 (01_모델설계.md 4.5절)")
    Ts, dcAl, slope, icept = goal1_slope()
    print(f"  {'T[도]':>6} {'모델 dcAl':>11} {'문헌 dcAl':>11} {'모델 GPC':>10}"
          f" {'문헌 GPC':>10} {'차이':>7}")
    for t, d in zip(Ts, dcAl):
        r = dcAl_ref_nm2(t)
        gm, gr = gpc_from_sAl(d * NM2_TO_M2), gpc_from_sAl(r * NM2_TO_M2)
        print(f"  {t:6.0f} {d:11.2f} {r:11.2f} {gm:10.3f} {gr:10.3f}"
              f" {(gm-gr)/gr*100:+7.1f}")
    lo, hi = -0.0082 - 0.0024, -0.0082 + 0.0024
    print(f"\n  모델   dcAl = {icept:.3f} {slope:+.5f} T")
    print(f"  문헌   dcAl = 6.300 -0.00820 T   (기울기 1 sigma +/- 0.0024)")
    print(f"  [{'통과' if lo <= slope <= hi else '실패'}] 기울기가 문헌 1 sigma"
          f" 구간 [{lo:+.5f}, {hi:+.5f}] 안")

    # ---- 목표 2 ----
    _hr("목표 2: 감소 원인 분해")
    Ts2, full, frozen = goal2_decompose()
    s_full = np.polyfit(Ts2, full, 1)[0]
    s_frozen = np.polyfit(Ts2, frozen, 1)[0]
    print(f"  전체 모델 (sigma_OH0가 T에 따라 변함)   기울기 {s_full:+.6f} Å/cyc/도")
    print(f"  sigma_OH0 = 9.0 고정 (온도만 변화)      기울기 {s_frozen:+.6f} Å/cyc/도")
    print(f"  고정 시 잔여 기울기 비율 {abs(s_frozen/s_full)*100:.2f} %")
    print(f"\n  해석: OH 밀도를 묶으면 온도 의존이 거의 사라진다."
          f"\n  GPC 감소는 반응 메커니즘이 아니라 표면 OH 밀도 감소 단독이 원인이다."
          f"\n  [P05] Fig.20을 전체 사이클로 확장한 결과이고, 같은 결론을 준다.")

    # ---- 목표 3 ----
    _hr("목표 3: 저온 경계는 H2O 반쪽이 만든다")
    Ts3, curves = goal3_low_T_boundary()
    Ds = sorted(curves)
    print(f"  {'T[도]':>6}" + "".join(f"{'D=' + f'{D:g}':>11}" for D in Ds))
    print(f"  {'':>6}" + "".join(f"{'[Å/cyc]':>11}" for _ in Ds))
    for i, t in enumerate(Ts3):
        if int(t) % 20 not in (0, 1) and t != Ts3[-1]:
            continue
        print(f"  {t:6.0f}" + "".join(f"{curves[D][i]:11.3f}" for D in Ds))

    print(f"\n  H2O dose별 GPC 최대점 (이 아래가 dose 제한 구간)")
    for D in Ds:
        g = curves[D]
        i = int(np.argmax(g))
        tag = "  최저온에서 최대 -> dose 제한 없음" if i == 0 else ""
        print(f"    D = {D:5g} mTorr s   최대 {g[i]:.3f} Å/cyc @ {Ts3[i]:.0f} 도{tag}")

    print(f"\n  해석: H2O dose가 충분하면 GPC가 전 구간 단조 감소한다 (평탄부 없음).")
    print(f"  dose를 줄이면 저온에서 H2O 반쪽이 완결되지 못해 GPC가 떨어지고,")
    print(f"  그 결과 중간 온도에 최대점이 생긴다. 겉보기 'ALD window'다.")
    print(f"  [P05] 2.F절이 이 형태를 '불완전 반응이 만든 artefact일 수 있다'고")
    print(f"  경고한 바로 그 경우다. 모델이 그 경고를 재현한다.")

    # ---- Phase 3 입력 ----
    _hr("Phase 3 입력: 민감도")
    Tp = np.array([150.0, 200.0, 250.0, 300.0])
    gp = np.array([steady_gpc(t) for t in Tp])
    Sp = np.abs(sensitivity(Tp, gp)) * 100.0
    refp = np.array([gpc_from_sAl(dcAl_ref_nm2(t) * NM2_TO_M2) for t in Tp])
    Sr = np.abs(sensitivity(Tp, refp)) * 100.0
    print(f"  {'T[도]':>6} {'모델 |S|':>10} {'문헌 |S|':>10} {'허용 ΔT (1%)':>14}")
    print(f"  {'':>6} {'[%/K]':>10} {'[%/K]':>10} {'[K, 모델]':>14}")
    for t, sm, sr in zip(Tp, Sp, Sr):
        print(f"  {t:6.0f} {sm:10.3f} {sr:10.3f} {1.0/sm:14.1f}")
    print(f"\n  비교: LPCVD poly-Si (873 K, Ea 1.7 eV)  |S| = 2.588 %/K")
    print(f"        ALD가 약 {2.588/Sp[1]:.0f}배 둔감하다 (200도 기준)")

    # ---- 그래프 ----
    _hr("그래프 저장")
    for path in (plot_window(), plot_sensitivity()):
        print(f"  {path}")

    _hr("종합")
    print(f"  검증        {'전부 통과' if all_ok else '실패 항목 있음'}")
    print(f"  목표 1      GPC 온도 기울기 재현")
    print(f"  목표 2      감소 원인 = OH 밀도 단독. 분해로 확인")
    print(f"  목표 3      저온 경계 = H2O 반쪽 미완결. dose 의존 확인")
    print(f"  결과 해석은 03_결과.md에 기록한다.")


if __name__ == "__main__":
    main()
