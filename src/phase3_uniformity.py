"""
Phase 3: 온도 편차에 따른 두께 산포와 허용 온도 폭.

Phase 2의 GPC(T)를 받아 양산 관리 관점으로 변환한다.

    민감도       S(T) = (1/GPC) dGPC/dT           [1/K]
    두께 산포    d = N x GPC(T)  ->  상대 산포는 GPC 상대 산포와 같다
    허용 폭      목표 산포에서 허용 온도 편차를 역산

이 Phase가 답하는 것 넷.

    1. 선형 근사 (산포 = S x dT)가 어디까지 유효한가
    2. 웨이퍼 내 온도 분포가 두께 분포로 어떻게 전파되는가
    3. 목표 두께 산포에서 필요한 온도 균일도는 얼마인가
    4. 같은 온도 편차에서 ALD와 LPCVD poly-Si의 차이는 얼마인가

온도 편차 시나리오(+/-1, 2, 5 K 등)는 문헌값이 아니라 **가정한 운전 조건**이다.
장비 스펙에서 실제 값이 나오면 그 값으로 표를 다시 채우면 된다.

실행: python src/phase3_uniformity.py
"""

import os
import math
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
import matplotlib.pyplot as plt

from phase1_saturation import FIG_DIR, NM2_TO_M2, dcAl_ref_nm2, gpc_from_sAl
from phase2_temperature import steady_gpc, D_TMA_SAT, D_H2O_SAT

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["figure.dpi"] = 110

KB_EV = 8.617333262e-5

# 비교 대상: 표면반응 율속 LPCVD poly-Si.
# Ea 약 1.7 eV, 통상 운전 873 K. S = Ea / (k_B T^2).
LPCVD = {"Ea_eV": 1.7, "T_K": 873.0}


# ----------------------------------------------------------------------
# 1. GPC 모델 래퍼
# ----------------------------------------------------------------------

# Phase 2 모델의 GPC 한 점을 구하는 데 약 0.9 s가 걸린다(12 사이클 x 2 반쪽 ODE).
# 분포 전파와 스펙 역산은 수천 번 호출하므로 직접 부르면 수십 분이 걸린다.
# GPC(T)는 매끄럽고 거의 선형이므로 격자에서 한 번 계산해 3차 스플라인으로
# 보간한다. 정확도는 test_spline_accuracy로 직접 검사한다.
_T_GRID = np.linspace(120.0, 360.0, 25)
_SPLINE = None


def _build_spline(verbose=True):
    global _SPLINE
    if _SPLINE is None:
        if verbose:
            print(f"  GPC(T) 스플라인 구축 중"
                  f" ({len(_T_GRID)}점, {_T_GRID[0]:.0f}~{_T_GRID[-1]:.0f}도)...",
                  end="", flush=True)
        g = np.array([steady_gpc(t, D_tma=D_TMA_SAT, D_h2o=D_H2O_SAT)
                      for t in _T_GRID])
        _SPLINE = CubicSpline(_T_GRID, g)
        if verbose:
            print(" 완료")
    return _SPLINE


def gpc_model(T_C):
    """Phase 2 전체 사이클 모델의 정상상태 GPC [Angstrom/cycle]. 스플라인 보간."""
    return float(_build_spline()(T_C))


def test_spline_accuracy():
    """
    스플라인이 원 모델을 충실히 재현하나. 격자에 없는 점에서 직접 계산과 대조한다.
    S(T)는 미분값이라 값보다 오차에 민감하므로 S도 함께 본다.
    """
    _build_spline()
    probes = [163.0, 207.0, 241.0, 288.0, 331.0]
    worst_g, worst_S = 0.0, 0.0
    for t in probes:
        direct = steady_gpc(t, D_tma=D_TMA_SAT, D_h2o=D_H2O_SAT)
        worst_g = max(worst_g, abs(gpc_model(t) - direct) / direct)
        # S를 원 모델로 직접 계산해 스플라인 미분과 비교
        h = 2.0
        up = steady_gpc(t + h, D_tma=D_TMA_SAT, D_h2o=D_H2O_SAT)
        dn = steady_gpc(t - h, D_tma=D_TMA_SAT, D_h2o=D_H2O_SAT)
        S_direct = (up - dn) / (2 * h) / direct
        worst_S = max(worst_S, abs(S_of(gpc_model, t) - S_direct) / abs(S_direct))
    return {"gpc": worst_g, "S": worst_S,
            "ok": worst_g < 1e-5 and worst_S < 1e-3}


def gpc_lit(T_C):
    """문헌 [P05] Fig.19 피팅 GPC [Angstrom/cycle]."""
    return gpc_from_sAl(dcAl_ref_nm2(T_C) * NM2_TO_M2)


def gpc_lpcvd(T_C):
    """
    비교용 LPCVD poly-Si. 표면반응 율속이므로 성장속도가 Arrhenius다.
    절대값은 의미 없고 상대 변화만 쓰므로 전인자는 1로 둔다.
    """
    return math.exp(-LPCVD["Ea_eV"] / (KB_EV * (T_C + 273.15)))


def S_of(f, T_C, h=1.0):
    """중심차분으로 S(T) = (1/f) df/dT  [1/K]."""
    return (f(T_C + h) - f(T_C - h)) / (2.0 * h) / f(T_C)


# ----------------------------------------------------------------------
# 2. 목표 1: 선형 근사가 어디까지 유효한가
# ----------------------------------------------------------------------

def linearity_check(T0_C, dT_list, f):
    """
    선형 근사  rel = S(T0) x dT
    실제       rel = [f(T0+dT) - f(T0)] / f(T0)

    반환: (dT, 선형 예측 [%], 실제 [%], 오차 [절대 %포인트])
    """
    S0 = S_of(f, T0_C)
    rows = []
    for dT in dT_list:
        lin = S0 * dT * 100.0
        act = (f(T0_C + dT) - f(T0_C)) / f(T0_C) * 100.0
        rows.append((dT, lin, act, act - lin))
    return rows


# ----------------------------------------------------------------------
# 3. 목표 2: 온도 분포 -> 두께 분포
# ----------------------------------------------------------------------

def propagate_uniform(T0_C, halfwidth_K, f, n=401):
    """
    웨이퍼 내 온도가 [T0-w, T0+w]에 균등 분포한다고 볼 때의 두께 분포.

    반환 dict
        nu_1sigma  1 시그마 상대 산포 [%]
        nu_range   (max - min) / (2 x mean) [%].  반도체에서 흔히 쓰는 표기
    """
    T = np.linspace(T0_C - halfwidth_K, T0_C + halfwidth_K, n)
    g = np.array([f(t) for t in T])
    m = g.mean()
    return {"nu_1sigma": g.std(ddof=0) / m * 100.0,
            "nu_range": (g.max() - g.min()) / (2.0 * m) * 100.0,
            "mean": m}


def propagate_normal(T0_C, sigma_T_K, f, n=2001, clip=4.0):
    """
    온도가 정규분포 N(T0, sigma_T)일 때의 두께 상대 산포 [%] (1 시그마).
    가중 적분으로 구한다 (몬테카를로보다 재현성이 좋다).
    """
    T = np.linspace(T0_C - clip * sigma_T_K, T0_C + clip * sigma_T_K, n)
    w = np.exp(-0.5 * ((T - T0_C) / sigma_T_K) ** 2)
    w /= w.sum()
    g = np.array([f(t) for t in T])
    m = float((w * g).sum())
    var = float((w * (g - m) ** 2).sum())
    return math.sqrt(var) / m * 100.0


# ----------------------------------------------------------------------
# 4. 목표 3: 관리 스펙 역산
# ----------------------------------------------------------------------

def allowed_dT(T0_C, target_nu_pct, f, mode="range"):
    """
    목표 두께 산포에서 허용 온도 편차를 역산한다.

    mode = "range"    +/- dT 균등 분포. nu_range 기준
    mode = "sigma"    정규분포. 1 시그마 기준

    GPC(T)가 거의 선형이라 해석적으로 dT = target / |S| 이지만,
    비선형 잔차를 포함해 수치로 푼다.
    """
    S0 = abs(S_of(f, T0_C))
    guess = target_nu_pct / (S0 * 100.0)          # 선형 초기값

    def nu(x):
        if mode == "range":
            return propagate_uniform(T0_C, x, f)["nu_range"]
        return propagate_normal(T0_C, x, f)

    lo, hi = 1e-4, max(guess * 10.0, 1.0)
    for _ in range(60):                            # 이분법
        mid = 0.5 * (lo + hi)
        if nu(mid) < target_nu_pct:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cp_index(spec_halfwidth_pct, nu_1sigma_pct):
    """
    공정능력지수 Cp = (규격 상한 - 하한) / (6 sigma) = 규격 반폭 / (3 sigma).
    중심이 규격 중앙에 있다고 볼 때의 값이므로 Cpk가 아니라 Cp다.
    """
    return spec_halfwidth_pct / (3.0 * nu_1sigma_pct)


# ----------------------------------------------------------------------
# 5. 그래프
# ----------------------------------------------------------------------

def plot_sensitivity_and_spec():
    """그래프 1: S(T) 비교와 목표 산포별 허용 온도 폭."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    Ts = np.linspace(150.0, 300.0, 16)
    S_m = np.array([abs(S_of(gpc_model, t)) for t in Ts]) * 100.0
    S_l = np.array([abs(S_of(gpc_lit, t)) for t in Ts]) * 100.0
    S_c = np.array([abs(S_of(gpc_lpcvd, t)) for t in Ts]) * 100.0
    S_lp873 = LPCVD["Ea_eV"] / (KB_EV * LPCVD["T_K"] ** 2) * 100.0

    ax1.plot(Ts, S_m, "o-", ms=3.5, label="ALD 모델")
    ax1.plot(Ts, S_l, "k--", label="ALD 문헌 [P05] Fig.19")
    ax1.plot(Ts, S_c, ":", color="tab:red",
             label="LPCVD poly-Si (같은 온도역으로 외삽)")
    ax1.axhline(S_lp873, color="tab:red", lw=1.2,
                label=f"LPCVD poly-Si 실제 운전 873 K = {S_lp873:.2f} %/K")
    ax1.set_yscale("log")
    ax1.set_ylim(0.08, 60.0)
    ax1.set_yticks([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}"))
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_xlabel("증착 온도 [도]")
    ax1.set_ylabel("|S| [%/K]")
    ax1.set_title("온도 민감도 비교")
    ax1.legend(fontsize=7.5)
    ax1.grid(alpha=0.3, which="both")

    targets = np.linspace(0.2, 3.0, 29)
    for T0, style in ((200.0, "-"), (300.0, "--")):
        dT_m = [allowed_dT(T0, t, gpc_model) for t in targets]
        dT_l = [allowed_dT(T0, t, gpc_lit) for t in targets]
        ax2.plot(targets, dT_m, style, color="tab:blue",
                 label=f"모델 @ {T0:.0f}도")
        ax2.plot(targets, dT_l, style, color="k", lw=1,
                 label=f"문헌 @ {T0:.0f}도")
    ax2.set_xlabel("목표 두께 산포 (range 기준) [%]")
    ax2.set_ylabel("허용 온도 편차 $\\pm\\Delta$T [K]")
    ax2.set_title("목표 산포별 허용 온도 폭")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("Phase 3: 온도 민감도와 관리 스펙", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "phase3_sensitivity_spec.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_linearity_and_dist():
    """그래프 2: 선형 근사 유효 범위와 두께 분포 전파."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    dTs = np.linspace(-50.0, 50.0, 41)
    for f, lab, c in ((gpc_model, "ALD 모델", "tab:blue"),
                      (gpc_lpcvd, "LPCVD poly-Si (873 K)", "tab:red")):
        T0 = 200.0 if f is gpc_model else 599.85     # 873 K
        S0 = S_of(f, T0)
        act = np.array([(f(T0 + d) - f(T0)) / f(T0) * 100.0 for d in dTs])
        ax1.plot(dTs, act, "-", color=c, label=f"{lab} 실제")
        ax1.plot(dTs, S0 * dTs * 100.0, "--", color=c, lw=1,
                 label=f"{lab} 선형 근사")
    ax1.set_xlabel("온도 편차 $\\Delta$T [K]")
    ax1.set_ylabel("GPC 상대 변화 [%]")
    ax1.set_title("선형 근사의 유효 범위")
    ax1.legend(fontsize=7.5)
    ax1.grid(alpha=0.3)
    ax1.axhline(0, color="k", lw=0.6)
    ax1.axvline(0, color="k", lw=0.6)

    sig = np.linspace(0.2, 10.0, 40)
    for T0, style in ((200.0, "-"), (300.0, "--")):
        nu = [propagate_normal(T0, s, gpc_model) for s in sig]
        ax2.plot(sig, nu, style, color="tab:blue", label=f"ALD 모델 @ {T0:.0f}도")
    nu_lp = [propagate_normal(599.85, s, gpc_lpcvd) for s in sig]
    ax2.plot(sig, nu_lp, ":", color="tab:red", label="LPCVD poly-Si @ 873 K")
    ax2.set_yscale("log")
    ax2.set_ylim(0.02, 60.0)
    ax2.set_yticks([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}"))
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlabel("온도 1 시그마 $\\sigma_T$ [K]")
    ax2.set_ylabel("두께 1 시그마 산포 [%]")
    ax2.set_title("온도 분포 -> 두께 분포 전파")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.suptitle("Phase 3: 선형성 검증과 분포 전파", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "phase3_linearity_distribution.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# 6. main
# ----------------------------------------------------------------------

def _hr(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def main():
    _hr("Phase 3: 온도 편차에 따른 두께 산포")

    _build_spline()
    acc = test_spline_accuracy()
    print(f"  [{'통과' if acc['ok'] else '실패'}] 스플라인 정확도"
          f"   GPC 최대 상대오차 {acc['gpc']:.2e},"
          f" S 최대 상대오차 {acc['S']:.2e}")

    print("\n  민감도 S(T) = (1/GPC) dGPC/dT")
    print(f"  {'T[도]':>6} {'모델 |S|':>10} {'문헌 |S|':>10} {'GPC 모델':>10}"
          f" {'GPC 문헌':>10}")
    print(f"  {'':>6} {'[%/K]':>10} {'[%/K]':>10} {'[Å/cyc]':>10} {'[Å/cyc]':>10}")
    for T in (150.0, 200.0, 250.0, 300.0):
        print(f"  {T:6.0f} {abs(S_of(gpc_model, T))*100:10.3f}"
              f" {abs(S_of(gpc_lit, T))*100:10.3f}"
              f" {gpc_model(T):10.3f} {gpc_lit(T):10.3f}")

    S_lp = LPCVD["Ea_eV"] / (KB_EV * LPCVD["T_K"] ** 2) * 100.0
    print(f"\n  비교 LPCVD poly-Si (873 K, Ea {LPCVD['Ea_eV']} eV)"
          f"   |S| = {S_lp:.3f} %/K")
    print(f"  ALD가 {S_lp/(abs(S_of(gpc_model,200.0))*100):.0f}배 둔감하다"
          f" (200도 모델 기준),"
          f" {S_lp/(abs(S_of(gpc_lit,200.0))*100):.0f}배 (문헌 기준)")

    # ---- 목표 1: 선형성 ----
    _hr("목표 1: 선형 근사 (산포 = S x dT)가 어디까지 유효한가")
    dTs = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    print("  ALD 모델 @ 200도")
    print(f"  {'dT[K]':>6} {'선형 예측':>11} {'실제':>11} {'오차':>11}"
          f" {'상대오차':>10}")
    print(f"  {'':>6} {'[%]':>11} {'[%]':>11} {'[%p]':>11} {'[%]':>10}")
    for dT, lin, act, err in linearity_check(200.0, dTs, gpc_model):
        print(f"  {dT:6.0f} {lin:11.3f} {act:11.3f} {err:+11.4f}"
              f" {abs(err/act)*100 if act else 0:10.2f}")

    print("\n  LPCVD poly-Si @ 873 K (600도)")
    print(f"  {'dT[K]':>6} {'선형 예측':>11} {'실제':>11} {'오차':>11}"
          f" {'상대오차':>10}")
    for dT, lin, act, err in linearity_check(599.85, dTs, gpc_lpcvd):
        print(f"  {dT:6.0f} {lin:11.3f} {act:11.3f} {err:+11.4f}"
              f" {abs(err/act)*100 if act else 0:10.2f}")

    print("\n  주의: ALD 쪽 오차가 0.00 %로 나오는 것은 부분적으로 자명한 결과다.")
    print("  모델의 sigma_OH0(T)가 선형 실측 관계이고 GPC가 거기에 선형으로")
    print("  붙으므로, GPC(T)가 선형인 것은 입력에서 물려받은 성질이다.")
    print("  독립적인 예측이 아니다.")
    print("\n  그래도 의미가 있는 이유는 문헌 GPC(T)도 선형 피팅이기 때문이다")
    print("  ([P05] Fig.19, 8개 그룹 데이터). 즉 선형성 자체가 실측으로 뒷받침된다.")
    print("  대비되는 것은 LPCVD다. Arrhenius라 dT가 커지면 빠르게 벌어지고")
    print("  부호에 따라 비대칭이다(+50 K에서 선형 예측 대비 46 % 어긋남).")
    print("  ALD의 장점은 민감도가 작다는 것만이 아니라 예측 가능하다는 것이다.")

    # ---- 목표 2: 분포 전파 ----
    _hr("목표 2: 웨이퍼 내 온도 분포 -> 두께 분포")
    print("  (온도 편차 값은 문헌이 아니라 가정한 운전 시나리오다)")
    print(f"\n  균등 분포 +/- dT, ALD 모델 @ 200도")
    print(f"  {'+/-dT[K]':>9} {'두께 1시그마':>13} {'두께 range':>12}")
    print(f"  {'':>9} {'[%]':>13} {'[%]':>12}")
    for w in (1.0, 2.0, 3.0, 5.0, 10.0):
        r = propagate_uniform(200.0, w, gpc_model)
        print(f"  {w:9.1f} {r['nu_1sigma']:13.4f} {r['nu_range']:12.4f}")

    print(f"\n  정규분포 sigma_T, 두께 1 시그마 산포 [%]")
    print(f"  {'sigma_T[K]':>11} {'ALD 200도':>11} {'ALD 300도':>11}"
          f" {'LPCVD 873K':>12}")
    for s in (0.5, 1.0, 2.0, 5.0):
        print(f"  {s:11.1f} {propagate_normal(200.0, s, gpc_model):11.4f}"
              f" {propagate_normal(300.0, s, gpc_model):11.4f}"
              f" {propagate_normal(599.85, s, gpc_lpcvd):12.3f}")

    # ---- 목표 3: 관리 스펙 ----
    _hr("목표 3: 관리 스펙. 목표 산포에서 허용 온도 폭 역산")
    print("  기준: 균등 분포 +/- dT, range 산포 = (max-min)/(2 x 평균)")
    print(f"\n  {'목표 산포':>10} {'모델 200도':>12} {'문헌 200도':>12}"
          f" {'모델 300도':>12} {'문헌 300도':>12}")
    print(f"  {'[%]':>10} {'[+/-K]':>12} {'[+/-K]':>12}"
          f" {'[+/-K]':>12} {'[+/-K]':>12}")
    for tgt in (0.3, 0.5, 1.0, 2.0, 3.0):
        print(f"  {tgt:10.1f}"
              f" {allowed_dT(200.0, tgt, gpc_model):12.1f}"
              f" {allowed_dT(200.0, tgt, gpc_lit):12.1f}"
              f" {allowed_dT(300.0, tgt, gpc_model):12.1f}"
              f" {allowed_dT(300.0, tgt, gpc_lit):12.1f}")

    print("\n  문헌 기준이 항상 더 빡빡하다. 모델 민감도가 문헌보다 22 % 낮기")
    print("  때문이다(Phase 2 8.3절). 관리 스펙은 보수적인 문헌 기준을 쓴다.")

    # ---- 공정능력 ----
    _hr("공정능력지수 Cp (규격 대비)")
    print("  Cp = 규격 반폭 / (3 x 두께 1 시그마).  Cp >= 1.33이 통상 목표")
    print(f"\n  {'규격':>8} {'sigma_T':>9} {'두께 1시그마':>13} {'Cp':>8} {'판정':>8}")
    print(f"  {'[+/-%]':>8} {'[K]':>9} {'[%]':>13} {'':>8} {'':>8}")
    for spec in (1.0, 2.0):
        for sT in (1.0, 2.0, 5.0):
            nu = propagate_normal(200.0, sT, gpc_lit)   # 보수적으로 문헌 기준
            cp = cp_index(spec, nu)
            print(f"  {spec:8.1f} {sT:9.1f} {nu:13.4f} {cp:8.1f}"
                  f" {'통과' if cp >= 1.33 else '미달':>8}")

    # 규격별로 Cp = 1.33을 만족하는 최대 sigma_T를 역산한다 (해석을 계산에서 뽑는다)
    print("\n  Cp = 1.33을 만족하는 최대 온도 1 시그마 (문헌 기준)")
    for spec in (1.0, 2.0):
        # 허용 두께 1 시그마 = 규격 반폭 / (3 x 1.33)
        nu_allow = spec / (3.0 * 1.33)
        s_lo, s_hi = 1e-3, 50.0
        for _ in range(60):
            mid = 0.5 * (s_lo + s_hi)
            if propagate_normal(200.0, mid, gpc_lit) < nu_allow:
                s_lo = mid
            else:
                s_hi = mid
        print(f"    규격 +/-{spec:.0f} %  ->  허용 두께 1시그마 {nu_allow:.3f} %,"
              f"  최대 sigma_T {0.5*(s_lo+s_hi):.2f} K")

    # ---- 예산 배분 ----
    _hr("두께 산포 예산 배분 (Phase 5로 넘기는 입력)")
    print("  두께 규격을 3 시그마로 잡고, 온도와 dose가 독립이라 보고 제곱합으로 나눈다.")
    print(f"\n  {'규격':>7} {'허용 1시그마':>12} {'sigma_T':>8} {'온도 몫':>9}"
          f" {'온도 비중':>10} {'남는 예산':>10}")
    print(f"  {'[+/-%]':>7} {'[%]':>12} {'[K]':>8} {'[%]':>9} {'[%]':>10} {'[%]':>10}")
    budget = {}
    for spec in (1.0, 2.0):
        allow = spec / 3.0
        for sT in (1.0, 2.0, 5.0):
            used = propagate_normal(200.0, sT, gpc_lit)
            frac = used / allow * 100.0
            rem_sq = allow ** 2 - used ** 2
            rem = math.sqrt(rem_sq) if rem_sq > 0 else float("nan")
            budget[(spec, sT)] = rem
            rem_s = f"{rem:10.3f}" if rem == rem else f"{'초과':>10}"
            print(f"  {spec:7.0f} {allow:12.3f} {sT:8.1f} {used:9.3f}"
                  f" {frac:10.1f}{rem_s}")

    print("\n  해석: 온도가 무시할 수 있는 항이 아니다.")
    print("  규격 +/-1 %에서는 sigma_T = 2 K만 되어도 온도 하나가 예산을 다 쓴다.")
    print("  규격 +/-2 %이거나 sigma_T가 1 K 수준이면 여유가 생긴다.")
    print(f"\n  Phase 5로 넘기는 값 (규격 +/-2 %, sigma_T = 2 K 기준)")
    print(f"    dose 변동이 쓸 수 있는 두께 1 시그마 예산"
          f" = {budget[(2.0, 2.0)]:.3f} %")
    print(f"    이 값에서 필요한 포화 마진을 역산한다")

    # ---- 그래프 ----
    _hr("그래프 저장")
    for path in (plot_sensitivity_and_spec(), plot_linearity_and_dist()):
        print(f"  {path}")

    _hr("종합")
    print("  목표 1   선형 근사는 ALD에서 dT 50 K까지 유효. LPCVD는 아님")
    print("           (단 ALD 선형성은 입력 sigma_OH0(T)에서 물려받은 성질)")
    print("  목표 2   온도 분포 -> 두께 분포 전파 완료")
    print("  목표 3   관리 스펙 표 산출. 보수적인 문헌 기준을 채택")
    print("  추가     온도는 무시할 수 있는 항이 아니다."
          " 규격 +/-1 %면 sigma_T <= 1.4 K 필요")
    print("  결과 해석은 03_결과.md에 기록한다.")


if __name__ == "__main__":
    main()
