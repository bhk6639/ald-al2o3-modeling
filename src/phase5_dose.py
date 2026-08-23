"""
Phase 5: 전구체 공급 변동에 따른 GPC 산포와 필요 포화 마진.

고체 전구체(MoO2Cl2 등)는 증기압이 액체보다 불안정해서 같은 노출 시간을 줘도
사이클마다 실제 dose가 달라진다. 텅스텐이 몰리브덴으로 대체되는 흐름에서
실제 관리 항목이 되는 문제다(CLAUDE.md 산업 맥락).

이 Phase의 장점은 정확한 동역학 파라미터가 필요 없다는 것이다. 포화 곡선의
형태만 있으면 성립한다. Mo의 동역학 값이 문헌에 없어도 논지가 선다.

답하는 것 다섯.

    1. 실제 포화 곡선과 지수형 가정이 얼마나 다른가
       (01_모델설계.md 6-2절의 예상 표를 실제 곡선으로 다시 계산)
    2. dose 민감도 S_D(D)와 포화 마진의 관계
    3. TMA 반쪽과 H2O 반쪽 중 어느 쪽이 dose 변동에 취약한가
    4. Phase 3이 넘긴 두께 예산에서 필요한 포화 마진은 얼마인가
    5. 마진을 키울 때의 처리량 상충

실행: python src/phase5_dose.py
"""

import os
import math
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
import matplotlib.pyplot as plt

from phase1_saturation import FIG_DIR, TORR
from phase2_temperature import (
    steady_gpc, make_cycle_params, D_TMA_SAT, D_H2O_SAT, h2o_half_time,
)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["figure.dpi"] = 110

T_REF = 200.0          # 기준 온도 [도]

# Phase 3이 넘긴 값. 규격 +/-2 %, sigma_T = 2 K일 때 dose 변동이 쓸 수 있는
# 두께 1 시그마 예산 [%]. 03_결과.md Phase 3 8절.
BUDGET_FROM_PHASE3 = 0.567


# ----------------------------------------------------------------------
# 1. 포화 곡선 (스플라인 보간)
# ----------------------------------------------------------------------

_CURVES = {}


def _build_curve(which, T_C=T_REF, n=34, verbose=True):
    """
    한쪽 반쪽의 dose를 바꿔가며 정상상태 GPC를 구하고 스플라인으로 만든다.
    다른 반쪽은 충분 포화 조건에 고정한다.

    which = "TMA" 또는 "H2O"
    """
    key = (which, T_C)
    if key in _CURVES:
        return _CURVES[key]

    if which == "TMA":
        lo, hi = 0.05, 60.0
        f = lambda D: steady_gpc(T_C, D_tma=D, D_h2o=D_H2O_SAT)
    else:
        lo, hi = 2.0, 4000.0
        f = lambda D: steady_gpc(T_C, D_tma=D_TMA_SAT, D_h2o=D)

    if verbose:
        print(f"  {which} 포화 곡선 구축 중 ({n}점)...", end="", flush=True)
    D = np.geomspace(lo, hi, n)
    g = np.array([f(d) for d in D])
    # 로그 dose에서 보간해야 넓은 범위에서 매끄럽다
    sp = CubicSpline(np.log(D), g)
    _CURVES[key] = {"D": D, "g": g, "spline": sp, "lo": lo, "hi": hi,
                    "gpc_sat": float(g[-1])}
    if verbose:
        print(" 완료")
    return _CURVES[key]


def gpc_at(which, D, T_C=T_REF):
    """dose D [mTorr s]에서의 정상상태 GPC [Angstrom/cycle]."""
    c = _build_curve(which, T_C)
    return float(c["spline"](math.log(np.clip(D, c["lo"], c["hi"]))))


def D0_of(which, T_C=T_REF):
    """
    특성 노출량 D0. 지수형 정의(63.2 % 도달점)를 실제 곡선에 적용한다.
    """
    c = _build_curve(which, T_C)
    target = (1.0 - math.exp(-1.0)) * c["gpc_sat"]
    lo, hi = c["lo"], c["hi"]
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if gpc_at(which, mid, T_C) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


# ----------------------------------------------------------------------
# 2. 목표 1: 실제 곡선 대 지수형 가정
# ----------------------------------------------------------------------

def exp_model(which, D, T_C=T_REF):
    """지수형 근사 GPC_sat [1 - exp(-D/D0)]."""
    c = _build_curve(which, T_C)
    return c["gpc_sat"] * (1.0 - math.exp(-D / D0_of(which, T_C)))


def compare_exp(which, margins=(0.5, 1, 2, 3, 5, 10), T_C=T_REF):
    """마진별로 실제 곡선과 지수형 근사를 대조."""
    D0 = D0_of(which, T_C)
    c = _build_curve(which, T_C)
    rows = []
    for m in margins:
        D = m * D0
        act = gpc_at(which, D, T_C) / c["gpc_sat"]
        ex = exp_model(which, D, T_C) / c["gpc_sat"]
        rows.append((m, D, act, ex, (act - ex) * 100.0))
    return rows


# ----------------------------------------------------------------------
# 3. 목표 2, 3: dose 민감도와 산포 전파
# ----------------------------------------------------------------------

def S_D(which, D, T_C=T_REF, rel_h=1e-3):
    """S_D = (1/GPC) dGPC/dD  [1/(mTorr s)]. 로그 미분으로 안정하게."""
    h = D * rel_h
    return (gpc_at(which, D + h, T_C) - gpc_at(which, D - h, T_C)) / (2 * h) \
        / gpc_at(which, D, T_C)


def SD_times_D(which, D, T_C=T_REF):
    """무차원 민감도 S_D x D. dose가 x 비율로 흔들릴 때 GPC는 이 값 x x 만큼 변한다."""
    return S_D(which, D, T_C) * D


def spread_uniform(which, D, rel_halfwidth, T_C=T_REF, n=201):
    """
    dose가 D(1 +/- x)에 균등 분포할 때 GPC 산포.
    포화 곡선이 오목해서 -x 쪽 손해가 +x 쪽 이득보다 크다. 비대칭을 그대로 담는다.
    """
    Ds = np.linspace(D * (1 - rel_halfwidth), D * (1 + rel_halfwidth), n)
    g = np.array([gpc_at(which, d, T_C) for d in Ds])
    m = g.mean()
    return {"nu_1sigma": g.std(ddof=0) / m * 100.0,
            "nu_range": (g.max() - g.min()) / (2 * m) * 100.0,
            "bias": (m - gpc_at(which, D, T_C)) / gpc_at(which, D, T_C) * 100.0,
            "linear": abs(SD_times_D(which, D, T_C)) * rel_halfwidth * 100.0}


def spread_lognormal(which, D, rel_sigma, T_C=T_REF, n=801, clip=4.0):
    """
    dose가 로그정규분포일 때 GPC 1 시그마 산포 [%].
    평균 D, 변동계수 rel_sigma.

    정규분포를 쓰지 않는 이유: dose는 양수여야 하는데 rel_sigma가 25 %를 넘으면
    정규분포의 4 시그마 꼬리가 0 이하로 내려가 GPC가 붕괴한다. 필요 마진이
    30 % 변동에서 113으로 튀는 비물리적 결과가 나왔다.
    증기압 변동은 본래 곱셈적이므로 로그정규가 물리에도 맞는다.
    """
    s2 = math.log(1.0 + rel_sigma ** 2)
    s = math.sqrt(s2)
    mu = math.log(D) - 0.5 * s2               # 산술평균이 D가 되도록
    z = np.linspace(-clip, clip, n)
    Ds = np.exp(mu + s * z)
    w = np.exp(-0.5 * z ** 2)
    w /= w.sum()
    g = np.array([gpc_at(which, d, T_C) for d in Ds])
    m = float((w * g).sum())
    var = float((w * (g - m) ** 2).sum())
    return math.sqrt(var) / m * 100.0


def curve_overshoot(which, T_C=T_REF):
    """
    포화 곡선이 단조인가. 최대값이 점근값을 넘으면 그 크기를 보고한다.
    H2O 반쪽에서 실제로 넘는다(아래 해설 참조).
    """
    c = _build_curve(which, T_C)
    i = int(np.argmax(c["g"]))
    return {"monotonic": bool(np.all(np.diff(c["g"]) >= -1e-12)),
            "peak_D": float(c["D"][i]), "peak_g": float(c["g"][i]),
            "asymptote": c["gpc_sat"],
            "excess_pct": (float(c["g"][i]) / c["gpc_sat"] - 1.0) * 100.0}


# ----------------------------------------------------------------------
# 4. 목표 4: 목표 산포에서 필요한 포화 마진 역산
# ----------------------------------------------------------------------

def required_margin(which, target_nu_pct, rel_sigma, T_C=T_REF,
                    m_lo=0.3, m_hi=200.0):
    """
    dose 변동이 rel_sigma일 때 목표 GPC 1 시그마 산포를 만족하는 최소 포화 마진.
    반환은 D / D0. 못 만족하면 None.
    """
    D0 = D0_of(which, T_C)
    if spread_lognormal(which, m_hi * D0, rel_sigma, T_C) > target_nu_pct:
        return None
    lo, hi = m_lo, m_hi
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if spread_lognormal(which, mid * D0, rel_sigma, T_C) > target_nu_pct:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


# ----------------------------------------------------------------------
# 5. 목표 5: 처리량 상충
# ----------------------------------------------------------------------

def cycle_time_s(m_tma, m_h2o, T_C=T_REF, t_purge_s=1.0):
    """
    마진 조합에서의 사이클 시간 [s].
    퍼지 시간은 문헌값을 확보하지 못해 가정치다. 기본 1 s x 2회.
    """
    p = make_cycle_params(T_C)
    D_t = m_tma * D0_of("TMA", T_C)
    D_w = m_h2o * D0_of("H2O", T_C)
    t_t = (D_t * 1e-3) * TORR / p["P_tma"]
    t_w = (D_w * 1e-3) * TORR / p["P_h2o"]
    return t_t + t_w + 2 * t_purge_s


# ----------------------------------------------------------------------
# 6. 그래프
# ----------------------------------------------------------------------

def plot_curves_and_sensitivity():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    for which, c_ in (("TMA", "tab:blue"), ("H2O", "tab:green")):
        c = _build_curve(which)
        D0 = D0_of(which)
        m = np.geomspace(0.1, 30.0, 200)
        ax1.plot(m, [gpc_at(which, x * D0) / c["gpc_sat"] for x in m],
                 color=c_, label=f"{which} 실제 곡선")
        ax1.plot(m, 1 - np.exp(-m), "--", color=c_, lw=1,
                 label=f"{which} 지수형 근사")
        ax2.plot(m, [abs(SD_times_D(which, x * D0)) for x in m],
                 color=c_, label=f"{which} 실제")
        ax2.plot(m, m * np.exp(-m) / (1 - np.exp(-m)), "--", color=c_, lw=1,
                 label=f"{which} 지수형")

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("포화 마진  D / D$_0$")
        ax.grid(alpha=0.3, which="both")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:g}"))

    ax1.set_ylabel("GPC / GPC$_{sat}$")
    ax1.set_title("포화 곡선: 실제 대 지수형 가정")
    ax1.axhline(1.0, color="k", lw=0.6, ls=":")
    ax1.legend(fontsize=8)

    ax2.set_yscale("log")
    ax2.set_ylim(1e-4, 2.0)
    ax2.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}"))
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_ylabel("무차원 dose 민감도  |S$_D$ x D|")
    ax2.set_title("dose 민감도")
    ax2.legend(fontsize=8)

    fig.suptitle("Phase 5: 포화 곡선과 dose 민감도", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "phase5_curve_sensitivity.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_margin_and_throughput():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    sigmas = np.array([0.05, 0.10, 0.20, 0.30])
    targets = np.geomspace(0.05, 3.0, 26)
    for s in sigmas:
        y = [required_margin("TMA", t, s) for t in targets]
        y = [v if v is not None else np.nan for v in y]
        ax1.plot(targets, y, label=f"dose 변동 1σ = {s*100:.0f} %")
    ax1.axvline(BUDGET_FROM_PHASE3, color="k", ls=":", lw=1.2,
                label=f"Phase 3 예산 {BUDGET_FROM_PHASE3:.3f} %")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("목표 GPC 1 시그마 산포 [%]")
    ax1.set_ylabel("필요 포화 마진  D / D$_0$")
    ax1.set_title("필요 포화 마진 (TMA 반쪽)")
    ax1.set_xticks([0.05, 0.1, 0.3, 1.0, 3.0])
    ax1.set_yticks([1, 2, 3, 5, 10, 20, 40])
    for axis in (ax1.xaxis, ax1.yaxis):
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    # 처리량 상충. 두 반쪽을 같은 축에 올려 비용 차이를 보인다.
    # H2O dose가 사이클 시간을 지배하므로 TMA 마진은 사실상 공짜다.
    m = np.geomspace(0.5, 40.0, 60)
    m_t0 = D_TMA_SAT / D0_of("TMA")
    m_w0 = D_H2O_SAT / D0_of("H2O")
    s = 0.20
    ax2.plot([cycle_time_s(x, m_w0) for x in m],
             [spread_lognormal("TMA", x * D0_of("TMA"), s) for x in m],
             color="tab:blue", label="TMA 마진을 키울 때")
    ax2.plot([cycle_time_s(m_t0, x) for x in m],
             [spread_lognormal("H2O", x * D0_of("H2O"), s) for x in m],
             color="tab:green", label="H2O 마진을 키울 때")
    ax2.plot([cycle_time_s(m_t0, m_w0)], [spread_lognormal("TMA", D_TMA_SAT, s)],
             "o", color="k", ms=7, label="현재 운전점 (TMA 기준)")
    ax2.axhline(BUDGET_FROM_PHASE3, color="k", ls=":", lw=1.2,
                label=f"Phase 3 예산 {BUDGET_FROM_PHASE3:.3f} %")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-3, 30.0)
    ax2.set_xticks([2, 3, 5, 10, 20, 40])
    ax2.set_yticks([1e-3, 1e-2, 1e-1, 1.0, 10.0])
    for axis in (ax2.xaxis, ax2.yaxis):
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlabel("사이클 시간 [s]  (퍼지 1 s x 2회 가정)")
    ax2.set_ylabel("GPC 1 시그마 산포 [%]  (dose 변동 1σ = 20 %)")
    ax2.set_title("처리량 상충: 어느 쪽 마진이 비싼가")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.suptitle("Phase 5: 필요 포화 마진과 처리량 상충", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "phase5_margin_throughput.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# 7. main
# ----------------------------------------------------------------------

def _hr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    _hr("Phase 5: 전구체 공급 변동에 따른 GPC 산포")
    for w in ("TMA", "H2O"):
        _build_curve(w)

    print(f"\n  기준 온도 {T_REF:.0f}도")
    print(f"  {'반쪽':>5} {'D0':>10} {'GPC_sat':>10} {'문헌 운전 dose':>15}"
          f" {'현재 마진':>10}")
    print(f"  {'':>5} {'[mTorr s]':>10} {'[Å/cyc]':>10} {'[mTorr s]':>15}"
          f" {'D/D0':>10}")
    for w, Dop in (("TMA", D_TMA_SAT), ("H2O", D_H2O_SAT)):
        D0 = D0_of(w)
        print(f"  {w:>5} {D0:10.2f} {_build_curve(w)['gpc_sat']:10.3f}"
              f" {Dop:15.1f} {Dop/D0:10.1f}")

    # ---- 곡선 단조성 점검 ----
    _hr("곡선 단조성 점검 (Phase 2 8.1절 한계가 드러난 곳)")
    for w in ("TMA", "H2O"):
        o = curve_overshoot(w)
        if o["monotonic"]:
            print(f"  {w:>4}  단조 증가. 이상 없음")
        else:
            print(f"  {w:>4}  단조가 아니다."
                  f"  D = {o['peak_D']:.1f} mTorr s에서 최대 {o['peak_g']:.4f},"
                  f" 점근값 {o['asymptote']:.4f}, 초과 {o['excess_pct']:+.2f} %")
    print("\n  H2O 반쪽에서 중간 dose가 점근값을 넘는다. 원인은 모델 구조다.")
    print("  리간드교환이 메틸을 전부 OH로 바꾸는 동시에 해리가 sigma_OH0를 향해")
    print("  채우므로 OH가 일시적으로 sigma_OH0를 초과한다. 응축이 그것을 되돌리는데,")
    print("  모델은 응축 속도를 물 플럭스에 묶어놨다. 그래서 물을 끊는 순간")
    print("  과잉 OH가 그대로 남아 다음 TMA 반쪽으로 넘어간다.")
    print("  실제 탈수산화는 물이 없어도 진행된다(퍼지 중에도). Phase 2 8.1절에")
    print("  적어둔 한계가 여기서 드러났다. 응축을 Ea_dehyd 기반 Arrhenius로")
    print("  바꾸고 퍼지 단계를 넣으면 해소된다.")
    print(f"\n  Phase 5 결론에 미치는 영향: 운전점(H2O 마진 42.3)은 봉우리")
    print(f"  (마진 약 5.7)에서 한참 떨어진 평탄부라 필요 마진 역산은 영향받지 않는다.")

    # ---- 목표 1 ----
    _hr("목표 1: 실제 포화 곡선 대 지수형 가정")
    print("  01_모델설계.md 6-2절의 예상 표는 순수 지수형 가정이었다.")
    print("  Phase 1에서 RMS 잔차 3.4~4.7 %가 남는 것을 확인했으므로 다시 계산한다.\n")
    for w in ("TMA", "H2O"):
        print(f"  {w} 반쪽")
        print(f"  {'마진':>6} {'dose':>10} {'실제':>10} {'지수형':>10} {'차이':>9}")
        print(f"  {'D/D0':>6} {'[mTorr s]':>10} {'GPC/sat':>10} {'GPC/sat':>10}"
              f" {'[%p]':>9}")
        for m, D, act, ex, diff in compare_exp(w):
            print(f"  {m:6.1f} {D:10.2f} {act:10.4f} {ex:10.4f} {diff:+9.2f}")
        print()

    # ---- 목표 2, 3 ----
    _hr("목표 2, 3: dose 민감도와 두 반쪽 비교")
    print("  무차원 민감도 |S_D x D|. dose가 x 비율 흔들리면 GPC는 이 값 x x 만큼 변한다.\n")
    print(f"  {'마진':>6} {'TMA 실제':>10} {'TMA 지수형':>11} {'H2O 실제':>10}"
          f" {'H2O 지수형':>11}")
    for m in (1, 2, 3, 5, 10):
        ex = m * math.exp(-m) / (1 - math.exp(-m))
        print(f"  {m:6d} {abs(SD_times_D('TMA', m*D0_of('TMA'))):10.4f}"
              f" {ex:11.4f} {abs(SD_times_D('H2O', m*D0_of('H2O'))):10.4f}"
              f" {ex:11.4f}")

    print(f"\n  01_모델설계.md 6-2절 예상 표를 실제 곡선으로 다시 계산"
          f" (dose 변동 +/-20 %, TMA)")
    print(f"  {'마진':>6} {'문서 예상':>11} {'실제 선형':>11} {'실제 비선형':>12}"
          f" {'평균 이동':>10}")
    print(f"  {'D/D0':>6} {'[%]':>11} {'[%]':>11} {'1σ [%]':>12} {'[%]':>10}")
    doc = {1: 11.6, 2: 6.3, 3: 3.1, 5: 0.7}
    for m in (1, 2, 3, 5):
        D = m * D0_of("TMA")
        r = spread_uniform("TMA", D, 0.20)
        print(f"  {m:6d} {doc[m]:11.1f} {r['linear']:11.2f}"
              f" {r['nu_1sigma']:12.2f} {r['bias']:+10.3f}")
    print("\n  문서 예상값은 '선형 x 균등분포 반폭' 정의라 1 시그마와 직접 비교되지")
    print("  않는다. 같은 정의(선형)로 보면 실제 곡선이 지수형보다 둔감하다.")

    # ---- 목표 4 ----
    _hr("목표 4: 필요 포화 마진 역산")
    print(f"  Phase 3이 넘긴 예산: dose 변동이 쓸 수 있는 두께 1 시그마"
          f" {BUDGET_FROM_PHASE3:.3f} %")
    print(f"  (규격 +/-2 %, sigma_T = 2 K 기준. 03_결과.md Phase 3 8절)\n")
    print(f"  {'dose 변동 1σ':>13} {'TMA 필요 마진':>14} {'H2O 필요 마진':>14}")
    print(f"  {'[%]':>13} {'D/D0':>14} {'D/D0':>14}")
    for s in (0.05, 0.10, 0.20, 0.30, 0.50):
        rt = required_margin("TMA", BUDGET_FROM_PHASE3, s)
        rw = required_margin("H2O", BUDGET_FROM_PHASE3, s)
        f = lambda v: f"{v:14.1f}" if v is not None else f"{'불가':>14}"
        print(f"  {s*100:13.0f}{f(rt)}{f(rw)}")

    print(f"\n  현재 운전점과 비교")
    for w, Dop in (("TMA", D_TMA_SAT), ("H2O", D_H2O_SAT)):
        m_now = Dop / D0_of(w)
        for s in (0.10, 0.20, 0.30):
            nu = spread_lognormal(w, Dop, s)
            ok = "여유" if nu < BUDGET_FROM_PHASE3 else "초과"
            print(f"    {w} 마진 {m_now:5.1f},  dose 변동 1σ {s*100:2.0f} %"
                  f"  ->  GPC 1σ {nu:7.4f} %   {ok}")

    # ---- 목표 5 ----
    _hr("목표 5: 마진 비용의 비대칭")
    m_h2o_base = D_H2O_SAT / D0_of("H2O")
    print("  주의: 아래 사이클 시간의 절대값은 실제 장비 값이 아니다.")
    print("  H2O 기준 dose 750 mTorr s는 [A19]의 종횡비 10000 LHAR 구조 값이고,")
    print("  압력 0.1 Torr와 퍼지 1 s x 2회도 가정이다. 비교할 실제 레시피가 없으므로")
    print("  사이클 시간 단축을 주장하지 않는다. 유효한 것은 두 반쪽의 상대 비용이다.")
    print()
    print(f"  퍼지는 1 s x 2회로 가정한다 (문헌값 미확보).")
    print(f"  H2O 마진은 현재값 {m_h2o_base:.1f}로 고정하고 TMA 마진만 바꾼다.\n")
    print(f"  {'TMA 마진':>9} {'TMA dose':>10} {'사이클 시간':>12}"
          f" {'GPC 1σ (변동 20%)':>18} {'판정':>7}")
    print(f"  {'D/D0':>9} {'[mTorr s]':>10} {'[s]':>12} {'[%]':>18} {'':>7}")
    for m in (1, 2, 3, 5, 10, 20):
        D = m * D0_of("TMA")
        nu = spread_lognormal("TMA", D, 0.20)
        print(f"  {m:9d} {D:10.2f} {cycle_time_s(m, m_h2o_base):12.2f}"
              f" {nu:18.4f} {'여유' if nu < BUDGET_FROM_PHASE3 else '초과':>7}")

    print(f"\n  H2O 쪽 상충 (TMA 마진은 현재값 고정)")
    m_tma_base = D_TMA_SAT / D0_of("TMA")
    print(f"  {'H2O 마진':>9} {'H2O dose':>10} {'사이클 시간':>12}"
          f" {'GPC 1σ (변동 20%)':>18} {'판정':>7}")
    for m in (1, 2, 3, 5, 10, 36):
        D = m * D0_of("H2O")
        nu = spread_lognormal("H2O", D, 0.20)
        print(f"  {m:9d} {D:10.2f} {cycle_time_s(m_tma_base, m):12.2f}"
              f" {nu:18.4f} {'여유' if nu < BUDGET_FROM_PHASE3 else '초과':>7}")

    # ---- 그래프 ----
    _hr("그래프 저장")
    for p in (plot_curves_and_sensitivity(), plot_margin_and_throughput()):
        print(f"  {p}")

    _hr("Mo 맥락 (정성)")
    print("  MoO2Cl2는 고체 전구체라 증기압이 액체 TMA보다 불안정하다.")
    print("  위 표에서 dose 변동 1 시그마가 5 %에서 30 %로 커질 때 필요한")
    print("  포화 마진이 어떻게 움직이는지가 그 대가다.")
    print("  Mo의 실제 dose 변동 폭과 포화 곡선은 문헌에 없다(02번 8절).")
    print("  따라서 위 값을 Mo에 그대로 적용하지 않는다. 마진을 확보해야 하는")
    print("  이유와 그 정량적 크기를 TMA 계로 보인 것까지가 이 Phase의 범위다.")

    _hr("종합")
    print("  목표 1   실제 곡선과 지수형 가정 대조 완료")
    print("  목표 2,3 dose 민감도 산출, 두 반쪽 비교")
    print("  목표 4   Phase 3 예산에서 필요 마진 역산")
    print("  목표 5   마진 비용의 비대칭. TMA는 거의 공짜, H2O가 시간을 씀")
    print("           (절대 사이클 시간은 가정값 조합이라 주장하지 않음)")
    print("  결과 해석은 03_결과.md에 기록한다.")


if __name__ == "__main__":
    main()
