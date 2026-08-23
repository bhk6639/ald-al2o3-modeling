"""
Phase 4: 하지막별 인큐베이션.

Phase 1~3, 5는 전부 정상 성장만 다뤘다. Al2O3 위에 Al2O3를 올리므로 매 사이클
같은 표면이 재생된다는 가정이다(01_모델설계.md 1절).

첫 사이클은 그렇지 않다. TMA가 만나는 것은 하지막이고, 하지막마다 표면 OH
밀도가 다르다. Phase 2에서 GPC를 정하는 유일한 인자가 sigma_OH0임을 보였으므로
(고정 시 온도 의존 0.14 %만 잔존), OH 밀도가 다르면 GPC가 다르다.

모델
    표면을 하지막과 Al2O3의 혼합으로 본다. 증착이 쌓이면서 Al2O3 비율이 늘고
    유효 OH 밀도가 하지막 값에서 Al2O3 값으로 옮겨간다.

        f_film = 1 - exp(-N_Al_누적 / N_mono)          랜덤 피복
        sigma_OH0_eff = f_film x sigma_OH0(Al2O3) + (1 - f_film) x sigma_OH0(하지막)

    N_mono는 Al2O3 한 층에 해당하는 Al 면밀도다. rho_Al x GPC로 계산한다.

    [P05]: SiO2 위에서 c_OH가 "3~5 nm^-2에서 시작해 증착과 함께 알루미늄
    산화물 고유값 약 7 nm^-2로 증가한다". 이 서술을 그대로 구현한 것이다.

다루는 범위
    Type 1 (기판 억제형, 단조 증가)까지만 정량으로 다룬다.
    Type 2 (H 종단 Si)는 섬 성장이라 mean-field로 표현할 수 없다. 왜 못 다루는지를
    설명하는 데만 쓴다. 6절 참조.

실행: python src/phase4_incubation.py
"""

import os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from phase1_saturation import (
    FIG_DIR, TORR, NM2_TO_M2, sigma_OH0_nm2, rho_Al, gpc_from_sAl,
    rhs as rhs_tma,
)
from phase2_temperature import (
    make_cycle_params, rhs_h2o, rhs_purge, _integrate,
    D_TMA_SAT, D_H2O_SAT, T_PURGE_S,
)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["figure.dpi"] = 110

T_REF = 200.0


# ----------------------------------------------------------------------
# 1. 하지막 (전부 문헌값)
# ----------------------------------------------------------------------

UNDERLAYERS = {
    # 완전 수산화 비정질 실리카. Kiselev-Zhuravlev 상수.
    # aOH,AVER = 4.6 (최소제곱) / 4.9 (산술평균) OH nm^-2, 100개 시료.  [Z00]
    "SiO2 (완전 수산화)": {"cOH": 4.6, "src": "[Z00] Kiselev-Zhuravlev 상수"},

    # 화학 산화막. [P05]가 SiO2 하지막의 출발값으로 제시한 범위의 중앙.  [P05] 5.H절
    "SiO2 (화학 산화막)": {"cOH": 4.0, "src": "[P05] 3~5 nm^-2의 중앙"},

    # 560도 열처리한 알루미나. 탈수산화로 OH가 줄어든 표면.  [P05] Fig.20
    "Al2O3 (560도 열처리)": {"cOH": 2.0, "src": "[P05] Fig.20"},

    # 정상 성장 Al2O3. 비교 기준선. 인큐베이션이 없어야 한다.
    "Al2O3 (정상 성장)": {"cOH": None, "src": "sigma_OH0(T)"},
}


def N_mono_m2(T_C=T_REF):
    """
    Al2O3 한 층에 해당하는 Al 면밀도 [m^-2].
    rho_Al x (한 사이클 두께)로 잡는다. 곧 한 사이클에 올라가는 Al 양이다.
    """
    return rho_Al() * 1.13e-10          # GPC 약 1.13 A/cycle (Phase 1)


# ----------------------------------------------------------------------
# 2. 인큐베이션 사이클 구동
# ----------------------------------------------------------------------

def run_incubation(cOH_under_nm2, T_C=T_REF, n_cycles=40,
                   D_tma=D_TMA_SAT, D_h2o=D_H2O_SAT, t_purge_s=T_PURGE_S,
                   reactive=True):
    """
    하지막 위에서 n_cycles 만큼 돌린다.

    cOH_under_nm2 : 하지막의 표면 OH 면밀도 [nm^-2]. None이면 Al2O3 정상값
    reactive      : False면 하지막이 반응하지 않는다(H 종단 Si 같은 경우).
                    이 모델은 그 경우를 제대로 못 다룬다. 6절 참조.

    반환: 사이클별 GPC 배열, 누적 두께 배열, 유효 OH 이력
    """
    p = make_cycle_params(T_C)
    s_OH0_film = p["s_OH0"]
    s_OH0_und = (s_OH0_film if cOH_under_nm2 is None
                 else cOH_under_nm2 * NM2_TO_M2)
    if not reactive:
        s_OH0_und = 0.0

    t_tma = (D_tma * 1e-3) * TORR / p["P_tma"]
    t_h2o = (D_h2o * 1e-3) * TORR / p["P_h2o"]
    Nm = N_mono_m2(T_C)

    s_OH, s_Me = s_OH0_und, 0.0
    cum_Al = 0.0
    gpc, thick, oh_eff = [], [], []

    for _ in range(n_cycles):
        # 현재 피복률에 따른 유효 평형 OH 밀도
        f = 1.0 - math.exp(-cum_Al / Nm)
        s_eff = f * s_OH0_film + (1.0 - f) * s_OH0_und
        q = dict(p)
        q["s_OH0"] = max(s_eff, 1e-6 * s_OH0_film)
        oh_eff.append(s_eff / NM2_TO_M2)

        s_OH, s_Me, s_Al = _integrate(rhs_tma, [s_OH, s_Me, 0.0], t_tma, q)
        cum_Al += s_Al
        gpc.append(gpc_from_sAl(s_Al))
        thick.append(gpc_from_sAl(cum_Al))

        if t_purge_s > 0:
            s_OH, s_Me, _ = _integrate(rhs_purge, [s_OH, s_Me, 0.0], t_purge_s, q)
        s_OH, s_Me, _ = _integrate(rhs_h2o, [s_OH, s_Me, 0.0], t_h2o, q)
        if t_purge_s > 0:
            s_OH, s_Me, _ = _integrate(rhs_purge, [s_OH, s_Me, 0.0], t_purge_s, q)

    return np.array(gpc), np.array(thick), np.array(oh_eff)


def incubation_length(gpc, frac=0.99):
    """정상값의 frac에 처음 도달하는 사이클 번호 (1부터)."""
    g_ss = gpc[-1]
    for i, g in enumerate(gpc, 1):
        if g >= frac * g_ss:
            return i
    return len(gpc)


def thickness_error(gpc, thick, n_target):
    """
    인큐베이션을 무시하고 두께 = GPC_ss x N으로 계산했을 때의 오차 [%].
    n_target 사이클 시점 기준.
    """
    naive = gpc[-1] * n_target
    actual = thick[n_target - 1]
    return (naive - actual) / actual * 100.0


# ----------------------------------------------------------------------
# 3. 그래프
# ----------------------------------------------------------------------

def plot_incubation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    n = 100
    x = np.arange(1, n + 1)
    for name, d in UNDERLAYERS.items():
        g, th, _ = run_incubation(d["cOH"], n_cycles=n)
        ax1.plot(x, g, "-", lw=1.6, label=f"{name}")
        naive = g[-1] * x
        ax2.plot(x, (naive - th) / th * 100.0, "-", lw=1.6, label=name)

    ax1.axhline(run_incubation(None, n_cycles=n)[0][-1], color="k", lw=0.7,
                ls=":", label="정상 성장 GPC")
    ax1.set_xlim(0.5, 25.5)          # 인큐베이션이 10 사이클 안에 끝난다
    ax1.set_xlabel("ALD 사이클 수")
    ax1.set_ylabel("GPC [Å/cycle]")
    ax1.set_title("하지막별 초기 GPC (200도)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.axhline(0, color="k", lw=0.7)
    ax2.set_xlabel("목표 사이클 수")
    ax2.set_ylabel("두께 과대예측 [%]")
    ax2.set_title("인큐베이션 무시 시 두께 오차")
    ax2.set_xscale("log")
    ax2.set_xticks([1, 2, 5, 10, 20, 45, 90])
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}"))
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.suptitle("Phase 4: 하지막별 인큐베이션", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "phase4_incubation.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# 4. main
# ----------------------------------------------------------------------

def _hr(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


def main():
    _hr("Phase 4: 하지막별 인큐베이션")
    print(f"  기준 온도 {T_REF:.0f}도, Al2O3 정상 성장 OH 밀도"
          f" {sigma_OH0_nm2(T_REF):.1f} nm^-2")
    print(f"  한 층 상당 Al 면밀도 N_mono = {N_mono_m2()/NM2_TO_M2:.2f} nm^-2")

    _hr("하지막 (전부 문헌값)")
    print(f"  {'하지막':<22} {'c_OH [nm^-2]':>13}  출처")
    for name, d in UNDERLAYERS.items():
        v = f"{d['cOH']:.1f}" if d["cOH"] is not None else f"{sigma_OH0_nm2(T_REF):.1f}"
        print(f"  {name:<22} {v:>13}  {d['src']}")

    _hr("인큐베이션 길이와 초기 GPC")
    n = 100
    print(f"  {'하지막':<22} {'1사이클':>9} {'정상':>9} {'1사이클/정상':>13}"
          f" {'인큐베이션':>11}")
    print(f"  {'':<22} {'[Å/cyc]':>9} {'[Å/cyc]':>9} {'':>13} {'[사이클]':>11}")
    results = {}
    for name, d in UNDERLAYERS.items():
        g, th, oh = run_incubation(d["cOH"], n_cycles=n)
        results[name] = (g, th, oh)
        print(f"  {name:<22} {g[0]:9.4f} {g[-1]:9.4f} {g[0]/g[-1]:13.1%}"
              f" {incubation_length(g):11d}")

    print(f"\n  인큐베이션은 정상값의 99 %에 처음 도달하는 사이클로 정의했다.")

    _hr("두께 예측 오차: 인큐베이션을 무시하면")
    print("  naive 두께 = 정상 GPC x 사이클 수. 실제보다 얼마나 크게 나오나.")
    print(f"\n  {'하지막':<22}" + "".join(f"{f'{k}회':>9}" for k in (5, 10, 20, 45, 90)))
    for name, (g, th, oh) in results.items():
        row = f"  {name:<22}"
        for k in (5, 10, 20, 45, 90):
            row += f"{thickness_error(g, th, k):8.2f}%"
        print(row)
    print(f"\n  45 사이클은 목표 5 nm, 90 사이클은 10 nm에 해당한다"
          f" (정상 GPC {results['Al2O3 (정상 성장)'][0][-1]:.3f} Å/cyc 기준).")

    _hr("유효 OH 밀도가 옮겨가는 과정")
    print(f"  {'사이클':>6}" + "".join(f"{n_[:18]:>20}" for n_ in list(UNDERLAYERS)[:3]))
    for k in (1, 2, 3, 5, 10, 20):
        row = f"  {k:6d}"
        for name in list(UNDERLAYERS)[:3]:
            row += f"{results[name][2][k-1]:20.2f}"
        print(row)
    print(f"\n  [P05] 5.H절: SiO2 위 c_OH가 3~5 nm^-2에서 시작해 증착과 함께")
    print(f"  알루미늄 산화물 고유값 약 7 nm^-2로 증가한다. 모델이 그 과정을 재현한다.")

    _hr("검증")
    g_ss, _, _ = run_incubation(None, n_cycles=n)
    tests = []
    tests.append(("정상 성장 기준선에 인큐베이션 없음",
                  incubation_length(g_ss), 1, incubation_length(g_ss) == 1))
    g4, _, _ = run_incubation(4.0, n_cycles=n)
    tests.append(("SiO2 초기 GPC가 정상보다 낮음",
                  g4[0] / g4[-1], 1.0, g4[0] < g4[-1]))
    tests.append(("SiO2가 정상값으로 수렴",
                  abs(g4[-1] / g_ss[-1] - 1), 1e-3, abs(g4[-1]/g_ss[-1]-1) < 1e-3))
    mono = all(np.diff(g4) > -1e-9)
    tests.append(("Type 1 단조 증가", 1.0 if mono else 0.0, 1.0, mono))
    for nm_, val, lim, ok in tests:
        print(f"  [{'통과' if ok else '실패'}] {nm_:<34} {val:.4g}")

    _hr("그래프 저장")
    print(f"  {plot_incubation()}")

    _hr("이 모델이 다룰 수 없는 것: Type 2")
    print("  H 종단 Si 위에서는 Si-H가 TMA와도 H2O와도 반응하지 않는다.")
    print("  TMA가 결함 자리에만 붙어 Al2O3 섬이 자라고, 그 섬이 주변 Si를")
    print("  산화시켜 새 반응 자리를 만든다. GPC가 낮게 시작해 오르다 최대를")
    print("  찍고 내려와 정상값에 안착한다([P05] Fig.27, Zhao 등).")
    print("\n  우리 모델은 mean-field라 표면을 평균 피복률 하나로만 본다.")
    print("  섬의 가장자리 길이나 섬이 주변을 산화시키는 2차원 효과를 표현할 수")
    print("  없다. 최대점은 원리적으로 나올 수 없다.")
    print("  Type 2를 다루려면 격자 몬테카를로나 섬 성장 모델이 필요하다.")
    print("  이 Phase는 Type 1까지만 정량으로 주장한다.")

    _hr("종합")
    print("  하지막 OH 밀도가 낮을수록 초기 GPC가 낮고 인큐베이션이 길다")
    print("  얇은 막일수록 두께 예측 오차가 크다")
    print("  결과 해석은 03_결과/phase4_인큐베이션.md에 기록한다")


if __name__ == "__main__":
    main()
