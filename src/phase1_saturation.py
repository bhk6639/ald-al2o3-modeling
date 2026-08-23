"""
Phase 1: TMA 반쪽 사이클 포화 곡선.

TMA + H2O ALD Al2O3의 TMA 반쪽 사이클을 mean-field로 적분해서
dose vs GPC 포화 곡선을 구한다.

모델 구조 (Puurunen 2005 [P05] 4절 서술을 그대로 따름)
    흡착        TMA -> 표면.  Al +1, Me +3.   rate = beta_ads * Gamma * theta_free
    리간드교환   Me* + OH* -> CH4(g).  Me -1, OH -1.   rate = k_ex * s_Me * s_OH

    입체 차폐를 만드는 것은 메틸기이므로 theta_free는 s_Me로 계산한다.
    Al과 Me를 따로 추적해야 Me/Al 비가 1.5~3으로 변하는 것을 재현할 수 있다.

    포화 시 물리
        OH 고갈  -> 교환 정지
        s_Me -> 1/a_Me -> theta_free -> 0 -> 흡착 정지
        질량수지  3 s_Al = s_Me + (소모된 OH)

파라미터 출처는 전부 02_파라미터_조사.md. 각 값 옆 [#n]이 그 문서의 항목 번호.
검증 계획은 01_모델설계.md 9절.

단위: 내부 계산은 전부 SI. 면밀도 m^-2, 플럭스 m^-2 s^-1, 시간 s.
      출력할 때만 nm^-2, Angstrom, ms, mTorr s로 바꾼다.

실행: python src/phase1_saturation.py
"""

import os
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0. 출력 설정
# ----------------------------------------------------------------------

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fig")

# 한글 폰트. Malgun Gothic은 Windows 기본 탑재, NanumGothic은 대체용.
for _font in ("Malgun Gothic", "NanumGothic"):
    if any(f.name == _font for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _font
        break
plt.rcParams["axes.unicode_minus"] = False   # 축 눈금의 마이너스 기호 깨짐 방지
# 한글 폰트에는 수식용 글리프(U+2212 등)가 없다. 수식만 DejaVu로 따로 지정한다.
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["figure.dpi"] = 110


# ----------------------------------------------------------------------
# 1. 물리 상수
# ----------------------------------------------------------------------

KB = 1.380649e-23          # J/K
NA = 6.02214076e23         # /mol
TORR = 133.32236842         # Pa per Torr
NM2_TO_M2 = 1e18            # 1 nm^-2 = 1e18 m^-2


# ----------------------------------------------------------------------
# 2. 파라미터 (전부 02_파라미터_조사.md에서 가져옴)
# ----------------------------------------------------------------------

P = {
    # --- 기체 ---
    "M_TMA": 72.09e-3,          # kg/mol, Al(CH3)3

    # --- [#2] TMA 초기 반응확률, Arts 2019 Table I, 기판 약 220도 ---
    "beta_TMA": 1.0e-3,         # 대표값. 문헌 범위 (0.5~2)e-3
    "beta_TMA_lo": 0.5e-3,
    "beta_TMA_hi": 2.0e-3,

    # --- [#4] 차폐 면적 ---
    # 이론 최대 메틸 밀도 7.2 nm^-2 -> a_Me = 1/7.2 = 0.139 nm^2 (물리적 상한)
    # 실측 포화 메틸 밀도 5~6 nm^-2 -> 유효 a_Me = 1/5.5 = 0.18 nm^2 (캘리브레이션)
    # 기본은 유효값. 01_모델설계.md 3.2절 참조.
    "sMe_max_nm2": 5.5,         # nm^-2, 유효 포화 메틸 밀도
    "sMe_max_theory_nm2": 7.2,  # nm^-2, Model III 이론 상한

    # --- [#9] 막밀도. [P05]가 Delta c_Al 환산에 쓴 값과 통일 ---
    "rho_film": 3.5e3,          # kg/m^3 (= 3.5 g/cm^3)
    "M_Al2O3": 101.9613e-3,     # kg/mol

    # --- [#10] 통상 노출 조건. Arts 2019 LHAR3 ---
    "P_TMA_torr": 0.1,          # Torr. 포화 시간 자릿수 점검과 같은 조건
}

# --- [#1] 표면 OH 면밀도. Knozinger 1978 via [P05] Fig.24 ---
# 200도에서 9 nm^-2, 300도에서 7 nm^-2, 그 사이 선형.
def sigma_OH0_nm2(T_C):
    """온도별 정상상태 표면 OH 면밀도 [nm^-2]. 01_모델설계.md 4.2절."""
    return 9.0 - 0.02 * (T_C - 200.0)


# --- [#5] 문헌 GPC. [P05] Fig.19 평판 기판 피팅. 검증 대조용 ---
def dcAl_ref_nm2(T_C):
    """문헌 Delta c_Al [nm^-2]. 모델 입력이 아니라 검증 기준값."""
    return 6.3 - 0.0082 * T_C


def rho_Al():
    """막 단위 부피당 Al 개수밀도 [m^-3]. Al2O3 화학식당 Al 2개."""
    return 2.0 * P["rho_film"] * NA / P["M_Al2O3"]


def hertz_knudsen(P_pa, T_K, M_kg_per_mol):
    """충돌 플럭스 [m^-2 s^-1]. 01_모델설계.md 3.1절."""
    m = M_kg_per_mol / NA
    return P_pa / math.sqrt(2.0 * math.pi * m * KB * T_K)


# ----------------------------------------------------------------------
# 3. 지배식
# ----------------------------------------------------------------------

def rhs(t, y, p):
    """
    상태 y = [s_OH, s_Me, s_Al], 전부 m^-2.

    흡착      rate_ads = beta * Gamma * theta_free      -> Al +1, Me +3
    리간드교환 rate_ex  = k_ex * s_Me * s_OH             -> Me -1, OH -1
    """
    s_OH, s_Me, s_Al = y

    # 입체 차폐. 메틸기가 만든다.
    theta_free = 1.0 - p["a_Me"] * s_Me
    if theta_free < 0.0:
        theta_free = 0.0

    rate_ads = p["beta"] * p["Gamma"] * theta_free
    rate_ex = p["k_ex"] * max(s_Me, 0.0) * max(s_OH, 0.0)

    return [
        -rate_ex,                    # OH는 교환으로만 소모
        3.0 * rate_ads - rate_ex,    # 흡착으로 3개 유입, 교환으로 1개씩 이탈
        rate_ads,                    # Al은 흡착으로만 증가
    ]


def make_params(T_C, beta=None, Da_ex=100.0, a_Me_nm2=None, sOH0_nm2=None,
                P_torr=None):
    """
    적분에 쓸 파라미터 묶음.

    Da_ex : 리간드교환 속도와 흡착 속도의 비 (무차원).
            k_ex의 절대값은 문헌에 없다. [P05]는 "리간드교환이 흡착보다 먼저
            완료된다"고만 서술한다. 따라서 Da_ex >> 1로 두고, 포화값이
            Da_ex에 둔감함을 감도 분석으로 보인다 (test_Da_insensitivity).
    """
    T_K = T_C + 273.15
    P_pa = (P_torr if P_torr is not None else P["P_TMA_torr"]) * TORR
    Gamma = hertz_knudsen(P_pa, T_K, P["M_TMA"])

    sMe_max = (1.0 / a_Me_nm2 if a_Me_nm2 is not None else P["sMe_max_nm2"]) * NM2_TO_M2
    a_Me = 1.0 / sMe_max
    s_OH0 = (sOH0_nm2 if sOH0_nm2 is not None else sigma_OH0_nm2(T_C)) * NM2_TO_M2
    b = beta if beta is not None else P["beta_TMA"]

    # Da_ex = (k_ex * sMe_max * s_OH0) / (beta * Gamma)  로 정의해 무차원화
    k_ex = Da_ex * b * Gamma / (sMe_max * s_OH0) if s_OH0 > 0 else 0.0

    return {
        "T_C": T_C, "T_K": T_K, "P_pa": P_pa, "Gamma": Gamma,
        "beta": b, "a_Me": a_Me, "sMe_max": sMe_max, "s_OH0": s_OH0,
        "k_ex": k_ex, "Da_ex": Da_ex,
    }


def run_half_cycle(p, t_end=None, n_out=4000, rtol=1e-10, atol=1e-6):
    """
    TMA 반쪽 사이클을 t_end까지 적분.

    atol이 1e-6인 이유: 상태변수가 1e18 규모(m^-2)라 절대허용오차는
    사실상 무의미하고 rtol이 지배한다.
    """
    if t_end is None:
        # 흡착만으로 입체 포화에 도달하는 시간의 10배
        tau_s = 1.0 / (3.0 * p["a_Me"] * p["beta"] * p["Gamma"])
        t_end = 10.0 * tau_s

    y0 = [p["s_OH0"], 0.0, 0.0]
    t_eval = np.linspace(0.0, t_end, n_out)

    sol = solve_ivp(rhs, (0.0, t_end), y0, args=(p,), method="LSODA",
                    t_eval=t_eval, rtol=rtol, atol=atol, max_step=t_end / 200.0)
    if not sol.success:
        raise RuntimeError(f"적분 실패: {sol.message}")
    return sol


def gpc_from_sAl(s_Al_m2):
    """Al 면밀도 [m^-2] -> GPC [Angstrom/cycle]."""
    return s_Al_m2 / rho_Al() * 1e10


# ----------------------------------------------------------------------
# 4. 검증 1단계: 코드 자체 (01_모델설계.md 9.1절)
# ----------------------------------------------------------------------

def test_analytic_steric():
    """
    해석해 대조 1: 리간드교환을 끄면(k_ex = 0) 메틸 축적이 지수 포화형이 된다.

        ds_Me/dt = 3 beta Gamma (1 - a_Me s_Me)
        s_Me(t)  = sMe_max (1 - exp(-t/tau)),  tau = 1/(3 a_Me beta Gamma)

    Phase 5가 가정하는 지수형 포화 GPC(D) = GPC_sat[1 - exp(-D/D0)]의 근거이기도 하다.
    """
    p = make_params(T_C=200.0, Da_ex=0.0)
    sol = run_half_cycle(p)

    tau = 1.0 / (3.0 * p["a_Me"] * p["beta"] * p["Gamma"])
    exact = p["sMe_max"] * (1.0 - np.exp(-sol.t / tau))
    num = sol.y[1]

    err = np.max(np.abs(num - exact)) / p["sMe_max"]
    return {"name": "해석해 대조 (입체 포화, k_ex=0)", "value": err,
            "limit": 1e-4, "ok": err < 1e-4, "fmt": "최대 상대오차 {:.2e}"}


def test_analytic_exchange():
    """
    해석해 대조 2: 메틸 밀도를 상수로 고정하면 OH가 단일지수로 감소한다.

        ds_OH/dt = -k_ex s_Me_fixed s_OH
        s_OH(t)  = s_OH0 exp(-t/tau),  tau = 1/(k_ex s_Me_fixed)

    01_모델설계.md 3.2절의 해석해와 같은 형태다.
    """
    p = make_params(T_C=200.0, Da_ex=10.0)
    s_Me_fixed = 0.5 * p["sMe_max"]
    tau = 1.0 / (p["k_ex"] * s_Me_fixed)
    t = np.linspace(0.0, 5.0 * tau, 2000)

    def rhs_fixed(_t, y):
        return [-p["k_ex"] * s_Me_fixed * y[0]]

    sol = solve_ivp(rhs_fixed, (t[0], t[-1]), [p["s_OH0"]], t_eval=t,
                    method="LSODA", rtol=1e-11, atol=1e-6)
    exact = p["s_OH0"] * np.exp(-t / tau)
    err = np.max(np.abs(sol.y[0] - exact)) / p["s_OH0"]
    return {"name": "해석해 대조 (OH 고갈, s_Me 고정)", "value": err,
            "limit": 1e-4, "ok": err < 1e-4, "fmt": "최대 상대오차 {:.2e}"}


def test_convergence():
    """적분 수렴: 허용오차를 100배 조여도 포화값이 0.1% 이내로 안 변해야 한다."""
    p = make_params(T_C=200.0)
    loose = run_half_cycle(p, rtol=1e-8).y[2][-1]
    tight = run_half_cycle(p, rtol=1e-12).y[2][-1]
    rel = abs(tight - loose) / tight
    return {"name": "적분 수렴 (rtol 1e-8 vs 1e-12)", "value": rel,
            "limit": 1e-3, "ok": rel < 1e-3, "fmt": "포화값 상대차 {:.2e}"}


def test_physical_range():
    """물리적 범위: 면밀도 음수 없음, 피복률 1 초과 없음."""
    worst = 0.0
    for T_C in (150.0, 200.0, 250.0, 300.0):
        p = make_params(T_C=T_C)
        sol = run_half_cycle(p)
        s_OH, s_Me, s_Al = sol.y
        worst = max(worst,
                    max(0.0, -s_OH.min()) / p["s_OH0"],
                    max(0.0, -s_Me.min()) / p["s_OH0"],
                    max(0.0, -s_Al.min()) / p["s_OH0"],
                    max(0.0, (s_Me / p["sMe_max"]).max() - 1.0))
    return {"name": "물리적 범위 (음수/1 초과 없음)", "value": worst,
            "limit": 1e-9, "ok": worst < 1e-9, "fmt": "최대 위반량 {:.2e}"}


def test_mass_balance():
    """
    질량수지: 3 s_Al = s_Me + (소모된 OH)가 항상 성립해야 한다.
    [P05] Eq.23과 같은 관계다. ODE 우변의 정합성을 직접 검사한다.
    """
    worst = 0.0
    for T_C in (150.0, 200.0, 250.0, 300.0):
        p = make_params(T_C=T_C)
        sol = run_half_cycle(p)
        s_OH, s_Me, s_Al = sol.y
        resid = 3.0 * s_Al - (s_Me + (p["s_OH0"] - s_OH))
        worst = max(worst, np.max(np.abs(resid)) / p["s_OH0"])
    return {"name": "질량수지 3 s_Al = s_Me + dOH", "value": worst,
            "limit": 1e-8, "ok": worst < 1e-8, "fmt": "최대 잔차 {:.2e}"}


def test_Da_insensitivity():
    """
    k_ex 절대값이 문헌에 없으므로, 포화값이 Da_ex에 둔감한지 확인한다.
    [P05]가 "리간드교환이 먼저 완료된다"고 한 조건(Da_ex >> 1)에서
    포화 Al 밀도가 수렴해야 한다.
    """
    vals = []
    for Da in (10.0, 100.0, 1000.0, 10000.0):
        p = make_params(T_C=200.0, Da_ex=Da)
        vals.append(run_half_cycle(p).y[2][-1])
    spread = (max(vals[1:]) - min(vals[1:])) / np.mean(vals[1:])   # Da >= 100 구간
    return {"name": "포화값의 k_ex 둔감성 (Da >= 100)", "value": spread,
            "limit": 1e-2, "ok": spread < 1e-2, "fmt": "상대 산포 {:.2e}"}


# ----------------------------------------------------------------------
# 5. 검증 2단계: 문헌 대조 (01_모델설계.md 9.2절)
# ----------------------------------------------------------------------

def saturation_at(T_C, **kw):
    """주어진 조건에서 TMA 반쪽 포화값을 [nm^-2] 단위로 반환."""
    p = make_params(T_C=T_C, **kw)
    sol = run_half_cycle(p)
    s_OH, s_Me, s_Al = sol.y[:, -1]
    return {
        "s_OH": s_OH / NM2_TO_M2,
        "s_Me": s_Me / NM2_TO_M2,
        "s_Al": s_Al / NM2_TO_M2,
        "GPC": gpc_from_sAl(s_Al),
        "MeAl": s_Me / s_Al if s_Al > 0 else float("nan"),
        "p": p,
    }


def check_absolute():
    """2단계-A: GPC 절대값과 온도 기울기를 문헌과 대조."""
    rows = []
    for T_C in (150.0, 200.0, 220.0, 250.0, 300.0):
        r = saturation_at(T_C)
        ref = dcAl_ref_nm2(T_C)
        rows.append((T_C, r["s_Al"], ref, r["GPC"], gpc_from_sAl(ref * NM2_TO_M2),
                     r["s_Me"], r["MeAl"]))
    return rows


def check_eq24():
    """
    2단계-B1: 모델이 [P05] Eq.24를 스스로 만들어내나.

        문헌 (모델 입력 아님)   dc_Al = 1.68 + 0.37 c_OH
        모델 예측               dc_Al = sMe_max/3 + (1/3) c_OH

    c_OH를 0~9 nm^-2로 스윕하며 포화 Al 밀도를 회귀한다.
    온도는 200도 고정. c_OH만 독립적으로 바꾼다.
    """
    cOH = np.linspace(0.0, 9.0, 19)
    sAl = np.array([saturation_at(200.0, sOH0_nm2=c)["s_Al"] for c in cOH])
    slope, intercept = np.polyfit(cOH, sAl, 1)
    return cOH, sAl, slope, intercept


def check_fig20():
    """
    2단계-B3: c_OH를 2.0 nm^-2로 고정하고 온도만 바꾼다.
    포화 Al 밀도가 평평해야 한다.

        [P05] Fig.20 실측 기울기 = -0.0003 +/- 0.0005 nm^-2 / 도  (통계적으로 0)

    Gamma가 1/sqrt(T)로 변하므로 포화 '속도'는 변한다. 그러나 포화 '값'은
    입체 한계와 c_OH가 정하므로 변하면 안 된다.
    """
    T = np.linspace(80.0, 300.0, 12)
    sAl = np.array([saturation_at(t, sOH0_nm2=2.0)["s_Al"] for t in T])
    slope = np.polyfit(T, sAl, 1)[0]
    return T, sAl, slope


# ----------------------------------------------------------------------
# 6. 포화 곡선 (Phase 1의 주 산출물)
# ----------------------------------------------------------------------

def saturation_curve(T_C, P_torr=None, beta=None, t_end=None):
    """
    dose vs GPC 곡선.

    한 번 적분해서 시간축을 dose = P * t 로 바꾸면 곡선 전체가 나온다.
    dose 단위는 문헌([A19])과 맞추기 위해 mTorr s 로 출력한다.
    """
    p = make_params(T_C=T_C, P_torr=P_torr, beta=beta)
    sol = run_half_cycle(p, t_end=t_end)
    dose_mtorr_s = (p["P_pa"] / TORR) * 1e3 * sol.t
    gpc = gpc_from_sAl(sol.y[2])
    return dose_mtorr_s, gpc, sol, p


def fit_D0(dose, gpc):
    """
    포화 곡선을 GPC(D) = GPC_sat [1 - exp(-D/D0)] 로 근사해 D0를 뽑는다.
    Phase 5가 이 D0를 쓴다. 곡선이 정확히 지수형이 아니므로 근사임을 명시한다.
    """
    gpc_sat = gpc[-1]
    # 63.2% 도달점이 D0 (지수형 정의)
    idx = np.searchsorted(gpc, (1.0 - math.exp(-1.0)) * gpc_sat)
    idx = min(idx, len(dose) - 1)
    D0 = dose[idx]
    resid = gpc - gpc_sat * (1.0 - np.exp(-dose / D0))
    rms = float(np.sqrt(np.mean(resid ** 2)) / gpc_sat)
    return D0, gpc_sat, rms


# ----------------------------------------------------------------------
# 7. 그래프
# ----------------------------------------------------------------------

def plot_saturation_curves():
    """그래프 1: dose vs GPC 포화 곡선 (온도 몇 개)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    temps = [150.0, 200.0, 250.0, 300.0]
    for T_C in temps:
        # 평탄부를 확실히 보이도록 기본 t_end의 3배까지 적분
        p = make_params(T_C=T_C)
        t_end = 30.0 / (3.0 * p["a_Me"] * p["beta"] * p["Gamma"])
        dose, gpc, _, _ = saturation_curve(T_C, t_end=t_end)
        ax1.plot(dose, gpc, label=f"{T_C:.0f} 도")
        ax2.plot(dose, gpc / gpc[-1], label=f"{T_C:.0f} 도")

    # 통상 ALD 펄스 범위 (평판 기준). 15~100 ms, 0.1 Torr -> 1.5~10 mTorr s
    for ax in (ax1, ax2):
        ax.axvspan(1.5, 10.0, color="0.87", zorder=0,
                   label="통상 TMA 펄스 (15~100 ms @ 0.1 Torr)")
        ax.set_xlabel("TMA 노출량 D [mTorr s]")
        ax.grid(alpha=0.3)

    ax1.set_xlim(0.0, 20.0)
    ax1.set_ylabel("GPC [Å/cycle]")
    ax1.set_title("포화 곡선 (선형축)")

    ax2.set_xscale("log")
    ax2.set_xlim(0.05, 30.0)
    # 로그축 눈금 라벨이 한글 폰트로 렌더링되면 지수 기호가 깨진다. 평문으로 찍는다.
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}"))
    ax2.set_ylabel("GPC / GPC$_{sat}$")
    ax2.set_title("포화 곡선 (로그축, 정규화)")
    ax2.axhline(1.0, color="k", lw=0.6, ls=":")

    ax1.legend(fontsize=8, loc="lower right")
    fig.suptitle("Phase 1: TMA 반쪽 사이클 포화 곡선", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "phase1_saturation_curve.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_validation():
    """그래프 2: 검증 3종 (해석해 대조, Eq.24 재현, Fig.20 재현)."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    # (a) 해석해 대조
    ax = axes[0]
    p = make_params(T_C=200.0, Da_ex=0.0)
    sol = run_half_cycle(p)
    tau = 1.0 / (3.0 * p["a_Me"] * p["beta"] * p["Gamma"])
    ax.plot(sol.t * 1e3, sol.y[1] / NM2_TO_M2, lw=3, alpha=0.45,
            label="수치해 (solve_ivp)")
    ax.plot(sol.t * 1e3, p["sMe_max"] * (1 - np.exp(-sol.t / tau)) / NM2_TO_M2,
            "k--", lw=1.2, label="해석해 exp")
    ax.set_xlabel("시간 [ms]")
    ax.set_ylabel("메틸 면밀도 $\\sigma_{Me}$ [nm$^{-2}$]")
    ax.set_title("(a) 해석해 대조 (k$_{ex}$=0, 200 도)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Eq.24 재현
    ax = axes[1]
    cOH, sAl, slope, intercept = check_eq24()
    ax.plot(cOH, sAl, "o-", ms=4, label=f"모델: {intercept:.2f} + {slope:.3f} c$_{{OH}}$")
    ax.plot(cOH, 1.68 + 0.37 * cOH, "r--",
            label="문헌 [P05] Eq.24: 1.68 + 0.37 c$_{OH}$")
    ax.set_xlabel("표면 OH 면밀도 c$_{OH}$ [nm$^{-2}$]")
    ax.set_ylabel("포화 Al 면밀도 $\\Delta$c$_{Al}$ [nm$^{-2}$]")
    ax.set_title("(b) Eq.24 재현 (모델 입력 아님)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Fig.20 재현
    ax = axes[2]
    T, sAl20, slope20 = check_fig20()
    ax.plot(T, sAl20, "o-", ms=4, label=f"모델 기울기 {slope20:+.5f} nm$^{{-2}}$/도")
    ax.axhline(sAl20.mean(), color="r", ls="--", lw=1,
               label="문헌 [P05] Fig.20: 기울기 0")
    ax.set_ylim(sAl20.mean() - 0.5, sAl20.mean() + 0.5)
    ax.set_xlabel("TMA 반응 온도 [도]")
    ax.set_ylabel("포화 Al 면밀도 [nm$^{-2}$]")
    ax.set_title("(c) 온도 독립성 (c$_{OH}$ = 2.0 고정)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Phase 1 검증: 해석해와 문헌 관계식 재현", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "phase1_validation.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# 8. main
# ----------------------------------------------------------------------

def _hr(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main():
    _hr("Phase 1: TMA 반쪽 사이클 포화")
    p200 = make_params(T_C=200.0)
    print(f"  조건        200 도, P_TMA = {P['P_TMA_torr']} Torr")
    print(f"  Gamma_TMA   {p200['Gamma']:.3e} m^-2 s^-1")
    print(f"  beta_TMA    {p200['beta']:.1e}        [02 #2]")
    print(f"  sigma_OH0   {p200['s_OH0']/NM2_TO_M2:.2f} nm^-2       [02 #1]")
    print(f"  sMe_max     {p200['sMe_max']/NM2_TO_M2:.2f} nm^-2       [02 #4]")
    print(f"  rho_Al      {rho_Al():.3e} m^-3      [02 #9]")

    # ---- 검증 1단계 ----
    _hr("검증 1단계: 코드 자체 (01_모델설계.md 9.1절)")
    tests = [test_analytic_steric(), test_analytic_exchange(), test_convergence(),
             test_physical_range(), test_mass_balance(), test_Da_insensitivity()]
    all_ok = True
    for t in tests:
        mark = "통과" if t["ok"] else "실패"
        all_ok &= t["ok"]
        print(f"  [{mark}] {t['name']:38s} " + t["fmt"].format(t["value"])
              + f"  (기준 {t['limit']:.0e})")

    # ---- 검증 2단계-A ----
    _hr("검증 2단계-A: 절대값 문헌 대조 (01_모델설계.md 9.2절)")
    print(f"  {'T[도]':>6} {'모델 dcAl':>10} {'문헌 dcAl':>10} {'모델 GPC':>10}"
          f" {'문헌 GPC':>10} {'차이':>7} {'sMe':>7} {'Me/Al':>7}")
    print(f"  {'':>6} {'[nm^-2]':>10} {'[nm^-2]':>10} {'[Å/cyc]':>10}"
          f" {'[Å/cyc]':>10} {'[%]':>7} {'[nm^-2]':>7} {'':>7}")
    for T_C, sAl, ref, gpc_m, gpc_r, sMe, meal in check_absolute():
        d = (gpc_m - gpc_r) / gpc_r * 100.0
        print(f"  {T_C:6.0f} {sAl:10.2f} {ref:10.2f} {gpc_m:10.3f}"
              f" {gpc_r:10.3f} {d:+7.1f} {sMe:7.2f} {meal:7.2f}")

    # 온도 기울기. 문헌 피팅은 -0.0082 +/- 0.0024 (1 sigma) [P05] Fig.19
    Ts = np.array([150.0, 200.0, 250.0, 300.0])
    model_sAl = np.array([saturation_at(t)["s_Al"] for t in Ts])
    slope_model = np.polyfit(Ts, model_sAl, 1)[0]
    lo, hi = -0.0082 - 0.0024, -0.0082 + 0.0024
    in_band = lo <= slope_model <= hi
    print(f"\n  GPC 온도 기울기   모델 {slope_model:+.5f} nm^-2/도")
    print(f"                    문헌 -0.00820 +/- 0.00240 nm^-2/도  [P05] Fig.19")
    print(f"                    [{'통과' if in_band else '실패'}] 문헌 1 sigma 구간"
          f" [{lo:+.5f}, {hi:+.5f}] 안")

    # Arts 실측 대조
    gpc220 = saturation_at(220.0)["GPC"]
    print(f"\n  220도 GPC        모델 {gpc220:.3f} Å/cyc")
    print(f"                    문헌 1.087 (Puurunen 피팅) / 1.150 (Arts 실측)")
    print(f"                    두 문헌값 사이 차이가 6 %. 모델은 그 사이에"
          f" {'있음' if 1.087 <= gpc220 <= 1.150 else '없음'}")

    # ---- 검증 2단계-B ----
    _hr("검증 2단계-B: 예측 검증. 모델에 넣지 않은 관계식 재현")
    cOH, sAl_sweep, slope, intercept = check_eq24()
    print(f"  B-1  Eq.24 회귀")
    print(f"       모델   dc_Al = {intercept:.3f} + {slope:.4f} c_OH")
    print(f"       문헌   dc_Al = 1.680 + 0.3700 c_OH        [P05] Eq.24")
    print(f"       절편 차이 {abs(intercept-1.68)/1.68*100:.1f} %,"
          f" 기울기 차이 {abs(slope-0.37)/0.37*100:.1f} %")

    print(f"\n  B-2  Me/Al 비의 c_OH 의존")
    print(f"       {'c_OH':>6} {'모델 Me/Al':>12} {'문헌 Me/Al':>12}")
    for c, ref in ((0.0, 3.0), (3.0, 2.0), (6.0, 1.5)):
        r = saturation_at(200.0, sOH0_nm2=c)
        print(f"       {c:6.1f} {r['MeAl']:12.2f} {ref:12.1f}")
    print(f"       문헌 출처 [P05] Fig.23")

    T20, sAl20, slope20 = check_fig20()
    print(f"\n  B-3  온도 독립성 (c_OH = 2.0 고정, 80~300도)")
    print(f"       모델 기울기 {slope20:+.6f} nm^-2/도")
    print(f"       문헌 기울기 -0.000300 +/- 0.000500 nm^-2/도   [P05] Fig.20")
    b3_ok = abs(slope20) < 0.0008
    print(f"       [{'통과' if b3_ok else '실패'}] 문헌 오차범위 안")

    # B-4: Eq.24 절편이 역으로 sMe_max를 지정한다. 이것도 예측 검증이다.
    print(f"\n  B-4  Eq.24 절편이 함의하는 포화 메틸 밀도")
    print(f"       모델은 절편 = sMe_max / 3 을 예측한다.")
    print(f"       문헌 절편 1.68 -> sMe_max = {1.68*3:.2f} nm^-2")
    print(f"       문헌 실측 포화 메틸 밀도 5 ~ 6 nm^-2                [P05] Fig.22b")
    print(f"       [{'통과' if 5.0 <= 1.68*3 <= 6.0 else '실패'}] 실측 범위 안")
    print(f"\n       sMe_max 감도 (02 #4의 불확실 구간)")
    print(f"       {'sMe_max':>9} {'절편':>8} {'기울기':>8} {'GPC@220':>9}")
    for smax in (5.0, 5.5, 6.0, 7.2):
        c = np.linspace(0.0, 9.0, 10)
        s = np.array([saturation_at(200.0, sOH0_nm2=x, a_Me_nm2=1.0/smax)["s_Al"]
                      for x in c])
        sl, ic = np.polyfit(c, s, 1)
        g = saturation_at(220.0, a_Me_nm2=1.0/smax)["GPC"]
        tag = "  <- 이론 상한" if smax == 7.2 else ""
        print(f"       {smax:9.1f} {ic:8.3f} {sl:8.4f} {g:9.3f}{tag}")

    # ---- 포화 곡선과 D0 ----
    _hr("Phase 1 주 산출물: 포화 곡선과 특성 노출량 D0")
    print(f"  {'T[도]':>6} {'D0':>10} {'GPC_sat':>10} {'지수형 근사':>12}"
          f" {'포화시간':>10}")
    print(f"  {'':>6} {'[mTorr s]':>10} {'[Å/cyc]':>10} {'RMS 잔차':>12}"
          f" {'[ms]':>10}")
    for T_C in (150.0, 200.0, 250.0, 300.0):
        dose, gpc, sol, p = saturation_curve(T_C)
        D0, gpc_sat, rms = fit_D0(dose, gpc)
        t_ms = D0 / ((p["P_pa"] / TORR) * 1e3) * 1e3
        print(f"  {T_C:6.0f} {D0:10.1f} {gpc_sat:10.3f} {rms:12.4f} {t_ms:10.2f}")

    print(f"\n  주의: [A19]의 dose 190 ~ 1200 mTorr s는 종횡비 10000짜리 LHAR")
    print(f"  구조를 채우기 위한 값이다. 확산 제한이 지배하므로 평판 표면의")
    print(f"  포화 dose와 직접 비교하면 안 된다. 이 모델은 평판 표면이다.")
    print(f"  평판 기준 비교 대상은 통상 ALD 펄스 시간이다.")
    print(f"      통상 TMA 펄스   15 ~ 100 ms  (0.1 Torr 기준 1.5 ~ 10 mTorr s)")
    print(f"      [R01] 포화 펄스 약 3 s (압력 미보고)")
    print(f"  모델 D0 약 1.7 mTorr s (포화시간 약 17 ms)는 이 범위 안이다.")

    # ---- 포화 원인 판정 ----
    _hr("포화 원인 판정 (01_모델설계.md 3.2절의 물음)")
    for T_C in (150.0, 200.0, 250.0, 300.0):
        r = saturation_at(T_C)
        left_OH = max(0.0, r["s_OH"]) / (r["p"]["s_OH0"] / NM2_TO_M2) * 100.0
        fill_Me = r["s_Me"] / (r["p"]["sMe_max"] / NM2_TO_M2) * 100.0
        cause = "입체 장애" if fill_Me > 99.0 else "OH 고갈"
        print(f"  {T_C:5.0f} 도   OH 잔량 {left_OH:5.1f} %   "
              f"메틸 충전율 {fill_Me:5.1f} %   ->  {cause}")

    # ---- 그래프 ----
    _hr("그래프 저장")
    for path in (plot_saturation_curves(), plot_validation()):
        print(f"  {path}")

    _hr("종합")
    print(f"  1단계 (코드 자체)  : {'전부 통과' if all_ok else '실패 항목 있음'}")
    print(f"  2단계-A (절대값)   : 위 표 참조")
    print(f"  2단계-B (예측)     : 위 표 참조")
    print(f"  결과 해석은 03_결과.md에 기록한다.")


if __name__ == "__main__":
    main()
