"""
Crédit Agricole — Swap Pricer
Pricing rigoureux : courbe bootstrap, DF continus, DV01, taux pair, ZC Inflation
"""

import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# STYLE CRÉDIT AGRICOLE
# ============================================================
CA_GREEN = "#006A3C"
CA_GREEN2 = "#00843D"
CA_LIGHT = "#E8F4EE"
CA_GREY = "#4A4A4A"
CA_LGREY = "#F5F5F5"
CA_WHITE = "#FFFFFF"

st.set_page_config(
    page_title="CA Swap Pricer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(f"""
<style>
/* Globals */
html, body, [data-testid="stAppViewContainer"] {{
    background: {CA_WHITE};
    color: {CA_GREY};
    font-family: 'Helvetica Neue', Arial, sans-serif;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {CA_LGREY};
    border-right: 1px solid #ddd;
}}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label {{
    font-size: 0.82rem;
    font-weight: 600;
    color: {CA_GREY};
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

/* Buttons */
.stButton > button {{
    background: {CA_GREEN} !important;
    color: {CA_WHITE} !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em;
    padding: 0.55rem 1.2rem !important;
}}
.stButton > button:hover {{
    background: {CA_GREEN2} !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab"] {{
    font-weight: 600;
    color: {CA_GREY};
}}
.stTabs [aria-selected="true"] {{
    color: {CA_GREEN} !important;
    border-bottom: 3px solid {CA_GREEN} !important;
}}

/* Metrics */
[data-testid="metric-container"] {{
    background: {CA_LGREY};
    border-radius: 6px;
    padding: 1rem;
    border-left: 4px solid {CA_GREEN};
}}
[data-testid="metric-container"] label {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {CA_GREY};
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-size: 1.4rem;
    font-weight: 700;
    color: {CA_GREEN};
}}

/* Divider */
hr {{ border-color: #ddd; }}

/* Dataframe */
.stDataFrame {{ border-radius: 6px; overflow: hidden; }}

/* Section headers */
.ca-section {{
    background: {CA_LIGHT};
    border-left: 4px solid {CA_GREEN};
    padding: 0.5rem 1rem;
    border-radius: 0 4px 4px 0;
    margin: 1rem 0 0.5rem 0;
    font-weight: 700;
    font-size: 0.9rem;
    color: {CA_GREEN};
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

/* Header banner */
.ca-header {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.6rem 1.2rem;
    background: {CA_GREEN};
    border-radius: 6px;
    margin-bottom: 1.5rem;
}}
.ca-header h1 {{
    color: {CA_WHITE};
    margin: 0;
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}}
.ca-header span {{
    color: rgba(255,255,255,0.7);
    font-size: 0.85rem;
}}

/* Result badge */
.result-positive {{ color: {CA_GREEN}; font-weight: 700; }}
.result-negative {{ color: #C0392B; font-weight: 700; }}

/* Info block */
.ca-info {{
    background: {CA_LIGHT};
    border: 1px solid #c3ddd0;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.83rem;
    color: {CA_GREY};
    margin-bottom: 0.8rem;
}}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MOTEUR DE PRICING
# ============================================================

class ZeroCurve:
    """
    Courbe zéro-coupon avec interpolation cubique en taux continus.
    Toute la logique de pricing utilise des taux continus (log-linéaire
    en facteurs d'actualisation), plus fiable que les taux linéaires simples.
    """

    def __init__(self, maturities: list, rates: list):
        self.maturities = np.array(maturities, dtype=float)
        self.rates = np.array(rates, dtype=float)
        # On interpole en taux continus directement
        self._cs = CubicSpline(self.maturities, self.rates, bc_type='natural')

    def rate(self, t: float) -> float:
        """Taux zéro continu à la maturité t (années)."""
        t = max(t, 0.0)
        if t <= self.maturities[0]:
            return float(self.rates[0])
        if t >= self.maturities[-1]:
            return float(self.rates[-1])
        return float(self._cs(t))

    def df(self, t: float) -> float:
        """Facteur d'actualisation : exp(-r*t)."""
        t = max(t, 0.0)
        if t == 0.0:
            return 1.0
        return float(np.exp(-self.rate(t) * t))

    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Taux forward continu entre t1 et t2 (implicite par la courbe).
        f(t1,t2) = [r2*t2 - r1*t1] / (t2-t1)
        """
        if t2 <= t1:
            raise ValueError(f"t2 ({t2:.4f}) doit être > t1 ({t1:.4f})")
        r1 = self.rate(t1)
        r2 = self.rate(t2)
        return (r2 * t2 - r1 * t1) / (t2 - t1)

    # Conversion en taux simple (pour l'affichage des coupons flottants EURIBOR)
    def simple_forward(self, t1: float, t2: float) -> float:
        """Taux forward simple (linéaire) entre t1 et t2."""
        dt = t2 - t1
        if dt <= 0:
            return 0.0
        df1 = self.df(t1)
        df2 = self.df(t2)
        return (df1 / df2 - 1.0) / dt


# ============================================================
# UTILITAIRES DATES
# ============================================================

def day_count_fraction(start: datetime, end: datetime, convention: str) -> float:
    days = (end - start).days
    if days <= 0:
        return 0.0
    if convention == 'ACT/360':
        return days / 360.0
    elif convention == 'ACT/365':
        return days / 365.0
    elif convention == 'ACT/ACT':
        # Méthode ISDA simplifiée
        return days / 365.25
    elif convention == '30/360':
        d1 = min(start.day, 30)
        d2 = min(end.day, 30) if d1 == 30 else end.day
        val = 360*(end.year - start.year) + 30*(end.month - start.month) + (d2 - d1)
        return val / 360.0
    else:
        raise ValueError(f"Convention inconnue : {convention}")


def add_months(dt: datetime, months: int) -> datetime:
    """Ajoute un nombre de mois à une date (end-of-month aware)."""
    m = dt.month + months
    y = dt.year + (m - 1) // 12
    m = ((m - 1) % 12) + 1
    import calendar
    last_day = calendar.monthrange(y, m)[1]
    d = min(dt.day, last_day)
    return dt.replace(year=y, month=m, day=d)


def generate_schedule(start: datetime, end: datetime, freq_months: int) -> list:
    """Génère une liste de dates de début/fin de période."""
    dates = [start]
    cur = start
    while True:
        nxt = add_months(cur, freq_months)
        if nxt >= end:
            dates.append(end)
            break
        dates.append(nxt)
        cur = nxt
    return dates


def t_years(d: datetime, pricing_date: datetime) -> float:
    """Temps en années entre pricing_date et d (peut être négatif)."""
    return (d - pricing_date).days / 365.25


# ============================================================
# PRICING VANILLA SWAP
# ============================================================

def price_vanilla_swap(
    pricing_date: datetime,
    start_date: datetime,
    end_date: datetime,
    fixed_rate: float,
    spread: float,
    freq_months: int,
    day_count: str,
    nominal: float,
    curve: ZeroCurve,
    spot_rate: float = None,   # taux spot flottant (si période en cours)
    is_ois: bool = False,
    bump_bps: float = 0.0      # bump pour DV01 (+1bp)
) -> dict:
    """
    Prix un IRS standard (payeur fixe / receveur flottant du point de vue du client).
    swap_value = PV_flottant - PV_fixe (positif = client gagne)

    Pour l'OIS, on utilise le même moteur de forward mais on indique la source.
    bump_bps : décalage uniforme de la courbe pour le calcul de sensibilité.
    """

    if bump_bps != 0.0:
        bump = bump_bps / 10000.0
        bumped_rates = curve.rates + bump
        curve = ZeroCurve(curve.maturities, bumped_rates)

    schedule = generate_schedule(start_date, end_date, freq_months)

    pv_fixed = 0.0
    pv_float = 0.0
    rows = []

    for i in range(len(schedule) - 1):
        ps = schedule[i]
        pe = schedule[i + 1]

        t_pay = t_years(pe, pricing_date)
        if t_pay < -1/365.25:   # période entièrement passée → on skip
            continue

        dcf = day_count_fraction(ps, pe, day_count)
        df = curve.df(max(t_pay, 0.0))

        # --- Jambe fixe ---
        fixed_cf = fixed_rate * dcf * nominal

        # --- Jambe flottante ---
        t1 = t_years(ps, pricing_date)
        t2 = t_years(pe, pricing_date)

        if t1 < 0 and spot_rate is not None:
            # Période en cours : on utilise le spot
            float_rate = spot_rate + spread
            source = "Spot"
        else:
            t1c = max(t1, 0.0)
            # Taux forward simple entre t1 et t2
            float_rate = curve.simple_forward(t1c, t2) + spread
            source = "OIS Forward" if is_ois else "Forward"

        float_cf = float_rate * dcf * nominal

        pv_fixed += fixed_cf * df
        pv_float += float_cf * df

        rows.append({
            'Début période':    ps.strftime('%Y-%m-%d'),
            'Fin période':      pe.strftime('%Y-%m-%d'),
            'DCF':              round(dcf, 6),
            'DF':               round(df, 6),
            'Taux flottant':    round(float_rate * 100, 6),
            'Source':           source,
            'CF Fixe (€)':      round(fixed_cf, 2),
            'CF Flottant (€)':  round(float_cf, 2),
            'PV Fixe (€)':      round(fixed_cf * df, 2),
            'PV Flottant (€)':  round(float_cf * df, 2),
        })

    swap_value = pv_float - pv_fixed
    annuity = sum(r['DCF'] * r['DF'] for r in rows) * nominal  # PV01 nominal
    par_rate = (pv_float / annuity) if annuity > 0 else 0.0    # taux pair

    return {
        'swap_value': swap_value,
        'pv_fixed':   pv_fixed,
        'pv_float':   pv_float,
        'annuity':    annuity,
        'par_rate':   par_rate,
        'nominal':    nominal,
        'rows':       rows,
    }


def compute_dv01(pricing_date, start_date, end_date, fixed_rate, spread,
                 freq_months, day_count, nominal, curve, spot_rate, is_ois) -> float:
    """DV01 : sensibilité à un mouvement de +1bp de toute la courbe."""
    base = price_vanilla_swap(pricing_date, start_date, end_date, fixed_rate, spread,
                              freq_months, day_count, nominal, curve, spot_rate, is_ois,
                              bump_bps=0.0)
    bump = price_vanilla_swap(pricing_date, start_date, end_date, fixed_rate, spread,
                              freq_months, day_count, nominal, curve, spot_rate, is_ois,
                              bump_bps=1.0)
    return bump['swap_value'] - base['swap_value']


# ============================================================
# PRICING ZERO-COUPON INFLATION SWAP (ZCIS)
# ============================================================

def price_zcis(
    pricing_date: datetime,
    start_date: datetime,
    end_date: datetime,
    fixed_rate: float,          # taux break-even fixe (ex : 2.10%)
    nominal: float,
    nominal_curve: ZeroCurve,
    inflation_curve: ZeroCurve,
    bump_bps: float = 0.0
) -> dict:
    """
    Zero-Coupon Inflation Swap (ZCIS) standard.

    Jambe Fixe (payée à maturité) :
        N × [(1 + K)^T - 1]

    Jambe Inflation (reçue à maturité) :
        N × [CPI(T)/CPI(0) - 1]
        = N × [(1 + r_inf_forward)^T - 1] avec r_inf_forward issu de la courbe ZC inflation

    swap_value = PV_inflation - PV_fixe (client reçoit l'inflation)
    """
    if bump_bps != 0.0:
        bump = bump_bps / 10000.0
        nominal_curve   = ZeroCurve(nominal_curve.maturities,   nominal_curve.rates   + bump)
        inflation_curve = ZeroCurve(inflation_curve.maturities, inflation_curve.rates + bump)

    T  = t_years(end_date,   pricing_date)
    T0 = t_years(start_date, pricing_date)

    if T <= 0:
        return {
            'swap_value':       0.0,
            'pv_fixed':         0.0,
            'pv_inflation':     0.0,
            'nominal':          nominal,
            'T':                T,
            'fixed_rate':       fixed_rate,
            'implied_inflation': 0.0,
            'par_breakeven':    0.0,
        }

    df_T = nominal_curve.df(T)

    # Taux ZC inflation forward entre T0 et T
    r_inf_T0 = inflation_curve.rate(max(T0, 0.0))
    r_inf_T  = inflation_curve.rate(T)

    if T0 <= 0:
        # Swap déjà démarré : le ratio CPI(0) est fixé, on projette CPI(T)
        cpi_growth = np.exp(r_inf_T * T)
    else:
        # Swap futur : on projette le ratio CPI(T)/CPI(T0)
        cpi_growth = np.exp(r_inf_T * T - r_inf_T0 * T0)

    inflation_payoff = nominal * (cpi_growth - 1.0)
    fixed_payoff     = nominal * ((1.0 + fixed_rate) ** T - 1.0)

    pv_inflation = inflation_payoff * df_T
    pv_fixed     = fixed_payoff     * df_T
    swap_value   = pv_inflation - pv_fixed

    # Taux break-even implicite (taux fixe qui annule le swap)
    par_breakeven = (cpi_growth ** (1.0 / T) - 1.0) if T > 0 else 0.0

    return {
        'swap_value':        swap_value,
        'pv_fixed':          pv_fixed,
        'pv_inflation':      pv_inflation,
        'nominal':           nominal,
        'T':                 T,
        'fixed_rate':        fixed_rate,
        'implied_inflation': (cpi_growth ** (1.0 / T) - 1.0) * 100,
        'par_breakeven':     par_breakeven,
        'df_T':              df_T,
        'fixed_payoff':      fixed_payoff,
        'inflation_payoff':  inflation_payoff,
    }


# ============================================================
# COURBES PAR DÉFAUT (Bloomberg / marché indicatif)
# ============================================================

CURVES = {
    'EURIBOR': {
        'maturities':    [0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30],
        'rates':         [0.01963, 0.01955, 0.01951, 0.01982, 0.02047, 0.02132, 0.02227,
                          0.02326, 0.02423, 0.02517, 0.02605, 0.02687, 0.02829, 0.02992,
                          0.03105, 0.03211, 0.03252],
        'is_ois':        False,
        'default_spot':  2.122,
        'default_freq':  6,
        'default_dc':    'ACT/360',
        'label':         'EURIBOR 6M',
    },
    'ESTR': {
        'maturities':    [1/52, 1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 15, 20, 30],
        'rates':         [0.01932, 0.01969, 0.02129, 0.02296, 0.02518, 0.02647, 0.02678,
                          0.02754, 0.02847, 0.02993, 0.03078, 0.03174, 0.03178],
        'is_ois':        True,
        'default_spot':  1.932,
        'default_freq':  12,
        'default_dc':    'ACT/360',
        'label':         'ESTR OIS',
    },
    'SOFR': {
        'maturities':    [1/52, 1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 15, 20, 30],
        'rates':         [0.03537, 0.03576, 0.03633, 0.03702, 0.03862, 0.03974, 0.03994,
                          0.04044, 0.04122, 0.04237, 0.04404, 0.04476, 0.04428],
        'is_ois':        True,
        'default_spot':  3.537,
        'default_freq':  12,
        'default_dc':    'ACT/360',
        'label':         'SOFR OIS',
    },
}

INFLATION_CURVE_DEFAULT = {
    'maturities': [1, 2, 3, 5, 7, 10, 15, 20, 25, 30],
    'rates':      [0.0210, 0.0225, 0.0230, 0.0225, 0.0218, 0.0210, 0.0195, 0.0185, 0.0180, 0.0175],
}


# ============================================================
# GRAPHIQUES
# ============================================================

PLOT_LAYOUT = dict(
    plot_bgcolor  = CA_WHITE,
    paper_bgcolor = CA_WHITE,
    font          = dict(color=CA_GREY, size=11),
    margin        = dict(l=50, r=20, t=40, b=40),
    height        = 340,
    xaxis         = dict(gridcolor='#e8e8e8', linecolor='#ddd'),
    yaxis         = dict(gridcolor='#e8e8e8', linecolor='#ddd'),
    legend        = dict(bgcolor='rgba(0,0,0,0)', bordercolor='#ddd', borderwidth=1),
)


def plot_cashflows(rows):
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_bar(x=df['Fin période'], y=df['PV Fixe (€)'],
                name='Jambe fixe', marker_color='#bcd4c8')
    fig.add_bar(x=df['Fin période'], y=df['PV Flottant (€)'],
                name='Jambe flottante', marker_color=CA_GREEN)
    fig.update_layout(**PLOT_LAYOUT, barmode='group',
                      title='Flux actualisés par période')
    return fig


def plot_fwd_rates(rows):
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_scatter(x=df['Fin période'], y=df['Taux flottant'],
                    mode='lines+markers',
                    line=dict(color=CA_GREEN, width=2),
                    marker=dict(size=5),
                    name='Taux flottant (%)')
    fig.update_layout(**PLOT_LAYOUT, title='Taux flottants forward (%)',
                      yaxis_title='%')
    return fig


def plot_zero_curve(curve: ZeroCurve, name: str):
    ts = np.linspace(curve.maturities[0], curve.maturities[-1], 200)
    rs = [curve.rate(t) * 100 for t in ts]
    fig = go.Figure()
    fig.add_scatter(x=ts, y=rs, mode='lines',
                    line=dict(color=CA_GREEN, width=2), name=name)
    fig.add_scatter(x=curve.maturities, y=curve.rates * 100,
                    mode='markers', marker=dict(color=CA_GREEN, size=7, symbol='circle'),
                    name='Points de courbe')
    fig.update_layout(**PLOT_LAYOUT, title=f'Courbe zéro-coupon — {name}',
                      xaxis_title='Maturité (années)', yaxis_title='Taux (%)')
    return fig


# ============================================================
# COMPOSANTS UI RÉUTILISABLES
# ============================================================

def section(title: str):
    st.markdown(f'<div class="ca-section">{title}</div>', unsafe_allow_html=True)


def ca_header(title: str, subtitle: str = ""):
    st.markdown(f"""
<div class="ca-header">
    <div>
        <h1>{title}</h1>
        {'<span>' + subtitle + '</span>' if subtitle else ''}
    </div>
</div>
""", unsafe_allow_html=True)


def fmt_eur(v): return f"€ {v:,.0f}"
def fmt_pct(v): return f"{v:.4f} %"
def fmt_bps(v): return f"{v*10000:.2f} bp"


def curve_editor_sidebar(key_prefix: str, default: dict) -> ZeroCurve:
    """Affiche un éditeur de courbe dans la sidebar."""
    with st.sidebar.expander("✏️ Modifier la courbe de taux", expanded=False):
        raw = st.text_area(
            "Maturités, Taux (CSV — une ligne par point)",
            value="\n".join(
                f"{m},{r*100:.4f}"
                for m, r in zip(default['maturities'], default['rates'])
            ),
            height=200,
            key=f"{key_prefix}_curve_csv"
        )
        try:
            mats, rates = [], []
            for line in raw.strip().split('\n'):
                m, r = line.split(',')
                mats.append(float(m))
                rates.append(float(r) / 100)
            return ZeroCurve(mats, rates)
        except Exception:
            st.warning("Format invalide — courbe par défaut utilisée.")
            return ZeroCurve(default['maturities'], default['rates'])

    return ZeroCurve(default['maturities'], default['rates'])


# ============================================================
# PAGE VANILLA SWAP
# ============================================================

def page_vanilla(curve_key: str):
    cfg = CURVES[curve_key]
    curve_default = {'maturities': cfg['maturities'], 'rates': cfg['rates']}

    ca_header(
        f"IRS Vanilla — {curve_key}",
        f"Taux de référence : {cfg['label']}"
    )

    # ---- Sidebar paramètres ----
    st.sidebar.header(f"Paramètres — {curve_key}")

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        pricing_date = st.date_input(
            "Date de pricing", value=datetime(2026, 3, 18).date(),
            min_value=datetime(2000, 1, 1).date(),
            max_value=datetime(2099, 12, 31).date(),
            key=f"pd_{curve_key}"
        )
        start_date = st.date_input(
            "Date de début", value=datetime(2025, 5, 19).date(),
            min_value=datetime(2000, 1, 1).date(),
            max_value=datetime(2099, 12, 31).date(),
            key=f"sd_{curve_key}"
        )
    with col_b:
        end_date = st.date_input(
            "Date de fin", value=datetime(2030, 5, 19).date(),
            min_value=datetime(2000, 1, 1).date(),
            max_value=datetime(2099, 12, 31).date(),
            key=f"ed_{curve_key}"
        )
        nominal = st.number_input(
            "Nominal (€M)", value=180.0, step=1.0,
            key=f"nom_{curve_key}"
        ) * 1_000_000

    st.sidebar.divider()
    col_c, col_d = st.sidebar.columns(2)
    with col_c:
        fixed_rate = st.number_input(
            "Taux fixe (%)", value=cfg['default_spot'],
            step=0.0001, format="%.4f",
            key=f"fx_{curve_key}"
        ) / 100
        spread = st.number_input(
            "Spread (bp)", value=0.0, step=0.5,
            key=f"sp_{curve_key}"
        ) / 10000
    with col_d:
        freq = st.selectbox(
            "Fréquence flottante", [1, 3, 6, 12],
            index=[1, 3, 6, 12].index(cfg['default_freq']),
            key=f"freq_{curve_key}"
        )
        day_count = st.selectbox(
            "Day count", ["ACT/360", "ACT/365", "ACT/ACT", "30/360"],
            index=["ACT/360", "ACT/365", "ACT/ACT", "30/360"].index(cfg['default_dc']),
            key=f"dc_{curve_key}"
        )

    st.sidebar.divider()
    if cfg['is_ois']:
        spot_rate = st.sidebar.number_input(
            "Taux spot OIS (%)", value=cfg['default_spot'], step=0.0001, format="%.4f",
            key=f"spot_{curve_key}"
        ) / 100
    else:
        spot_rate = st.sidebar.number_input(
            f"Taux {freq}M spot (%)", value=cfg['default_spot'], step=0.0001, format="%.4f",
            key=f"spot_{curve_key}"
        ) / 100

    # Éditeur de courbe
    curve = curve_editor_sidebar(curve_key, curve_default)
    if curve is None:
        curve = ZeroCurve(curve_default['maturities'], curve_default['rates'])

    st.sidebar.divider()
    calc_btn = st.sidebar.button("Calculer", use_container_width=True, key=f"calc_{curve_key}")

    # Validation dates
    pd_ = datetime.combine(pricing_date, datetime.min.time())
    sd_ = datetime.combine(start_date,   datetime.min.time())
    ed_ = datetime.combine(end_date,     datetime.min.time())

    if sd_ >= ed_:
        st.error("La date de début doit être antérieure à la date de fin.")
        return

    # ---- Calcul ----
    if calc_btn:
        try:
            res = price_vanilla_swap(pd_, sd_, ed_, fixed_rate, spread,
                                     freq, day_count, nominal, curve,
                                     spot_rate, cfg['is_ois'])
            dv01 = compute_dv01(pd_, sd_, ed_, fixed_rate, spread,
                                freq, day_count, nominal, curve,
                                spot_rate, cfg['is_ois'])
            res['dv01'] = dv01
            st.session_state[f'res_{curve_key}'] = res
        except Exception as e:
            st.error(f"Erreur de calcul : {e}")
            return

    # ---- Affichage ----
    if f'res_{curve_key}' in st.session_state:
        r  = st.session_state[f'res_{curve_key}']
        sv = r['swap_value']

        section("Résultats")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Valeur du swap", fmt_eur(sv),
                  delta="▲ Favorable" if sv > 0 else "▼ Défavorable",
                  delta_color="normal" if sv > 0 else "inverse")
        c2.metric("PV Jambe fixe",      fmt_eur(r['pv_fixed']))
        c3.metric("PV Jambe flottante", fmt_eur(r['pv_float']))
        c4.metric("Taux pair",          fmt_pct(r['par_rate'] * 100))
        c5.metric("DV01",               fmt_eur(r['dv01']))

        section("Flux par période")
        df_show = pd.DataFrame(r['rows'])
        for col in ['CF Fixe (€)', 'CF Flottant (€)', 'PV Fixe (€)', 'PV Flottant (€)']:
            df_show[col] = df_show[col].apply(lambda x: f"{x:,.0f}")
        df_show['Taux flottant'] = df_show['Taux flottant'].apply(lambda x: f"{x:.4f}%")
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        section("Visualisations")
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(plot_cashflows(r['rows']), use_container_width=True)
        with g2:
            st.plotly_chart(plot_fwd_rates(r['rows']), use_container_width=True)

        section("Courbe zéro-coupon")
        st.plotly_chart(plot_zero_curve(curve, curve_key), use_container_width=True)

        section("Export")
        raw_df = pd.DataFrame(r['rows'])
        st.download_button(
            "Télécharger CSV", raw_df.to_csv(index=False),
            file_name=f"swap_{curve_key}.csv", mime="text/csv"
        )


# ============================================================
# PAGE INFLATION SWAP
# ============================================================

def page_inflation():
    ca_header("Zero-Coupon Inflation Swap", "Réplication CPI — ZCIS standard")

    st.sidebar.header("Paramètres — Inflation")

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        pricing_date = st.date_input(
            "Date de pricing", value=datetime(2026, 3, 18).date(),
            min_value=datetime(2000, 1, 1).date(),
            max_value=datetime(2099, 12, 31).date(),
            key="pd_infl"
        )
        start_date = st.date_input(
            "Date de début", value=datetime(2026, 3, 20).date(),
            min_value=datetime(2000, 1, 1).date(),
            max_value=datetime(2099, 12, 31).date(),
            key="sd_infl"
        )
    with col_b:
        end_date = st.date_input(
            "Date de fin", value=datetime(2036, 3, 20).date(),
            min_value=datetime(2000, 1, 1).date(),
            max_value=datetime(2099, 12, 31).date(),
            key="ed_infl"
        )
        nominal = st.number_input(
            "Nominal (€M)", value=100.0, step=1.0,
            key="nom_infl"
        ) * 1_000_000

    st.sidebar.divider()
    fixed_rate = st.sidebar.number_input(
        "Taux fixe inflation (%)", value=2.10, step=0.001, format="%.3f",
        key="fx_infl"
    ) / 100

    st.sidebar.markdown(
        '<div class="ca-info">Le taux fixe est le break-even négocié. '
        'Le swap se règle en zero-coupon à maturité.</div>',
        unsafe_allow_html=True
    )

    # Courbes éditables
    with st.sidebar.expander("✏️ Courbe nominale (EURIBOR)", expanded=False):
        cfg_eur = CURVES['EURIBOR']
        raw_nom = st.text_area(
            "Mat, Taux (CSV)",
            value="\n".join(
                f"{m},{r*100:.4f}"
                for m, r in zip(cfg_eur['maturities'], cfg_eur['rates'])
            ),
            height=160, key="nom_curve_infl"
        )
        try:
            mats, rates = [], []
            for line in raw_nom.strip().split('\n'):
                m, r = line.split(',')
                mats.append(float(m))
                rates.append(float(r) / 100)
            nom_curve = ZeroCurve(mats, rates)
        except Exception:
            nom_curve = ZeroCurve(cfg_eur['maturities'], cfg_eur['rates'])

    with st.sidebar.expander("✏️ Courbe ZC Inflation (Bloomberg ZCPN)", expanded=False):
        raw_inf = st.text_area(
            "Mat, Taux (CSV)",
            value="\n".join(
                f"{m},{r*100:.4f}"
                for m, r in zip(INFLATION_CURVE_DEFAULT['maturities'],
                                INFLATION_CURVE_DEFAULT['rates'])
            ),
            height=160, key="inf_curve_infl"
        )
        try:
            mats, rates = [], []
            for line in raw_inf.strip().split('\n'):
                m, r = line.split(',')
                mats.append(float(m))
                rates.append(float(r) / 100)
            inf_curve = ZeroCurve(mats, rates)
        except Exception:
            inf_curve = ZeroCurve(INFLATION_CURVE_DEFAULT['maturities'],
                                  INFLATION_CURVE_DEFAULT['rates'])

    st.sidebar.divider()
    calc_btn = st.sidebar.button("Calculer ZCIS", use_container_width=True, key="calc_infl")

    pd_ = datetime.combine(pricing_date, datetime.min.time())
    sd_ = datetime.combine(start_date,   datetime.min.time())
    ed_ = datetime.combine(end_date,     datetime.min.time())

    if sd_ >= ed_:
        st.error("La date de début doit être antérieure à la date de fin.")
        return

    if calc_btn:
        try:
            res = price_zcis(pd_, sd_, ed_, fixed_rate, nominal, nom_curve, inf_curve)
            # DV01 : bump nominal curve +1bp
            r_up = price_zcis(pd_, sd_, ed_, fixed_rate, nominal, nom_curve, inf_curve, bump_bps=1.0)
            res['dv01'] = r_up['swap_value'] - res['swap_value']
            st.session_state['res_infl'] = res
        except Exception as e:
            st.error(f"Erreur de calcul : {e}")
            return

    if 'res_infl' in st.session_state:
        r  = st.session_state['res_infl']
        sv = r['swap_value']

        section("Résultats ZCIS")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Valeur du swap", fmt_eur(sv),
                  delta="▲ Favorable" if sv > 0 else "▼ Défavorable",
                  delta_color="normal" if sv > 0 else "inverse")
        c2.metric("PV Jambe inflation",    fmt_eur(r['pv_inflation']))
        c3.metric("PV Jambe fixe",         fmt_eur(r['pv_fixed']))
        c4.metric("Break-even implicite",  f"{r['implied_inflation']:.4f} %")
        c5.metric("DV01",                  fmt_eur(r['dv01']))

        section("Détail du règlement")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
<div class="ca-info">
    <b>Maturité (T)</b> : {r['T']:.4f} ans<br>
    <b>Facteur d'actualisation</b> : {r['df_T']:.6f}<br>
    <b>Taux fixe appliqué</b> : {r['fixed_rate']*100:.4f} %<br>
    <b>Break-even implicite courbe</b> : {r['implied_inflation']:.4f} %<br>
    <b>Payoff fixe brut</b> : {fmt_eur(r['fixed_payoff'])}<br>
    <b>Payoff inflation brut</b> : {fmt_eur(r['inflation_payoff'])}
</div>
""", unsafe_allow_html=True)
        with col2:
            # Graphe des deux courbes
            fig = go.Figure()
            ts_nom = np.linspace(nom_curve.maturities[0], nom_curve.maturities[-1], 150)
            fig.add_scatter(
                x=ts_nom, y=[nom_curve.rate(t) * 100 for t in ts_nom],
                mode='lines', name='Nominale',
                line=dict(color=CA_GREEN, width=2)
            )
            ts_inf = np.linspace(inf_curve.maturities[0], inf_curve.maturities[-1], 150)
            fig.add_scatter(
                x=ts_inf, y=[inf_curve.rate(t) * 100 for t in ts_inf],
                mode='lines', name='ZC Inflation',
                line=dict(color='#888', width=2, dash='dash')
            )
            fig.add_hline(
                y=r['fixed_rate'] * 100,
                line=dict(color='#c0392b', dash='dot', width=1.5),
                annotation_text=f"Break-even fixe {r['fixed_rate']*100:.3f}%",
                annotation_position="top left"
            )
            fig.update_layout(**PLOT_LAYOUT, title="Courbes & Break-even",
                              xaxis_title="Maturité (années)", yaxis_title="%")
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MAIN
# ============================================================

st.markdown("""
<div style="display:flex;align-items:center;gap:1rem;padding:0.4rem 0 1.2rem 0;
            border-bottom:2px solid #006A3C;margin-bottom:1.2rem">
    <div style="width:44px;height:44px;background:#006A3C;border-radius:50%;
                display:flex;align-items:center;justify-content:center">
        <span style="color:white;font-size:1.4rem;font-weight:900">CA</span>
    </div>
    <div>
        <div style="font-size:1.2rem;font-weight:800;color:#006A3C;
                    letter-spacing:0.02em">Swap Pricer</div>
        <div style="font-size:0.78rem;color:#888;letter-spacing:0.04em;
                    text-transform:uppercase">Crédit Agricole — Marchés de taux</div>
    </div>
</div>
""", unsafe_allow_html=True)

tab_eur, tab_estr, tab_sofr, tab_infl = st.tabs(
    ["EURIBOR", "ESTR (OIS)", "SOFR (OIS)", "ZC Inflation"]
)

with tab_eur:
    page_vanilla('EURIBOR')
with tab_estr:
    page_vanilla('ESTR')
with tab_sofr:
    page_vanilla('SOFR')
with tab_infl:
    page_inflation()
