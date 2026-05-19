import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline
import pandas as pd
import plotly.graph_objects as go

class SwapPricer:
    def __init__(self, curve_data, euribor_rates=None):
        self.maturities = np.array(curve_data['maturities'])
        self.rates = np.array(curve_data['rates'])
        self.cs = CubicSpline(self.maturities, self.rates)
        self.euribor_rates = euribor_rates or {}
        
    def get_forward_rate(self, t1, t2):
        if t1 == 0:
            r1 = self.rates[0]
        else:
            r1 = self.cs(t1)
        r2 = self.cs(t2)
        
        forward = ((1 + r2 * t2) / (1 + r1 * t1) - 1) / (t2 - t1)
        return forward
    
    def get_discount_factor(self, t):
        r = self.cs(t)
        return 1 / (1 + r * t)
    
    def day_count_fraction(self, start_date, end_date, convention):
        delta_days = (end_date - start_date).days
        
        if convention == 'ACT/ACT':
            year_start = start_date.year
            year_end = end_date.year
            
            if year_start == year_end:
                days_in_year = 366 if self._is_leap_year(year_start) else 365
                return delta_days / days_in_year
            else:
                return delta_days / 365.25
        
        elif convention == 'ACT/360':
            return delta_days / 360
        
        elif convention == 'ACT/365':
            return delta_days / 365
        
        elif convention == '30/360':
            d1 = min(start_date.day, 30)
            d2 = min(end_date.day, 30) if d1 == 30 else end_date.day
            
            days = 360 * (end_date.year - start_date.year) + \
                   30 * (end_date.month - start_date.month) + \
                   (d2 - d1)
            return days / 360
        
        else:
            raise ValueError(f"Convention inconnue: {convention}")
    
    def _is_leap_year(self, year):
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    
    def generate_payment_dates(self, start_date, end_date, frequency_months):
        dates = []
        current = start_date
        
        while current <= end_date:
            dates.append(current)
            month = current.month + frequency_months
            year = current.year + (month - 1) // 12
            month = ((month - 1) % 12) + 1
            
            try:
                current = current.replace(year=year, month=month)
            except ValueError:
                current = current.replace(year=year, month=month, day=28)
        
        return dates
    
    def _get_euribor_key(self, tenor_months):
        if tenor_months == 3:
            return '3M'
        elif tenor_months == 6:
            return '6M'
        elif tenor_months == 12:
            return '12M'
        else:
            return None
    
    def price_swap(self, pricing_date, start_date, end_date, fixed_rate, 
                   spread, floating_tenor_months, day_count_convention, nominal,
                   historical_rates=None):

        payment_dates = self.generate_payment_dates(start_date, end_date, 
                                                     floating_tenor_months)
        
        if payment_dates[-1] != end_date:
            payment_dates.append(end_date)
        
        pv_fixed = 0
        pv_floating = 0
        details = []
        euribor_key = self._get_euribor_key(floating_tenor_months)
        
        for i in range(len(payment_dates) - 1):
            period_start = payment_dates[i]
            period_end = payment_dates[i + 1]
            
            accrual_fraction = self.day_count_fraction(period_start, period_end, 
                                                       day_count_convention)
            
            time_to_payment = (period_end - pricing_date).days / 365.25
            
            if time_to_payment < 0:
                continue
            
            df = self.get_discount_factor(time_to_payment)
            
            fixed_coupon = fixed_rate * accrual_fraction * nominal
            pv_fixed += fixed_coupon * df
            
            if period_start < pricing_date and historical_rates and period_start in historical_rates:
                floating_rate = historical_rates[period_start] + spread
                rate_source = "Historique"
            elif period_start < pricing_date and euribor_key in self.euribor_rates:
                floating_rate = self.euribor_rates[euribor_key] + spread
                rate_source = "Euribor spot"
            else:
                t1 = max(0, (period_start - pricing_date).days / 365.25)
                t2 = (period_end - pricing_date).days / 365.25
                forward_rate = self.get_forward_rate(t1, t2)
                floating_rate = forward_rate + spread
                rate_source = "Forward"
            
            floating_coupon = floating_rate * accrual_fraction * nominal
            pv_floating += floating_coupon * df
            
            details.append({
                'period_start': period_start,
                'period_end': period_end,
                'accrual_fraction': accrual_fraction,
                'time_to_payment': time_to_payment,
                'discount_factor': df,
                'floating_rate': floating_rate,
                'rate_source': rate_source,
                'fixed_coupon': fixed_coupon,
                'floating_coupon': floating_coupon,
                'pv_fixed': fixed_coupon * df,
                'pv_floating': floating_coupon * df
            })
        
        swap_value = pv_floating - pv_fixed
        
        return {
            'swap_value': swap_value,
            'pv_fixed_leg': pv_fixed,
            'pv_floating_leg': pv_floating,
            'details': details,
            'nominal': nominal
        }

# Configuration Streamlit
st.set_page_config(page_title="Swap Pricer", layout="wide")
st.title("💱 Swap Pricer Interactif")

# Données par défaut
curve_data = {
    'maturities': [0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    'rates': [0.01963708, 0.01955476, 0.01951525, 0.01951413, 0.01981745, 0.02046718, 
              0.02131799, 0.02227016, 0.0232563, 0.02423195, 0.0251688, 0.02604979, 
              0.02686562, 0.02761227, 0.02828921, 0.02889817, 0.02944227, 0.02992544, 
              0.03035198, 0.03072635, 0.03105292, 0.03133594, 0.03157939, 0.03178702, 
              0.03196229, 0.03210839, 0.03222823, 0.03232447, 0.03239953, 0.03245559, 
              0.03249465, 0.0325185, 0.03251876]
}

euribor_rates = {
    '3M': 0.02122,
    '6M': 0.02529,
    '12M': 0.02932
}

historical_rates = {
    datetime(2024, 1, 15): 0.0380,
    datetime(2024, 7, 15): 0.0388,
}

pricer = SwapPricer(curve_data, euribor_rates=euribor_rates)

# Sidebar pour les paramètres
st.sidebar.header("📋 Paramètres du Swap")

pricing_date = st.sidebar.date_input("Date de pricing", value=datetime(2026, 3, 30))
start_date = st.sidebar.date_input("Date de début", value=datetime(2015, 3, 25))
end_date = st.sidebar.date_input("Date de fin", value=datetime(2029, 11, 26))

col1, col2 = st.sidebar.columns(2)
with col1:
    fixed_rate = st.number_input("Taux fixe (%)", value=0.0, step=0.01) / 100
    spread = st.number_input("Spread (%)", value=0.36, step=0.01) / 100

with col2:
    nominal = st.number_input("Nominal (€)", value=180000000, step=1000000)
    floating_tenor = st.selectbox("Tenor flottant", [3, 6, 12], index=1)

day_count = st.sidebar.selectbox("Convention de comptage", 
                                  ["ACT/360", "ACT/365", "ACT/ACT", "30/360"])

# Convertir les dates
pricing_date = datetime.combine(pricing_date, datetime.min.time())
start_date = datetime.combine(start_date, datetime.min.time())
end_date = datetime.combine(end_date, datetime.min.time())

# Calculer le swap
if st.sidebar.button("🔄 Calculer", use_container_width=True):
    result = pricer.price_swap(pricing_date, start_date, end_date, fixed_rate, 
                               spread, floating_tenor, day_count, nominal,
                               historical_rates=historical_rates)
    
    st.session_state.result = result

# Afficher les résultats
if 'result' in st.session_state:
    result = st.session_state.result
    
    # KPIs principaux
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Valeur du Swap", f"€ {result['swap_value']:,.0f}")
    with col2:
        st.metric("PV Jambe Fixe", f"€ {result['pv_fixed_leg']:,.0f}")
    with col3:
        st.metric("PV Jambe Flottante", f"€ {result['pv_floating_leg']:,.0f}")
    with col4:
        st.metric("Nominal", f"€ {result['nominal']:,.0f}")
    
    st.divider()
    
    # Détails des périodes
    st.subheader("📊 Détails des Périodes")
    
    df_details = pd.DataFrame(result['details'])
    df_details['period_start'] = df_details['period_start'].dt.strftime('%Y-%m-%d')
    df_details['period_end'] = df_details['period_end'].dt.strftime('%Y-%m-%d')
    
    # Formater les colonnes numériques
    numeric_cols = ['accrual_fraction', 'time_to_payment', 'discount_factor', 
                    'floating_rate', 'fixed_coupon', 'floating_coupon', 'pv_fixed', 'pv_floating']
    for col in numeric_cols:
        df_details[col] = df_details[col].apply(lambda x: f"{x:.6f}" if col in ['accrual_fraction', 'discount_factor', 'floating_rate'] else f"{x:,.2f}")
    
    st.dataframe(df_details, use_container_width=True)
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Flux de trésorerie")
        df_plot = pd.DataFrame(result['details'])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_plot['period_end'].dt.strftime('%Y-%m-%d'), 
                             y=df_plot['pv_fixed'], name='PV Fixe'))
        fig.add_trace(go.Bar(x=df_plot['period_end'].dt.strftime('%Y-%m-%d'), 
                             y=df_plot['pv_floating'], name='PV Flottant'))
        fig.update_layout(barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Taux flottants par période")
        df_plot = pd.DataFrame(result['details'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot['period_end'].dt.strftime('%Y-%m-%d'), 
                                 y=df_plot['floating_rate']*100, mode='lines+markers', name='Taux Flottant'))
        fig.update_yaxes(title_text="Taux (%)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Export
    st.divider()
    st.subheader("📥 Exporter les résultats")
    
    csv = df_details.to_csv(index=False)
    st.download_button(label="Télécharger CSV", data=csv, file_name="swap_details.csv", mime="text/csv")