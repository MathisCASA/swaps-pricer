import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline
import pandas as pd
import plotly.graph_objects as go
from enum import Enum

class SwapLegType(Enum):
    FIXED = "Fixe"
    FLOATING = "Flottante"
    INFLATION = "Inflation"

class SwapPricer:
    def __init__(self, curve_data, rates_dict=None, is_ois=False):
        self.maturities = np.array(curve_data['maturities'])
        self.rates = np.array(curve_data['rates'])
        self.cs = CubicSpline(self.maturities, self.rates)
        self.rates_dict = rates_dict or {}
        self.is_ois = is_ois
        
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
    
    def get_rate_at_maturity(self, t):
        """Récupère le taux à une maturity donnée (en années)"""
        if t < self.maturities[0]:
            return self.rates[0]
        elif t > self.maturities[-1]:
            return self.rates[-1]
        else:
            return self.cs(t)
    
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
    
    def _get_rate_key(self, tenor_months):
        if tenor_months == 3:
            return '3M'
        elif tenor_months == 6:
            return '6M'
        elif tenor_months == 12:
            return '12M'
        else:
            return None
    
    def calculate_ois_daily_composition(self, period_start, period_end, pricing_date, 
                                       day_count_convention, spot_rate):
        """
        Compose le taux OIS jour par jour en interpolant sur la courbe.
        spot_rate : le taux intraday actuel (taux overnight du jour)
        """
        current_date = period_start
        compounded_rate = 1.0
        
        while current_date < period_end:
            next_date = current_date + timedelta(days=1)
            if next_date > period_end:
                next_date = period_end
            
            days_from_pricing = (current_date - pricing_date).days
            maturity_years = days_from_pricing / 365.25
            
            if current_date == period_start and current_date == pricing_date:
                daily_rate = spot_rate
            else:
                daily_rate = self.get_rate_at_maturity(maturity_years)
            
            daily_fraction = self.day_count_fraction(current_date, next_date, day_count_convention)
            compounded_rate *= (1 + daily_rate * daily_fraction)
            current_date = next_date
        
        accrual_fraction = self.day_count_fraction(period_start, period_end, day_count_convention)
        ois_rate = (compounded_rate - 1) / accrual_fraction
        
        return ois_rate
    
    def price_swap(self, pricing_date, start_date, end_date, fixed_rate, 
                   spread, floating_tenor_months, day_count_convention, nominal,
                   spot_rate=None, historical_rates=None):

        payment_dates = self.generate_payment_dates(start_date, end_date, 
                                                     floating_tenor_months)
        
        if payment_dates[-1] != end_date:
            payment_dates.append(end_date)
        
        pv_fixed = 0
        pv_floating = 0
        details = []
        rate_key = self._get_rate_key(floating_tenor_months)
        
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
            
            if self.is_ois:
                floating_rate = self.calculate_ois_daily_composition(period_start, period_end, 
                                                                     pricing_date, day_count_convention,
                                                                     spot_rate)
                floating_rate += spread
                rate_source = "OIS Composé"
            else:
                if period_start < pricing_date and historical_rates and period_start in historical_rates:
                    floating_rate = historical_rates[period_start] + spread
                    rate_source = "Historique"
                elif rate_key in self.rates_dict:
                    floating_rate = self.rates_dict[rate_key] + spread
                    rate_source = "Spot"
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


class InflationSwapPricer:
    """
    Pricer pour les Inflation-Linked Swaps
    Jambe 1 : Taux fixe d'inflation (coupon d'inflation constant)
    Jambe 2 : EURIBOR variable + spread
    """
    
    def __init__(self, inflation_curve_data, nominal_curve, rates_dict=None):
        """
        inflation_curve_data: dict avec 'maturities' et 'rates' (taux ZC inflation de Bloomberg)
        nominal_curve: SwapPricer object pour les taux nominaux (EURIBOR)
        rates_dict: dict avec clés '3M', '6M', '12M' pour les taux spot
        """
        self.inflation_maturities = np.array(inflation_curve_data['maturities'])
        self.inflation_rates = np.array(inflation_curve_data['rates'])
        self.inflation_cs = CubicSpline(self.inflation_maturities, self.inflation_rates)
        
        self.nominal_pricer = nominal_curve
        self.rates_dict = rates_dict or {}
    
    def get_inflation_rate_at_maturity(self, t):
        """Récupère le taux ZC inflation à une maturity donnée (en années)"""
        if t < self.inflation_maturities[0]:
            return self.inflation_rates[0]
        elif t > self.inflation_maturities[-1]:
            return self.inflation_rates[-1]
        else:
            return self.inflation_cs(t)
    
    def get_forward_inflation_rate(self, t1, t2):
        """Calcule le taux d'inflation forward entre t1 et t2"""
        if t1 == 0:
            r1 = self.inflation_rates[0]
        else:
            r1 = self.inflation_cs(t1)
        r2 = self.inflation_cs(t2)
        
        forward = ((1 + r2) ** t2 / (1 + r1) ** t1 - 1) ** (1 / (t2 - t1)) - 1
        return forward
    
    def day_count_fraction(self, start_date, end_date, convention):
        """Calcule la fraction de jour"""
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
        """Génère les dates de paiement"""
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
    
    def _get_rate_key(self, tenor_months):
        if tenor_months == 3:
            return '3M'
        elif tenor_months == 6:
            return '6M'
        elif tenor_months == 12:
            return '12M'
        else:
            return None
    
    def price_inflation_swap(self, pricing_date, start_date, end_date, 
                            inflation_fixed_rate, spread, floating_tenor_months,
                            day_count_convention, nominal,
                            seasonality_adjustment=0.0):
        """
        Price un Inflation-Linked Swap
        
        Jambe 1 (Inflation) : Coupon fixe d'inflation * Nominal
        Jambe 2 (Flottante) : EURIBOR forward + spread
        
        seasonality_adjustment: ajustement saisonnier (pour les indices saisonniers)
        """
        
        payment_dates = self.generate_payment_dates(start_date, end_date, floating_tenor_months)
        
        if payment_dates[-1] != end_date:
            payment_dates.append(end_date)
        
        pv_inflation_leg = 0
        pv_floating_leg = 0
        details = []
        rate_key = self._get_rate_key(floating_tenor_months)
        
        for i in range(len(payment_dates) - 1):
            period_start = payment_dates[i]
            period_end = payment_dates[i + 1]
            
            accrual_fraction = self.day_count_fraction(period_start, period_end, 
                                                       day_count_convention)
            
            time_to_payment = (period_end - pricing_date).days / 365.25
            
            if time_to_payment < 0:
                continue
            
            # Discount factor (courbe nominale EURIBOR)
            df = self.nominal_pricer.get_discount_factor(time_to_payment)
            
            # ========== JAMBE INFLATION ==========
            # Coupon fixe d'inflation
            inflation_coupon = inflation_fixed_rate * accrual_fraction * nominal
            pv_inflation_leg += inflation_coupon * df
            
            # ========== JAMBE FLOTTANTE (EURIBOR) ==========
            # Récupérer le taux EURIBOR forward
            if period_start < pricing_date:
                # Utiliser le spot si passé
                if rate_key in self.rates_dict:
                    floating_rate = self.rates_dict[rate_key] + spread
                    rate_source = "Spot"
                else:
                    floating_rate = self.nominal_pricer.get_rate_at_maturity(0) + spread
                    rate_source = "Spot (Default)"
            else:
                # Utiliser le forward
                t1 = max(0, (period_start - pricing_date).days / 365.25)
                t2 = (period_end - pricing_date).days / 365.25
                forward_rate = self.nominal_pricer.get_forward_rate(t1, t2)
                floating_rate = forward_rate + spread
                rate_source = "Forward"
            
            floating_coupon = floating_rate * accrual_fraction * nominal
            pv_floating_leg += floating_coupon * df
            
            # Taux forward d'inflation pour le détail
            t1 = max(0, (period_start - pricing_date).days / 365.25)
            t2 = (period_end - pricing_date).days / 365.25
            forward_inflation = self.get_forward_inflation_rate(t1, t2)
            
            details.append({
                'period_start': period_start,
                'period_end': period_end,
                'accrual_fraction': accrual_fraction,
                'time_to_payment': time_to_payment,
                'discount_factor': df,
                'inflation_fixed_rate': inflation_fixed_rate,
                'forward_inflation_rate': forward_inflation,
                'floating_rate': floating_rate,
                'rate_source': rate_source,
                'inflation_coupon': inflation_coupon,
                'floating_coupon': floating_coupon,
                'pv_inflation': inflation_coupon * df,
                'pv_floating': floating_coupon * df
            })
        
        swap_value = pv_floating_leg - pv_inflation_leg
        
        return {
            'swap_value': swap_value,
            'pv_inflation_leg': pv_inflation_leg,
            'pv_floating_leg': pv_floating_leg,
            'details': details,
            'nominal': nominal,
            'swap_type': 'Inflation-Linked'
        }


def display_vanilla_swap_results(result):
    """Affiche les résultats du pricing pour vanilla swaps"""
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
    
    st.subheader("📊 Détails des Périodes")
    
    df_details = pd.DataFrame(result['details'])
    df_details['period_start'] = df_details['period_start'].dt.strftime('%Y-%m-%d')
    df_details['period_end'] = df_details['period_end'].dt.strftime('%Y-%m-%d')
    
    numeric_cols = ['accrual_fraction', 'time_to_payment', 'discount_factor', 
                    'floating_rate', 'fixed_coupon', 'floating_coupon', 'pv_fixed', 'pv_floating']
    for col in numeric_cols:
        df_details[col] = df_details[col].apply(lambda x: f"{x:.6f}" if col in ['accrual_fraction', 'discount_factor', 'floating_rate'] else f"{x:,.2f}")
    
    st.dataframe(df_details, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Flux de trésorerie")
        df_plot = pd.DataFrame(result['details'])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_plot['period_end'].dt.strftime('%Y-%m-%d'), 
                             y=df_plot['pv_fixed'], name='PV Fixe'))
        fig.add_trace(go.Bar(x=df_plot['period_end'].dt.strftime('%Y-%m-%d'), 
                             y=df_plot['pv_floating'], name='PV Flottant'))
        fig.update_layout(barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Taux flottants par période")
        df_plot = pd.DataFrame(result['details'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot['period_end'].dt.strftime('%Y-%m-%d'), 
                                 y=df_plot['floating_rate']*100, mode='lines+markers', name='Taux Flottant'))
        fig.update_yaxes(title_text="Taux (%)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.subheader("📥 Exporter les résultats")
    
    csv = df_details.to_csv(index=False)
    st.download_button(label="Télécharger CSV", data=csv, file_name="swap_details.csv", mime="text/csv")


def display_inflation_swap_results(result):
    """Affiche les résultats du pricing pour inflation swaps"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Valeur du Swap", f"€ {result['swap_value']:,.0f}")
    with col2:
        st.metric("PV Jambe Inflation", f"€ {result['pv_inflation_leg']:,.0f}")
    with col3:
        st.metric("PV Jambe Flottante", f"€ {result['pv_floating_leg']:,.0f}")
    with col4:
        st.metric("Nominal", f"€ {result['nominal']:,.0f}")
    
    st.divider()
    
    st.subheader("📊 Détails des Périodes")
    
    df_details = pd.DataFrame(result['details'])
    df_details['period_start'] = df_details['period_start'].dt.strftime('%Y-%m-%d')
    df_details['period_end'] = df_details['period_end'].dt.strftime('%Y-%m-%d')
    
    numeric_cols = ['accrual_fraction', 'time_to_payment', 'discount_factor', 
                    'inflation_fixed_rate', 'forward_inflation_rate', 'floating_rate', 
                    'inflation_coupon', 'floating_coupon', 'pv_inflation', 'pv_floating']
    for col in numeric_cols:
        if col in df_details.columns:
            df_details[col] = df_details[col].apply(
                lambda x: f"{x:.6f}" if col in ['accrual_fraction', 'discount_factor', 'inflation_fixed_rate', 
                                                  'forward_inflation_rate', 'floating_rate'] else f"{x:,.2f}"
            )
    
    st.dataframe(df_details, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Flux de trésorerie par jambe")
        df_plot = pd.DataFrame(result['details'])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_plot['period_end'].dt.strftime('%Y-%m-%d'), 
                             y=df_plot['pv_inflation'], name='PV Inflation'))
        fig.add_trace(go.Bar(x=df_plot['period_end'].dt.strftime('%Y-%m-%d'), 
                             y=df_plot['pv_floating'], name='PV Flottante (EURIBOR)'))
        fig.update_layout(barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Comparaison Taux Fixes vs Forwards")
        df_plot = pd.DataFrame(result['details'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot['period_end'].dt.strftime('%Y-%m-%d'), 
                                 y=df_plot['inflation_fixed_rate']*100, mode='lines+markers', 
                                 name='Taux Fixe Inflation'))
        fig.add_trace(go.Scatter(x=df_plot['period_end'].dt.strftime('%Y-%m-%d'), 
                                 y=df_plot['forward_inflation_rate']*100, mode='lines+markers', 
                                 name='Taux Forward Inflation'))
        fig.add_trace(go.Scatter(x=df_plot['period_end'].dt.strftime('%Y-%m-%d'), 
                                 y=df_plot['floating_rate']*100, mode='lines+markers', 
                                 name='Taux EURIBOR Forward'))
        fig.update_yaxes(title_text="Taux (%)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.subheader("📥 Exporter les résultats")
    
    csv = df_details.to_csv(index=False)
    st.download_button(label="Télécharger CSV", data=csv, file_name="inflation_swap_details.csv", mime="text/csv")


def create_vanilla_pricer_page(page_name, curve_data, default_spot_rate, is_ois=False):
    """Crée une page de pricer pour une devise donnée (Vanilla Swaps)"""
    st.title(f"🔄 Vanilla Swap Pricer {page_name}")
    
    st.sidebar.header(f"⚙️ Paramètres du Swap {page_name}")
    
    min_date = datetime(2000, 1, 1).date()
    max_date = datetime(2050, 12, 31).date()
    
    pricing_date = st.sidebar.date_input("Date de pricing", 
                                         value=datetime(2026, 3, 18).date(), 
                                         min_value=min_date,
                                         max_value=max_date,
                                         key=f"pricing_{page_name}")
    start_date = st.sidebar.date_input("Date de début", 
                                       value=datetime(2025, 5, 19).date(), 
                                       min_value=min_date,
                                       max_value=max_date,
                                       key=f"start_{page_name}")
    end_date = st.sidebar.date_input("Date de fin", 
                                     value=datetime(2030, 5, 19).date(), 
                                     min_value=min_date,
                                     max_value=max_date,
                                     key=f"end_{page_name}")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        fixed_rate = st.number_input("Taux fixe (%)", value=0.0, step=0.0001, format="%.6f", key=f"fixed_{page_name}") / 100
        spread = st.number_input("Spread (%)", value=0.0, step=0.0001, format="%.6f", key=f"spread_{page_name}") / 100
    
    with col2:
        nominal = st.number_input("Nominal (€)", value=180000000, step=1000000, key=f"nominal_{page_name}")
        floating_tenor = st.selectbox("Tenor flottant", [3, 6, 12], index=1, key=f"tenor_{page_name}")
    
    day_count = st.sidebar.selectbox("Convention de comptage", 
                                      ["ACT/360", "ACT/365", "ACT/ACT", "30/360"], key=f"daycount_{page_name}")
    
    st.sidebar.divider()
    
    rates_dict = {}
    spot_rate = None
    
    if is_ois:
        st.sidebar.subheader(f"🌙 Taux Spot {page_name}")
        spot_rate = st.sidebar.number_input(f"Taux Spot {page_name} (%)", 
                                            value=default_spot_rate*100, 
                                            step=0.0001, format="%.6f",
                                            key=f"spot_{page_name}") / 100
    else:
        st.sidebar.subheader(f"📊 Taux {page_name}")
        tenor_label = f"{floating_tenor}M"
        rate_value = st.sidebar.number_input(f"Taux {tenor_label} (%)", 
                                             value=default_spot_rate*100, 
                                             step=0.0001, format="%.6f",
                                             key=f"rate_{page_name}") / 100
        rates_dict[tenor_label] = rate_value
    
    pricing_date = datetime.combine(pricing_date, datetime.min.time())
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.min.time())
    
    pricer = SwapPricer(curve_data, rates_dict=rates_dict, is_ois=is_ois)
    
    if st.sidebar.button("🚀 Calculer Vanilla Swap", use_container_width=True, key=f"calc_{page_name}"):
        result = pricer.price_swap(pricing_date, start_date, end_date, fixed_rate, 
                                   spread, floating_tenor, day_count, nominal,
                                   spot_rate=spot_rate)
        st.session_state[f'result_{page_name}'] = result
    
    if f'result_{page_name}' in st.session_state:
        display_vanilla_swap_results(st.session_state[f'result_{page_name}'])


def create_inflation_pricer_page(curve_data_euribor, inflation_curve_data):
    """Crée une page de pricer pour les Inflation Swaps"""
    st.title("🌍 Inflation-Linked Swap Pricer")
    
    st.sidebar.header("⚙️ Paramètres du Swap Inflation")
    
    min_date = datetime(2000, 1, 1).date()
    max_date = datetime(2050, 12, 31).date()
    
    pricing_date = st.sidebar.date_input("Date de pricing", 
                                         value=datetime(2026, 3, 18).date(), 
                                         min_value=min_date,
                                         max_value=max_date,
                                         key="pricing_inflation")
    start_date = st.sidebar.date_input("Date de début", 
                                       value=datetime(2025, 5, 19).date(), 
                                       min_value=min_date,
                                       max_value=max_date,
                                       key="start_inflation")
    end_date = st.sidebar.date_input("Date de fin", 
                                     value=datetime(2030, 5, 19).date(), 
                                     min_value=min_date,
                                     max_value=max_date,
                                     key="end_inflation")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        inflation_fixed_rate = st.number_input("Taux Fixe Inflation (%)", value=2.0, step=0.0001, format="%.6f", key="infl_fixed") / 100
        spread = st.number_input("Spread EURIBOR (%)", value=0.0, step=0.0001, format="%.6f", key="infl_spread") / 100
    
    with col2:
        nominal = st.number_input("Nominal (€)", value=180000000, step=1000000, key="infl_nominal")
        floating_tenor = st.selectbox("Tenor flottant (EURIBOR)", [3, 6, 12], index=1, key="infl_tenor")
    
    day_count = st.sidebar.selectbox("Convention de comptage", 
                                      ["ACT/360", "ACT/365", "ACT/ACT", "30/360"], key="infl_daycount")
    
    st.sidebar.divider()
    
    st.sidebar.subheader("📈 Données Inflation (Bloomberg ZCPN)")
    st.sidebar.info("⚠️ Conseil : Injecte ta courbe ZC Inflation Bloomberg directement")
    
    tenor_label = f"{floating_tenor}M"
    euribor_rate = st.sidebar.number_input(f"Taux EURIBOR {tenor_label} (%)", 
                                          value=2.122, 
                                          step=0.0001, format="%.6f",
                                          key="euribor_rate") / 100
    
    rates_dict = {tenor_label: euribor_rate}
    
    pricing_date = datetime.combine(pricing_date, datetime.min.time())
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.min.time())
    
    # Créer le pricer EURIBOR
    nominal_pricer = SwapPricer(curve_data_euribor, rates_dict=rates_dict, is_ois=False)
    
    # Créer le pricer Inflation
    inflation_pricer = InflationSwapPricer(inflation_curve_data, nominal_pricer, rates_dict=rates_dict)
    
    if st.sidebar.button("🚀 Calculer Inflation Swap", use_container_width=True, key="calc_inflation"):
        result = inflation_pricer.price_inflation_swap(pricing_date, start_date, end_date, 
                                                       inflation_fixed_rate, spread, 
                                                       floating_tenor, day_count, nominal)
        st.session_state['result_inflation'] = result
    
    if 'result_inflation' in st.session_state:
        display_inflation_swap_results(st.session_state['result_inflation'])


# ============================================================================
# CONFIGURATION STREAMLIT & DONNÉES
# ============================================================================

st.set_page_config(page_title="Advanced Swap Pricer", layout="wide")

# Données des courbes EURIBOR
curve_data_euribor = {
    'maturities': [0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    'rates': [0.01963708, 0.01955476, 0.01951525, 0.01951413, 0.01981745, 0.02046718, 
              0.02131799, 0.02227016, 0.0232563, 0.02423195, 0.0251688, 0.02604979, 
              0.02686562, 0.02761227, 0.02828921, 0.02889817, 0.02944227, 0.02992544, 
              0.03035198, 0.03072635, 0.03105292, 0.03133594, 0.03157939, 0.03178702, 
              0.03196229, 0.03210839, 0.03222823, 0.03232447, 0.03239953, 0.03245559, 
              0.03249465, 0.0325185, 0.03251876]
}

# Données des courbes SOFR
curve_data_sofr = {
    'maturities': [1/52, 2/52, 3/52, 1/12, 2/12, 3/12, 4/12, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 11/12, 1, 
                   1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50],
    'rates': [0.03537, 0.03560, 0.03561, 0.03576, 0.03607, 0.03633, 0.03653, 0.03676, 0.03702, 0.03725, 0.03756, 0.03785, 0.03809, 0.03837, 0.03862,
              0.03936, 0.03974, 0.03994, 0.04013, 0.04044, 0.04082, 0.04122, 0.04161, 0.04199, 0.04237, 0.04311, 0.04404, 0.04476, 0.04469, 0.04428, 0.04310, 0.04178]
}

# Données des courbes ESTR
curve_data_estr = {
    'maturities': [1/52, 2/52, 1/12, 2/12, 3/12, 4/12, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 11/12, 1, 
                   1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 30, 40, 50],
    'rates': [0.01932, 0.01932, 0.01969, 0.02060, 0.02129, 0.02181, 0.02242, 0.02296, 0.02335, 0.02381, 0.02422, 0.02455, 0.02487, 0.02518,
              0.02609, 0.02647, 0.02678, 0.02713, 0.02754, 0.02798, 0.02847, 0.02899, 0.02947, 0.02993, 0.03033, 0.03078, 0.03174, 0.03232, 0.03216, 0.03178, 0.03070, 0.02946]
}

# EXEMPLE : Courbe inflation ZCPN (Bloomberg format) - À remplacer par ta propre courbe
curve_data_inflation = {
    'maturities': [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30],
    'rates': [0.01850, 0.02100, 0.02250, 0.02300, 0.02280, 0.02250, 0.02220, 0.02180, 0.02150, 0.02120, 0.02100, 0.02050, 0.01950, 0.01850, 0.01800, 0.01750]
}

# Navigation avec onglets
tab_vanilla, tab_inflation = st.tabs(["🔄 Vanilla Swaps", "🌍 Inflation Swaps"])

with tab_vanilla:
    st.header("Vanilla Swap Pricer")
    
    page = st.radio("📍 Sélectionner une devise", ["EURIBOR", "SOFR", "ESTR"], key="vanilla_page", horizontal=True)
    
    if page == "EURIBOR":
        create_vanilla_pricer_page("EURIBOR", curve_data_euribor, 0.02122, is_ois=False)
    elif page == "SOFR":
        create_vanilla_pricer_page("SOFR", curve_data_sofr, 0.03537, is_ois=True)
    elif page == "ESTR":
        create_vanilla_pricer_page("ESTR", curve_data_estr, 0.01932, is_ois=True)

with tab_inflation:
    create_inflation_pricer_page(curve_data_euribor, curve_data_inflation)
