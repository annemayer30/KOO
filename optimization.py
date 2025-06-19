import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from scipy.optimize import linprog
import math

@st.cache_data
def load_data():
    location_df = pd.read_excel("https://raw.githubusercontent.com/annemayer30/KOO/main/location.xlsx")
    traffic_df  = pd.read_excel("https://raw.githubusercontent.com/annemayer30/KOO/main/trafficData01.xlsx", header=None)
    load_df     = pd.read_excel("https://raw.githubusercontent.com/annemayer30/KOO/main/loadData.xlsx",  header=None)
    cost_df     = pd.read_excel("https://raw.githubusercontent.com/annemayer30/KOO/main/costData.xlsx",  header=None)
    time_df     = pd.read_excel("https://raw.githubusercontent.com/annemayer30/KOO/main/time.xlsx",      header=None)
    return location_df, traffic_df, load_df, cost_df, time_df.values.flatten()

def optimize_ess(Pload_raw, Cost, Ppv_raw, SoC_max, SoC_min, Einit_ratio, dt,
                 battery_range_kWh, pcs_range_kW,
                 battery_cost_per_kWh, pcs_cost_per_kW):
    N = len(Pload_raw)
    best_ROI = -np.inf
    best_result = None

    for batt_kWh in battery_range_kWh:
        battEnergy = batt_kWh * 3.6e6
        Emin = SoC_min * battEnergy
        Emax = SoC_max * battEnergy
        Einit = Einit_ratio * battEnergy

        for pcs_kW in pcs_range_kW:
            Pmin = -pcs_kW * 1e3
            Pmax = pcs_kW * 1e3

            c = np.concatenate([dt * Cost, np.zeros(N), np.zeros(N)])
            bounds = [(0, max(Pload_raw)*0.9)] * N + [(Pmin, Pmax)] * N + [(Emin, Emax)] * N

            A_eq = np.zeros((N + 1, 3 * N))
            b_eq = np.zeros(N + 1)
            A_eq[0, N] = dt
            A_eq[0, 2*N] = 1
            b_eq[0] = Einit
            for i in range(1, N):
                A_eq[i, N+i] = dt
                A_eq[i, 2*N + i] = 1
                A_eq[i, 2*N + i-1] = -1
            A_eq[N, 2*N + N - 1] = 1
            b_eq[N] = Einit

            A_load = np.zeros((N, 3*N))
            for i in range(N):
                A_load[i, i] = 1
                A_load[i, N+i] = 1
            b_load = Pload_raw - Ppv_raw

            res = linprog(
                c=c,
                A_eq=np.vstack([A_eq, A_load]),
                b_eq=np.concatenate([b_eq, b_load]),
                bounds=bounds,
                method='highs'
            )

            if not res.success:
                continue

            x = res.x
            Pgrid = x[:N]
            Pbatt = x[N:2*N]
            Ebatt = x[2*N:]
            SoC = Ebatt / battEnergy * 100

            LoadCost = np.sum(((Pload_raw - Ppv_raw) / 1e3) * Cost)
            GridCost = np.sum((Pgrid / 1e3) * Cost)
            true_sav = max(0, LoadCost - GridCost)
            annual_sv = true_sav * 365
            sv_10yr = annual_sv * 10

            batt_cost = batt_kWh * battery_cost_per_kWh
            pcs_cost = pcs_kW * pcs_cost_per_kW
            total_cost = batt_cost + pcs_cost

            ROI = (sv_10yr - total_cost) / total_cost * 100 if total_cost > 0 else -np.inf
            payback = total_cost / true_sav if true_sav > 0 else math.inf

            if ROI > best_ROI:
                best_ROI = ROI
                best_result = {
                    "batt_kWh": batt_kWh,
                    "pcs_kW": pcs_kW,
                    "Pgrid": Pgrid,
                    "Pbatt": Pbatt,
                    "Ebatt": Ebatt,
                    "SoC": SoC,
                    "Cost": Cost,
                    "Pload": Pload_raw,
                    "Ppv": Ppv_raw,
                    "thour": np.arange(1, N+1) * dt / 3600,
                    "summary": {
                        "Battery Capacity (kWh)": batt_kWh,
                        "PCS Rating (kW)": pcs_kW,
                        "Annual Saving (₩)": annual_sv,
                        "10‑yr Saving (₩)": sv_10yr,
                        "CAPEX Battery (₩)": batt_cost,
                        "CAPEX PCS (₩)": pcs_cost,
                        "Total CAPEX (₩)": total_cost,
                        "ROI (%)": ROI,
                        "Payback (days)": payback
                    }
                }

    return best_result

def main():
    st.set_page_config(layout="wide")
    st.title("서울시 압전 기반 ESS ROI 최적화 도구")

    SoC_max = st.sidebar.slider("Max SoC", 0.5, 1.0, 0.8)
    SoC_min = st.sidebar.slider("Min SoC", 0.0, 0.5, 0.2)
    Einit_ratio = st.sidebar.slider("Initial SoC Ratio", 0.0, 1.0, 0.3)
    loadSelect = st.sidebar.selectbox("Load Scenario (0‑4)", [0,1,2,3,4])
    loadBase = st.sidebar.number_input("Load Base [W]", value=350000.0)

    piezo_unit_output = st.sidebar.number_input("Piezo Output per Tile [W]", value=0.62, format="%.8f")
    piezo_count = st.sidebar.number_input("Piezo Tiles per Wheel", value=1000, step=1000)

    lamp_count = st.sidebar.number_input("Lamp Count", value=1000)
    lamp_power = st.sidebar.number_input("Lamp Power [W]", value=150.0)

    timeOptimize = st.sidebar.number_input("Optimization Interval [min]", value=60)
    battery_cost_per_kWh = st.sidebar.number_input("Battery Cost (₩/kWh)", value=400000)
    pcs_cost_per_kW = st.sidebar.number_input("PCS Cost (₩/kW)", value=300000)

    battery_range_kWh = np.arange(500, 3001, 500)
    pcs_range_kW = np.arange(100, 901, 200)

    location_df, traffic_df, load_df, cost_df, time_vec = load_data()
    address_list = traffic_df.iloc[0].tolist()
    traffic_values = traffic_df.iloc[1:].T.values

    m = folium.Map(location=[37.55, 126.98], zoom_start=11)
    for idx, row in location_df.iterrows():
        addr = row['지점 위치']
        if addr in address_list:
            folium.Marker(
                location=[row['위도'], row['경도']],
                popup=addr,
                tooltip=addr,
                icon=folium.Icon(color='blue')
            ).add_to(m)

    st_folium_obj = st_folium(m, width=1000, height=600)

    if st_folium_obj and st_folium_obj['last_object_clicked_tooltip']:
        clicked_addr = st_folium_obj['last_object_clicked_tooltip']
        st.subheader(f"ESS 최적화 결과 – {clicked_addr}")

        idx = address_list.index(clicked_addr)
        traffic_series = traffic_values[idx]
        Ppv_full = traffic_series * piezo_unit_output * piezo_count * 4

        dt = timeOptimize * 60
        stepAdjust = int(dt / (time_vec[1] - time_vec[0]))
        Ppv = Ppv_full[2::stepAdjust][:len(time_vec[2::stepAdjust])]
        N = len(Ppv)

        if loadSelect == 4:
            Pload_raw = load_df.iloc[2::stepAdjust, loadSelect].values[:N] * (lamp_count * lamp_power / 1000)
        else:
            Pload_raw = load_df.iloc[2::stepAdjust, loadSelect].values[:N] + loadBase

        Cost = cost_df.iloc[2::stepAdjust, 0].values[:N]

        best = optimize_ess(Pload_raw, Cost, Ppv, SoC_max, SoC_min, Einit_ratio, dt,
                            battery_range_kWh, pcs_range_kW,
                            battery_cost_per_kWh, pcs_cost_per_kW)

        if best is None:
            st.error("유효한 해를 찾지 못했습니다.")
            return

        st.dataframe(pd.DataFrame([best['summary']]))

        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        ax1 = axes[0]
        ax1.plot(best['thour'], best['Ebatt'] / 3.6e6, label='Battery Energy [kWh]', color='tab:blue', linewidth=1.5)
        ax1.set_ylabel("Battery Energy [kWh]", color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True)
        ax1.set_xlim([1, 24])

        ax1b = ax1.twinx()
        ax1b.plot(best['thour'], best['SoC'], 'r-', label='SoC (%)', linewidth=1.5)
        ax1b.axhspan(SoC_min*100, SoC_max*100, color='gray', alpha=0.2)
        ax1b.set_ylabel("SoC [%]", color='tab:red')
        ax1b.tick_params(axis='y', labelcolor='tab:red')
        ax1b.set_ylim(0, 100)

        axes[1].plot(best['thour'], best['Cost'], linewidth=1.5)
        axes[1].set_ylabel("Grid Price [₩/kWh]")
        axes[1].grid(True)
        axes[1].set_xlim([1, 24])

        axes[2].plot(best['thour'], best['Pgrid']/1e3, label='Grid [kW]')
        axes[2].plot(best['thour'], best['Pload']/1e3, label='Load [kW]')
        axes[2].plot(best['thour'], best['Pbatt']/1e3, label='Battery [kW]')
        axes[2].plot(best['thour'], best['Ppv']/1e3, label='Piezo PV [kW]')
        axes[2].set_ylabel("Power [kW]")
        axes[2].grid(True)
        axes[2].set_xlim([1, 24])
        axes[2].legend()

        st.pyplot(fig)

if __name__ == "__main__":
    main()
