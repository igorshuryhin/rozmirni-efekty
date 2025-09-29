# -*- coding: utf-8 -*-
# Інтерактивна візуалізація розмірних ефектів (Streamlit)
# Моделі:
#   A) Квантова точка: Eg(R) (модель ефективної маси)
#   B) Теплопровідність κ(L) з граничним розсіюванням
#
# Запуск: streamlit run app.py

import io
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --- Фізичні константи ---
HBAR = 1.054e-34       # Дж·с
E_CHARGE = 1.602e-19   # Кл
EPS0 = 8.854e-12       # Ф/м
M_E = 9.10938356e-31   # кг (маса електрона)

# --- Приблизні параметри матеріалів для демонстрації ---
@dataclass
class Material:
    Eg_bulk_eV: float
    me_eff: float
    mh_eff: float
    eps_r: float

MATERIALS = {
    "CdSe": Material(Eg_bulk_eV=1.74, me_eff=0.13*M_E, mh_eff=0.45*M_E, eps_r=9.5),
    "GaAs": Material(Eg_bulk_eV=1.42, me_eff=0.067*M_E, mh_eff=0.45*M_E, eps_r=12.9),
    "Si":   Material(Eg_bulk_eV=1.12, me_eff=0.26*M_E,  mh_eff=0.39*M_E, eps_r=11.7),
}

# --- Службові функції ---
def fig_to_png_bytes(fig, dpi=180) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

st.set_page_config(page_title="Розмірні ефекти у фізиці", layout="wide")

st.title("Розмірні ефекти у фізиці — інтерактивна візуалізація")

mode = st.sidebar.radio(
    "Оберіть модель",
    ["A) Квантова точка: Eg(R)", "B) Теплопровідність: κ(L)"],
)

st.sidebar.caption("Всі підписи та одиниці — у правій частині інтерфейсу.")

# =========================================================
# A) Квантова точка Eg(R)
# =========================================================
if mode.startswith("A"):
    st.subheader("A) Квантова точка: залежність ширини забороненої зони Eg від радіуса R")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        material_name = st.selectbox("Матеріал", list(MATERIALS.keys()), index=0)
        m = MATERIALS[material_name]

        r_min = st.number_input("Мінімальний радіус R, нм", min_value=0.1, value=1.0, step=0.1)
        r_max = st.number_input("Максимальний радіус R, нм", min_value=r_min+0.1, value=10.0, step=0.1)
        points = st.slider("Кількість точок", min_value=50, max_value=2000, value=400, step=50)

        st.markdown("**Параметри матеріалу (можеш підкоригувати):**")
        Eg_bulk_eV = st.number_input("Eg (bulk), еВ", min_value=0.0, value=float(m.Eg_bulk_eV), step=0.01)
        me_eff_rel = st.number_input("m*_e / m_e", min_value=0.01, value=float(m.me_eff / M_E), step=0.01)
        mh_eff_rel = st.number_input("m*_h / m_e", min_value=0.01, value=float(m.mh_eff / M_E), step=0.01)
        eps_r = st.number_input("Відносна діелектрична проникність ε_r", min_value=1.0, value=float(m.eps_r), step=0.1)

        # Обчислення
        R = np.linspace(r_min*1e-9, r_max*1e-9, points)  # в метрах
        Eg_bulk_J = Eg_bulk_eV * E_CHARGE
        me_eff = me_eff_rel * M_E
        mh_eff = mh_eff_rel * M_E

        conf = (HBAR**2 * (np.pi**2) / (2 * R**2)) * (1/me_eff + 1/mh_eff)  # Дж
        coul = 1.8 * E_CHARGE**2 / (4 * np.pi * EPS0 * eps_r * R)           # Дж
        Eg_J = Eg_bulk_J + conf - coul
        Eg_eV = Eg_J / E_CHARGE

        df = pd.DataFrame({"R, нм": R*1e9, "Eg, еВ": Eg_eV})

    with col_right:
        st.markdown("**Графік Eg(R)**")
        fig, ax = plt.subplots()
        ax.plot(df["R, нм"], df["Eg, еВ"])
        ax.set_xlabel("R, нм")
        ax.set_ylabel("Eg, еВ")
        ax.set_title(f"Квантове обмеження для {material_name}")
        ax.grid(True)
        st.pyplot(fig, clear_figure=True)

        # Завантаження
        png_bytes = fig_to_png_bytes(fig)
        csv_bytes = df_to_csv_bytes(df)

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button("⬇️ Завантажити PNG", data=png_bytes,
                               file_name=f"Eg_vs_R_{material_name}.png", mime="image/png")
        with dl_col2:
            st.download_button("⬇️ Завантажити CSV", data=csv_bytes,
                               file_name=f"Eg_vs_R_{material_name}.csv", mime="text/csv")

        st.markdown("**Пояснення:** при зменшенні R кінетичний внесок зростає (~1/R²), тому Eg збільшується; кулонівська поправка (~1/R) трохи зменшує енергію, але загальний тренд — до більшого Eg для менших частинок.")

# =========================================================
# B) Теплопровідність κ(L)
# =========================================================
else:
    st.subheader("B) Розмірна залежність теплопровідності κ(L) (граничне розсіювання)")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        l_min = st.number_input("Мінімальний розмір L, нм", min_value=0.1, value=1.0, step=0.1)
        l_max_um = st.number_input("Максимальний розмір L, мкм", min_value=0.1, value=100.0, step=0.1)
        points = st.slider("Кількість точок", min_value=50, max_value=2000, value=300, step=50)

        k_bulk = st.number_input("κ_bulk, Вт/м·К", min_value=0.0, value=200.0, step=1.0)
        lambda_nm = st.number_input("Ефективна довжина вільного пробігу Λ, нм", min_value=0.1, value=100.0, step=1.0)

        # сітка L в метрах (логарифмічно)
        L = np.logspace(np.log10(l_min*1e-9), np.log10(l_max_um*1e-6), points)
        Lambda = lambda_nm * 1e-9

        kappa = k_bulk / (1 + Lambda / L)

        df = pd.DataFrame({"L, нм": L*1e9, "κ(L), Вт/м·К": kappa})

    with col_right:
        st.markdown("**Графік κ(L)** (X лог-шкала)")
        fig, ax = plt.subplots()
        ax.semilogx(df["L, нм"], df["κ(L), Вт/м·К"])
        ax.set_xlabel("L, нм (лог шкала)")
        ax.set_ylabel("κ(L), Вт/м·К")
        ax.set_title("Граничне розсіювання: падіння κ при малих L")
        ax.grid(True, which="both")
        st.pyplot(fig, clear_figure=True)

        # Завантаження
        png_bytes = fig_to_png_bytes(fig)
        csv_bytes = df_to_csv_bytes(df)

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button("⬇️ Завантажити PNG", data=png_bytes,
                               file_name="kappa_vs_L.png", mime="image/png")
        with dl_col2:
            st.download_button("⬇️ Завантажити CSV", data=csv_bytes,
                               file_name="kappa_vs_L.csv", mime="text/csv")

        st.markdown("**Пояснення:** коли L порівняний або менший за Λ, зростає розсіювання на межах зразка, тому ефективна теплопровідність падає; при великих L → κ → κ_bulk.")
