import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Ambang Batas Aman NO2",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul aplikasi
st.title("Sistem Prediksi Ambang Batas Aman NO2")
st.markdown("---")
st.markdown("**Prediksi konsentrasi NO2 berdasarkan data historis dengan evaluasi standar WHO**")

# Konstanta
CONVERSION_FACTOR = 46010  # mol/m² to µg/m³
WHO_ANNUAL = 10  # µg/m³
WHO_24HOUR = 25  # µg/m³

# Fungsi load model
@st.cache_resource
def load_models():
    """Load model dan scaler"""
    try:
        model = joblib.load('knn_model.pkl')
        scaler = joblib.load('minmax_scaler.pkl')
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Model file tidak ditemukan: {str(e)}")
        st.info("Pastikan file 'knn_model.pkl' dan 'minmax_scaler.pkl' ada di direktori yang sama.")
        return None, None

# Fungsi evaluasi WHO
def evaluate_who_status(no2_ugm3):
    """Evaluasi status berdasarkan standar WHO"""
    if no2_ugm3 <= WHO_ANNUAL:
        return "SANGAT BAIK", "green", "✅"
    elif no2_ugm3 <= WHO_24HOUR:
        return "PERHATIAN", "orange", "⚠️"
    else:
        return "BERBAHAYA", "red", "❌"

# Fungsi prediksi
def predict_no2(model, scaler, no2_t2, no2_t1):
    """Prediksi NO2 berdasarkan 2 hari sebelumnya"""
    try:
        # Siapkan input data
        input_data = np.array([[no2_t2, no2_t1]])
        
        # Normalisasi
        input_scaled = scaler.transform(input_data)
        
        # Prediksi
        prediction_mol = model.predict(input_scaled)[0]
        prediction_ugm3 = prediction_mol * CONVERSION_FACTOR
        
        return prediction_mol, prediction_ugm3
    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}")
        return None, None

# Load model dan scaler
model, scaler = load_models()

if model is not None and scaler is not None:
    # Sidebar - Informasi Model
    st.sidebar.header("Informasi Model")
    st.sidebar.write("**Model**: K-Nearest Neighbors Regressor")
    st.sidebar.write("**Preprocessing**: MinMax Scaler")
    st.sidebar.write("**Input**: NO2(t-2) dan NO2(t-1)")
    st.sidebar.write("**Output**: Prediksi NO2(t)")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Standar WHO")
    st.sidebar.write(f"🟢 **Sangat Baik**: ≤ {WHO_ANNUAL} µg/m³")
    st.sidebar.write(f"🟡 **Perhatian**: ≤ {WHO_24HOUR} µg/m³")
    st.sidebar.write(f"🔴 **Berbahaya**: > {WHO_24HOUR} µg/m³")
    
    # Main interface
    st.header("Input Data NO2")
    
    # Pilihan input method
    input_method = st.radio(
        "Pilih metode input:",
        ["Manual Input", "Skenario Pre-defined"]
    )
    
    if input_method == "Manual Input":
        col1, col2 = st.columns(2)
        
        with col1:
            no2_t2 = st.number_input(
                "NO2 (t-2) dalam mol/m²",
                min_value=0.0,
                max_value=0.01,
                value=0.0001,
                step=0.000001,
                format="%.6f",
                help="Konsentrasi NO2 dua hari yang lalu"
            )
        
        with col2:
            no2_t1 = st.number_input(
                "NO2 (t-1) dalam mol/m²",
                min_value=0.0,
                max_value=0.01,
                value=0.00012,
                step=0.000001,
                format="%.6f",
                help="Konsentrasi NO2 satu hari yang lalu"
            )
    
    else:
        st.subheader("Skenario Pre-defined")
        scenario_option = st.selectbox(
            "Pilih skenario:",
            [
                "Skenario 1 - Rendah (Sangat Baik)",
                "Skenario 2 - Sedang (Perhatian)", 
                "Skenario 3 - Tinggi (Berbahaya)"
            ]
        )
        
        # Skenario data
        scenarios = {
            "Skenario 1 - Rendah (Sangat Baik)": [0.10, 0.12],
            "Skenario 2 - Sedang (Perhatian)": [0.00035, 0.00040],
            "Skenario 3 - Tinggi (Berbahaya)": [0.00080, 0.00090]
        }
        
        no2_t2, no2_t1 = scenarios[scenario_option]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("NO2 (t-2)", f"{no2_t2:.6f} mol/m²")
        with col2:
            st.metric("NO2 (t-1)", f"{no2_t1:.6f} mol/m²")
    
    # Tombol prediksi
    if st.button("Prediksi NO2", type="primary"):
        with st.spinner("Melakukan prediksi..."):
            # Prediksi
            pred_mol, pred_ugm3 = predict_no2(model, scaler, no2_t2, no2_t1)
            
            if pred_mol is not None:
                # Evaluasi WHO
                status, color, icon = evaluate_who_status(pred_ugm3)
                
                # Display hasil
                st.subheader("Hasil Prediksi")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Prediksi NO2",
                        f"{pred_mol:.6f} mol/m²",
                        help="Prediksi konsentrasi NO2"
                    )
                
                with col2:
                    st.metric(
                        "Konversi µg/m³", 
                        f"{pred_ugm3:.2f} µg/m³",
                        help="Konversi ke mikrogram per meter kubik"
                    )
                
                with col3:
                    st.metric(
                        "Status WHO",
                        f"{icon} {status}",
                        help="Evaluasi berdasarkan standar WHO"
                    )
                
                # Visualisasi
                st.subheader("Visualisasi Hasil")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Plot 1: Timeline mol/m²
                days = ['t-2', 't-1', 'PREDIKSI']
                values_mol = [no2_t2, no2_t1, pred_mol]
                colors = ['lightblue', 'skyblue', color]
                
                bars1 = ax1.bar(days, values_mol, color=colors, alpha=0.7, edgecolor='black')
                ax1.set_ylabel('NO2 (mol/m²)')
                ax1.set_title('Timeline Prediksi NO2')
                ax1.grid(True, alpha=0.3)
                
                for bar, val in zip(bars1, values_mol):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{val:.6f}', ha='center', va='bottom', fontsize=9)
                
                # Plot 2: Evaluasi WHO µg/m³
                values_ugm3 = [v * CONVERSION_FACTOR for v in [no2_t2, no2_t1]] + [pred_ugm3]
                
                bars2 = ax2.bar(days, values_ugm3, color=colors, alpha=0.7, edgecolor='black')
                ax2.axhline(WHO_ANNUAL, color='green', linestyle='--', linewidth=2, 
                           label=f'WHO Annual ({WHO_ANNUAL} µg/m³)')
                ax2.axhline(WHO_24HOUR, color='red', linestyle='--', linewidth=2,
                           label=f'WHO 24-Hour ({WHO_24HOUR} µg/m³)')
                
                ax2.set_ylabel('NO2 (µg/m³)')
                ax2.set_title(f'Evaluasi WHO: {status}')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                for bar, val in zip(bars2, values_ugm3):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{val:.2f}', ha='center', va='bottom', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Interpretasi hasil
                st.subheader("Interpretasi Hasil")
                
                if status == "SANGAT BAIK":
                    st.success(f"""
                    ✅ **Kondisi Sangat Baik**
                    
                    Konsentrasi NO2 yang diprediksi ({pred_ugm3:.2f} µg/m³) berada di bawah ambang batas WHO 
                    untuk rata-rata tahunan ({WHO_ANNUAL} µg/m³). Kondisi udara sangat baik dan aman 
                    untuk kesehatan masyarakat.
                    """)
                elif status == "PERHATIAN":
                    st.warning(f"""
                    ⚠️ **Perlu Perhatian**
                    
                    Konsentrasi NO2 yang diprediksi ({pred_ugm3:.2f} µg/m³) berada di antara ambang batas 
                    WHO tahunan ({WHO_ANNUAL} µg/m³) dan 24-jam ({WHO_24HOUR} µg/m³). 
                    Mulai perlu dilakukan monitoring dan upaya pengurangan emisi.
                    """)
                else:
                    st.error(f"""
                    ❌ **Kondisi Berbahaya**
                    
                    Konsentrasi NO2 yang diprediksi ({pred_ugm3:.2f} µg/m³) melebihi ambang batas WHO 
                    untuk 24-jam ({WHO_24HOUR} µg/m³). Kondisi ini berbahaya bagi kesehatan dan 
                    memerlukan tindakan segera untuk mengurangi emisi NO2.
                    """)
                
                # Prediksi multi-hari (opsional)
                with st.expander("Prediksi Multi-Hari (Eksperimental)"):
                    if st.button("Prediksi 2 Hari Ke Depan"):
                        st.write("Prediksi berurutan untuk 2 hari ke depan:")
                        
                        current_input = np.array([no2_t1, pred_mol])
                        predictions_multi = []
                        
                        for day in range(1, 3):
                            input_scaled = scaler.transform([current_input])
                            pred_mol_day = model.predict(input_scaled)[0]
                            pred_ugm3_day = pred_mol_day * CONVERSION_FACTOR
                            status_day, _, icon_day = evaluate_who_status(pred_ugm3_day)
                            
                            predictions_multi.append({
                                'Hari': f'+{day}',
                                'NO2 (mol/m²)': f"{pred_mol_day:.6f}",
                                'NO2 (µg/m³)': f"{pred_ugm3_day:.2f}",
                                'Status WHO': f"{icon_day} {status_day}"
                            })
                            
                            # Update untuk prediksi berikutnya
                            current_input = np.array([current_input[1], pred_mol_day])
                        
                        st.dataframe(pd.DataFrame(predictions_multi), use_container_width=True)
    
    # Footer informasi
    st.markdown("---")
    st.markdown("""
    **Tentang Aplikasi:**
    
    Aplikasi ini menggunakan model K-Nearest Neighbors (KNN) untuk memprediksi konsentrasi NO2 
    berdasarkan data historis 2 hari sebelumnya. Model telah dilatih dengan data time series 
    dan dievaluasi menggunakan standar kualitas udara WHO.
    
    **Catatan Penting:**
    - Model ini untuk tujuan akademis/penelitian
    - Prediksi multi-hari bersifat eksperimental
    - Selalu konsultasikan dengan ahli lingkungan untuk keputusan penting
    """)

else:
    st.error("Tidak dapat memuat model. Pastikan file model tersedia di direktori yang benar.")
    st.info("""
    **File yang diperlukan:**
    - `knn_model.pkl` - Model KNN yang telah dilatih
    - `minmax_scaler.pkl` - Scaler untuk normalisasi data
    
    Jalankan notebook training terlebih dahulu untuk menghasilkan file-file ini.
    """)