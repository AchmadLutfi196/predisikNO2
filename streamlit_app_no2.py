import streamlit as st
import numpy as np
import pandas as pd
try:
    import joblib
except ImportError:
    st.error("❌ Modul 'joblib' tidak ditemukan!")
    st.info("📋 Pastikan requirements.txt berisi 'joblib==1.3.2' dan redeploy aplikasi.")
    st.stop()
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

def predict_no2_multi_day(model, scaler, no2_t2, no2_t1, days=3):
    """Prediksi NO2 untuk beberapa hari ke depan"""
    try:
        predictions = []
        current_t2 = no2_t2
        current_t1 = no2_t1
        
        for day in range(days):
            # Prediksi untuk hari ini
            pred_mol, pred_ugm3 = predict_no2(model, scaler, current_t2, current_t1)
            
            if pred_mol is None:
                return None
                
            # Evaluasi status WHO
            status, color, icon = evaluate_who_status(pred_ugm3)
            
            # Simpan hasil
            predictions.append({
                'day': day + 1,
                'date': datetime.now() + timedelta(days=day+1),
                'no2_mol': pred_mol,
                'no2_ugm3': pred_ugm3,
                'status': status,
                'color': color,
                'icon': icon,
                'input_t2': current_t2,
                'input_t1': current_t1
            })
            
            # Update input untuk prediksi hari berikutnya
            current_t2 = current_t1
            current_t1 = pred_mol
        
        return predictions
    except Exception as e:
        st.error(f"Error dalam prediksi multi-hari: {str(e)}")
        return None

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
    st.header("🔮 Sistem Prediksi NO2")
    
    # Tab untuk pilihan prediksi
    tab1, tab2 = st.tabs(["📅 Prediksi 1 Hari", "📊 Prediksi 3 Hari"])
    
    with tab1:
        st.subheader("Prediksi NO2 untuk 1 Hari ke Depan")
        
        # Pilihan input method
        input_method = st.radio(
            "Pilih metode input:",
            ["Manual Input", "Skenario Pre-defined"],
            key="single_day"
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
                    help="Konsentrasi NO2 dua hari yang lalu",
                    key="single_t2"
                )
            
            with col2:
                no2_t1 = st.number_input(
                    "NO2 (t-1) dalam mol/m²",
                    min_value=0.0,
                    max_value=0.01,
                    value=0.00012,
                    step=0.000001,
                    format="%.6f",
                    help="Konsentrasi NO2 satu hari yang lalu",
                    key="single_t1"
                )
        
        else:
            st.subheader("Skenario Pre-defined")
            scenario_option = st.selectbox(
                "Pilih skenario:",
                [
                    "Skenario 1 - Rendah (Sangat Baik)",
                    "Skenario 2 - Sedang (Perhatian)", 
                    "Skenario 3 - Tinggi (Berbahaya)"
                ],
                key="single_scenario"
            )
            
            # Skenario data
            scenarios = {
                "Skenario 1 - Rendah (Sangat Baik)": [0.00010, 0.00012],
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
        if st.button("Prediksi NO2", type="primary", key="single_predict"):
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

    with tab2:
        st.subheader("Prediksi NO2 untuk 3 Hari ke Depan")
        st.info("🚀 Fitur prediksi multi-hari menggunakan teknik sequential prediction dimana hasil prediksi hari sebelumnya menjadi input untuk prediksi hari berikutnya.")
        
        # Pilihan input method untuk 3 hari
        input_method_3d = st.radio(
            "Pilih metode input:",
            ["Manual Input", "Skenario Pre-defined"],
            key="multi_day"
        )
        
        if input_method_3d == "Manual Input":
            col1, col2 = st.columns(2)
            
            with col1:
                no2_t2_3d = st.number_input(
                    "NO2 (t-2) dalam mol/m²",
                    min_value=0.0,
                    max_value=0.01,
                    value=0.0001,
                    step=0.000001,
                    format="%.6f",
                    help="Konsentrasi NO2 dua hari yang lalu",
                    key="multi_t2"
                )
            
            with col2:
                no2_t1_3d = st.number_input(
                    "NO2 (t-1) dalam mol/m²",
                    min_value=0.0,
                    max_value=0.01,
                    value=0.00012,
                    step=0.000001,
                    format="%.6f",
                    help="Konsentrasi NO2 satu hari yang lalu",
                    key="multi_t1"
                )
        
        else:
            st.subheader("Skenario Pre-defined")
            scenario_option_3d = st.selectbox(
                "Pilih skenario:",
                [
                    "Skenario 1 - Rendah (Sangat Baik)",
                    "Skenario 2 - Sedang (Perhatian)", 
                    "Skenario 3 - Tinggi (Berbahaya)"
                ],
                key="multi_scenario"
            )
            
            # Skenario data
            scenarios_3d = {
                "Skenario 1 - Rendah (Sangat Baik)": [0.00010, 0.00012],
                "Skenario 2 - Sedang (Perhatian)": [0.00035, 0.00040],
                "Skenario 3 - Tinggi (Berbahaya)": [0.00080, 0.00090]
            }
            
            no2_t2_3d, no2_t1_3d = scenarios_3d[scenario_option_3d]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("NO2 (t-2)", f"{no2_t2_3d:.6f} mol/m²")
            with col2:
                st.metric("NO2 (t-1)", f"{no2_t1_3d:.6f} mol/m²")
        
        # Tombol prediksi 3 hari
        if st.button("Prediksi 3 Hari ke Depan", type="primary", key="multi_predict"):
            with st.spinner("Melakukan prediksi untuk 3 hari ke depan..."):
                # Prediksi multi-hari
                predictions = predict_no2_multi_day(model, scaler, no2_t2_3d, no2_t1_3d, days=3)
                
                if predictions is not None:
                    # Display hasil dalam tabel
                    st.subheader("📊 Hasil Prediksi 3 Hari")
                    
                    # Buat dataframe untuk display
                    df_results = []
                    for pred in predictions:
                        df_results.append({
                            'Hari': f"Hari {pred['day']} ({pred['date'].strftime('%d-%m-%Y')})",
                            'NO2 (mol/m²)': f"{pred['no2_mol']:.6f}",
                            'NO2 (µg/m³)': f"{pred['no2_ugm3']:.2f}",
                            'Status WHO': f"{pred['icon']} {pred['status']}",
                            'Warna Status': pred['color']
                        })
                    
                    df_display = pd.DataFrame(df_results)
                    st.dataframe(df_display.drop('Warna Status', axis=1), use_container_width=True)
                    
                    # Visualisasi timeline lengkap
                    st.subheader("📈 Visualisasi Timeline 5 Hari")
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Siapkan data untuk plot
                    all_days = ['t-2', 't-1'] + [f'Prediksi\nHari {i+1}' for i in range(3)]
                    all_values_mol = [no2_t2_3d, no2_t1_3d] + [pred['no2_mol'] for pred in predictions]
                    all_values_ugm3 = [v * CONVERSION_FACTOR for v in [no2_t2_3d, no2_t1_3d]] + [pred['no2_ugm3'] for pred in predictions]
                    all_colors = ['lightgray', 'lightgray'] + [pred['color'] for pred in predictions]
                    
                    # Plot 1: Timeline mol/m²
                    bars1 = ax1.bar(all_days, all_values_mol, color=all_colors, alpha=0.7, edgecolor='black')
                    ax1.set_ylabel('NO2 (mol/m²)')
                    ax1.set_title('Timeline Lengkap NO2 (5 Hari)')
                    ax1.grid(True, alpha=0.3)
                    ax1.tick_params(axis='x', rotation=45)
                    
                    for bar, val in zip(bars1, all_values_mol):
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                f'{val:.6f}', ha='center', va='bottom', fontsize=8)
                    
                    # Plot 2: Evaluasi WHO µg/m³
                    bars2 = ax2.bar(all_days, all_values_ugm3, color=all_colors, alpha=0.7, edgecolor='black')
                    ax2.axhline(WHO_ANNUAL, color='green', linestyle='--', linewidth=2, 
                               label=f'WHO Annual ({WHO_ANNUAL} µg/m³)')
                    ax2.axhline(WHO_24HOUR, color='red', linestyle='--', linewidth=2,
                               label=f'WHO 24-Hour ({WHO_24HOUR} µg/m³)')
                    
                    ax2.set_ylabel('NO2 (µg/m³)')
                    ax2.set_title('Evaluasi WHO untuk Timeline Lengkap')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    ax2.tick_params(axis='x', rotation=45)
                    
                    for bar, val in zip(bars2, all_values_ugm3):
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Analisis tren
                    st.subheader("📋 Analisis Tren 3 Hari")
                    
                    # Hitung tren
                    pred_values = [pred['no2_ugm3'] for pred in predictions]
                    
                    if pred_values[2] > pred_values[0]:
                        trend = "📈 **MENINGKAT**"
                        trend_color = "red"
                        trend_desc = "Konsentrasi NO2 cenderung meningkat selama 3 hari ke depan."
                    elif pred_values[2] < pred_values[0]:
                        trend = "📉 **MENURUN**"
                        trend_color = "green"
                        trend_desc = "Konsentrasi NO2 cenderung menurun selama 3 hari ke depan."
                    else:
                        trend = "➡️ **STABIL**"
                        trend_color = "blue"
                        trend_desc = "Konsentrasi NO2 relatif stabil selama 3 hari ke depan."
                    
                    # Hitung rata-rata dan variasi
                    avg_pred = np.mean(pred_values)
                    max_pred = max(pred_values)
                    min_pred = min(pred_values)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Tren", trend.split('**')[1].strip())
                    
                    with col2:
                        st.metric("Rata-rata", f"{avg_pred:.2f} µg/m³")
                    
                    with col3:
                        st.metric("Maksimum", f"{max_pred:.2f} µg/m³")
                    
                    with col4:
                        st.metric("Minimum", f"{min_pred:.2f} µg/m³")
                    
                    # Status keseluruhan
                    dangerous_days = sum(1 for pred in predictions if pred['status'] == 'BERBAHAYA')
                    attention_days = sum(1 for pred in predictions if pred['status'] == 'PERHATIAN')
                    good_days = sum(1 for pred in predictions if pred['status'] == 'SANGAT BAIK')
                    
                    st.subheader("🏥 Ringkasan Kesehatan")
                    
                    if dangerous_days > 0:
                        st.error(f"""
                        ❌ **PERHATIAN TINGGI DIPERLUKAN**
                        
                        Terdapat {dangerous_days} hari dengan kondisi berbahaya dalam 3 hari ke depan.
                        Konsentrasi NO2 melebihi ambang batas WHO 24-jam ({WHO_24HOUR} µg/m³).
                        
                        **Rekomendasi:**
                        - Hindari aktivitas outdoor intensif
                        - Gunakan masker saat keluar rumah
                        - Tingkatkan ventilasi indoor
                        - Pantau kondisi kesehatan kelompok rentan
                        """)
                    elif attention_days > 0:
                        st.warning(f"""
                        ⚠️ **MONITORING DIPERLUKAN**
                        
                        Terdapat {attention_days} hari dengan kondisi perhatian dalam 3 hari ke depan.
                        Konsentrasi NO2 di antara batas WHO tahunan dan 24-jam.
                        
                        **Rekomendasi:**
                        - Pantau kualitas udara secara berkala
                        - Kurangi aktivitas outdoor yang berat
                        - Perhatikan kelompok rentan (anak-anak, lansia)
                        """)
                    else:
                        st.success(f"""
                        ✅ **KONDISI BAIK**
                        
                        Semua 3 hari menunjukkan kondisi yang sangat baik.
                        Konsentrasi NO2 berada di bawah ambang batas WHO tahunan ({WHO_ANNUAL} µg/m³).
                        
                        **Status:** Aman untuk semua aktivitas outdoor normal.
                        """)
                    
                    # Tren detail
                    st.info(f"""
                    **Analisis Tren:** {trend_desc}
                    
                    **Variasi:** {max_pred - min_pred:.2f} µg/m³ (selisih max-min)
                    
                    **Catatan:** Prediksi ini menggunakan metode sequential prediction dan bersifat eksperimental.
                    Akurasi menurun untuk prediksi yang lebih jauh ke depan.
                    """)

    # Footer informasi
    st.markdown("---")
    st.markdown("""
    **Tentang Aplikasi:**
    
    Aplikasi ini menggunakan model K-Nearest Neighbors (KNN) untuk memprediksi konsentrasi NO2 
    berdasarkan data historis 2 hari sebelumnya. Model telah dilatih dengan data time series 
    dan dievaluasi menggunakan standar kualitas udara WHO.
    
    **Fitur Baru - Prediksi 3 Hari:**
    - Menggunakan teknik sequential prediction
    - Hasil prediksi hari pertama menjadi input untuk hari kedua
    - Analisis tren dan evaluasi kesehatan komprehensif
    
    **Catatan Penting:**
    - Model ini untuk tujuan akademis/penelitian
    - Prediksi multi-hari bersifat eksperimental dan akurasi menurun untuk jangka waktu lebih jauh
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