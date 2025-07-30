import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title="Klasifikasi Obat dengan Naive Bayes",
    page_icon="💊"
)

# Fungsi untuk memuat data dengan penanganan error
def load_data():
    """Memuat dataset drug200.csv dengan penanganan error"""
    try:
        if not os.path.exists("drug200.csv"):
            st.error("File 'drug200.csv' tidak ditemukan! Pastikan file berada di direktori yang sama dengan aplikasi ini.")
            st.stop()

        df = pd.read_csv("drug200.csv", header=0, names=["Age", "BP", "Cholesterol", "Na_to_K", "ClassDrug"])

        if df.empty:
            st.error("File 'drug200.csv' kosong atau tidak dapat dibaca dengan benar.")
            st.stop()

        return df

    except FileNotFoundError:
        st.error("File 'drug200.csv' tidak ditemukan! Pastikan file berada di direktori yang sama dengan aplikasi ini.")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("File 'drug200.csv' kosong atau format tidak valid.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {str(e)}")
        st.stop()

# Fungsi-fungsi Naive Bayes
def calculate_prior(data, alpha):
    """Menghitung probabilitas prior untuk setiap kelas"""
    class_counts = data["ClassDrug"].value_counts().to_dict()
    total = len(data)
    return {
        cls: count / total
        for cls, count in class_counts.items()
    }

def get_unique_values(data, field):
    """Mendapatkan nilai unik dari suatu kolom"""
    return list(set(data[field]))

def calculate_likelihood(data, field, value, target_class, alpha):
    """Menghitung likelihood dengan Laplace smoothing"""
    subset = data[data["ClassDrug"] == target_class]
    match = len(subset[subset[field] == value])
    feature_values = get_unique_values(data, field)
    return (match + alpha) / (len(subset) + alpha * len(feature_values))

def predict(data, age, bp, chol, na_to_k, alpha):
    """Melakukan prediksi menggunakan Naive Bayes"""
    priors = calculate_prior(data, alpha)
    probs = {}

    for cls in priors:
        p = priors[cls]
        p *= calculate_likelihood(data, "Age", age, cls, alpha)
        p *= calculate_likelihood(data, "BP", bp, cls, alpha)
        p *= calculate_likelihood(data, "Cholesterol", chol, cls, alpha)
        p *= calculate_likelihood(data, "Na_to_K", na_to_k, cls, alpha)
        probs[cls] = p

    # Normalisasi probabilitas
    total_prob = sum(probs.values())
    for cls in probs:
        probs[cls] = probs[cls] / total_prob if total_prob > 0 else 0

    return max(probs, key=probs.get), probs

def calculate_likelihood_table(data, feature, alpha):
    """Menghitung tabel likelihood untuk suatu fitur"""
    classes = data["ClassDrug"].unique()
    feature_values = get_unique_values(data, feature)

    likelihood_data = []
    for cls in classes:
        for value in feature_values:
            likelihood = calculate_likelihood(data, feature, value, cls, alpha)
            likelihood_data.append({
                'Nilai_Fitur': value,
                'Kelas': cls,
                'Likelihood': round(likelihood, 4)
            })

    # Pivot table untuk tampilan yang lebih baik
    likelihood_df = pd.DataFrame(likelihood_data)
    return likelihood_df.pivot(index='Nilai_Fitur', columns='Kelas', values='Likelihood')

# Memuat data
data = load_data()

# Tab untuk navigasi
tab1, tab2 = st.tabs(["Pelatihan & Analisis", "Prediksi"])

with tab1:
    st.title("Pelatihan & Analisis")
    st.markdown("### Ringkasan Dataset")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Data", len(data))
        st.metric("Jumlah Fitur", len(data.columns) - 1)

    with col2:
        st.metric("Jumlah Kelas", data["ClassDrug"].nunique())
        st.metric("Data Hilang", data.isnull().sum().sum())

    # Tampilkan dataset
    st.markdown("### Dataset")
    st.dataframe(data, use_container_width=True)

    # Parameter alpha
    alpha = st.slider("Laplace Smoothing (α)", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

    st.markdown("---")

    # Probabilitas Prior
    st.markdown("### Probabilitas Prior")
    priors = calculate_prior(data, alpha)
    prior_df = pd.DataFrame([
        {"Kelas": cls, "Jumlah": data["ClassDrug"].value_counts()[cls],
         "Probabilitas Prior": round(prob, 4)}
        for cls, prob in priors.items()
    ])
    st.dataframe(prior_df, use_container_width=True)

    # Grafik probabilitas prior
    fig_prior = px.bar(prior_df, x="Kelas", y="Probabilitas Prior",
                      title="Probabilitas Prior Berdasarkan Kelas",
                      color="Probabilitas Prior", color_continuous_scale="viridis")
    st.plotly_chart(fig_prior, use_container_width=True)

    st.markdown("---")

    # Analisis Likelihood
    st.markdown("### Analisis Likelihood")

    # Mapping nama fitur untuk tampilan
    feature_mapping = {
        "Age": "Age (Usia)",
        "BP": "BP (Tekanan Darah)",
        "Cholesterol": "Cholesterol",
        "Na_to_K": "(Na_to_K) Rasio Natrium-Kalium"
    }

    features = ["Age", "BP", "Cholesterol", "Na_to_K"]

    for feature in features:
        feature_name = feature_mapping[feature]
        st.markdown(f"#### {feature_name}")

        st.markdown("**Tabel Kontingensi**")
        likelihood_table = calculate_likelihood_table(data, feature, alpha)
        st.dataframe(likelihood_table, use_container_width=True)

        # Heatmap untuk likelihood
        fig_heatmap = px.imshow(likelihood_table.T,
                               title="Heatmap",
                               color_continuous_scale="RdYlBu_r",
                               aspect="auto")
        fig_heatmap.update_xaxes(title=f"Nilai {feature_name}")
        fig_heatmap.update_yaxes(title="Kelas Obat")
        st.plotly_chart(fig_heatmap, use_container_width=True)

with tab2:
    st.title("Prediksi")
    st.markdown("Prediksi jenis obat berdasarkan data kondisi pasien.")

    # Form input
    col1, col2 = st.columns(2)

    with col1:
        age = st.selectbox("Usia", ["YOUNG", "ADULT", "OLD"],
                          format_func=lambda x: {"YOUNG": "Muda", "ADULT": "Dewasa", "OLD": "Tua"}[x])
        bp = st.selectbox("Tekanan Darah", ["LOW", "NORMAL", "HIGH"],
                         format_func=lambda x: {"LOW": "Rendah", "NORMAL": "Normal", "HIGH": "Tinggi"}[x])

    with col2:
        chol = st.selectbox("Kolesterol", ["NORMAL", "HIGH"],
                           format_func=lambda x: {"NORMAL": "Normal", "HIGH": "Tinggi"}[x])
        na_to_k = st.selectbox("Rasio Natrium-Kalium", ["LOW", "NORMAL", "HIGH"],
                              format_func=lambda x: {"LOW": "Rendah", "NORMAL": "Normal", "HIGH": "Tinggi"}[x])

    predict_btn = st.button("Lakukan Prediksi", use_container_width=True, type="primary")

    if predict_btn:
        # Konversi kembali ke nilai asli untuk prediksi
        age_val = age
        bp_val = bp
        chol_val = chol
        na_to_k_val = na_to_k

        pred_class, probs = predict(data, age_val, bp_val, chol_val, na_to_k_val, alpha)

        st.markdown("---")
        st.markdown("### HASIL PREDIKSI")

        # Ringkasan input
        input_data = {
            "Fitur": ["Usia", "Tekanan Darah", "Kolesterol", "Rasio Natrium-Kalium"],
            "Nilai": [
                {"YOUNG": "Muda", "ADULT": "Dewasa", "OLD": "Tua"}[age],
                {"LOW": "Rendah", "NORMAL": "Normal", "HIGH": "Tinggi"}[bp],
                {"NORMAL": "Normal", "HIGH": "Tinggi"}[chol],
                {"LOW": "Rendah", "NORMAL": "Normal", "HIGH": "Tinggi"}[na_to_k]
            ]
        }
        input_df = pd.DataFrame(input_data)
        st.dataframe(input_df, use_container_width=True)

        # Hasil probabilitas
        prob_df = pd.DataFrame([
            {"Kelas": cls, "Probabilitas": round(prob, 4), "Persentase (%)": round(prob * 100, 2)}
            for cls, prob in probs.items()
        ]).sort_values("Probabilitas", ascending=False)

        st.markdown("#### Probabilitas Setiap Kelas")
        st.dataframe(prob_df, use_container_width=True)

        # Visualisasi diagram lingkaran
        fig_pie = px.pie(prob_df, names="Kelas", values="Persentase (%)",
                        title="Distribusi Probabilitas Prediksi")
        st.plotly_chart(fig_pie, use_container_width=True)

        # Prediksi akhir
        st.markdown("### Hasil Akhir")
        max_prob = prob_df.iloc[0]["Persentase (%)"]
        confidence_level = "tinggi" if max_prob > 70 else "sedang" if max_prob > 50 else "rendah"

        st.success(f"**Jenis obat yang direkomendasikan: {pred_class}**")
        st.info(f"Tingkat keyakinan: {max_prob}% ({confidence_level})")

        if max_prob < 60:
            st.warning("Tingkat keyakinan relatif rendah. Disarankan untuk konsultasi lebih lanjut dengan tenaga medis.")
