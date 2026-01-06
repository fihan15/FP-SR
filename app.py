import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# CONFIG & THEME
# ===============================
st.set_page_config(
    page_title="Sistem Rekomendasi Wisata Yogya",
    page_icon="🏖️",
    layout="wide"
)

# Custom CSS untuk tampilan card
st.markdown("""
    <style>
    .stMetric { 
        background-color: rgba(151, 151, 151, 0.1); 
        padding: 15px; 
        border-radius: 10px; 
    }
    .rec-card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        /* Menggunakan transparansi agar menyesuaikan background utama */
        background-color: rgba(151, 151, 151, 0.1); 
        border-left: 8px solid #4CAF50;
        border-right: 1px solid rgba(151, 151, 151, 0.2);
        border-top: 1px solid rgba(151, 151, 151, 0.2);
        border-bottom: 1px solid rgba(151, 151, 151, 0.2);
    }
    /* Memastikan teks judul tetap kontras */
    .rec-card h3 {
        margin: 0; 
        color: #4CAF50 !important;
    }
    /* Memastikan teks tabel/deskripsi mengikuti warna tema */
    .rec-card p, .rec-card td {
        color: inherit;
    }
    </style>
    """, unsafe_allow_html=True)

# ===============================
# LOAD DATA (DENGAN CACHING)
# ===============================
@st.cache_data
def load_all_data():
    tour_url = "https://raw.githubusercontent.com/AwalDinz/rekomendasi-wisata-yogya/main/dataset/tour.csv"
    rating_url = "https://raw.githubusercontent.com/AwalDinz/rekomendasi-wisata-yogya/main/dataset/tour_rating.csv"
    
    tour_df = pd.read_csv(tour_url)
    rating_df = pd.read_csv(rating_url)
    return tour_df, rating_df

tour, rating = load_all_data()

# ===============================
# PREPROCESSING & SIMILARITY
# ===============================
@st.cache_resource
def get_similarity_matrix(df_rating):
    # Buat matrix User-Item
    matrix = df_rating.pivot_table(index='User_Id', columns='Place_Id', values='Place_Ratings').fillna(0)
    # Hitung Cosine Similarity
    sim = cosine_similarity(matrix)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
    return matrix, sim_df

user_item_matrix, user_similarity_df = get_similarity_matrix(rating)

# ===============================
# RECOMMENDATION LOGIC
# ===============================
def recommend_places(user_id, top_n=5):
    if user_id not in user_similarity_df.index:
        return pd.DataFrame()

    similar_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id)
    user_ratings = user_item_matrix.loc[user_id]
    unseen_places = user_ratings[user_ratings == 0].index

    scores = {}
    for place in unseen_places:
        num, den = 0, 0
        for sim_user, similarity in similar_users.items():
            rating_sim = user_item_matrix.loc[sim_user, place]
            if rating_sim > 0:
                num += similarity * rating_sim
                den += similarity
        if den > 0:
            scores[place] = num / den

    recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    result = pd.DataFrame(recommended, columns=["Place_Id", "Predicted_Rating"])
    return result.merge(tour, on="Place_Id")

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.title("⚙️ Panel Kontrol")
    
    tab_selection = st.radio("Pilih Menu:", ["📊 Dashboard Analisis", "🎯 Cari Rekomendasi"])
    
    st.divider()
    if tab_selection == "🎯 Cari Rekomendasi":
        target_user = st.selectbox("Pilih User ID", user_item_matrix.index)
        n_rec = st.slider("Jumlah Rekomendasi", 3, 10, 5)
    
    st.info("Sistem ini menggunakan algoritma **User-Based Collaborative Filtering**.")

# ===============================
# MAIN CONTENT
# ===============================

# --- TAB 1: DASHBOARD ANALISIS ---
if tab_selection == "📊 Dashboard Analisis":
    st.title("📊 Analisis Data Wisata Yogyakarta")
    
    # Row 1: Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Wisata", len(tour))
    m2.metric("Total User", len(user_item_matrix))
    m3.metric("Total Rating", len(rating))
    
    st.divider()
    
    # Row 2: Charts
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📍 Top 10 Wisata Terpopuler")
        top_10 = rating.merge(tour, on='Place_Id')['Place_Name'].value_counts().head(10)
        fig, ax = plt.subplots()
        sns.barplot(x=top_10.values, y=top_10.index, palette='magma', ax=ax)
        st.pyplot(fig)
        
    with c2:
        st.subheader("🏷️ Kategori Wisata")
        cat_data = tour['Category'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(cat_data, labels=cat_data.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        st.pyplot(fig)

    st.subheader("⭐ Distribusi Rating yang Diberikan Pengguna")
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.countplot(x='Place_Ratings', data=rating, palette='viridis')
    st.pyplot(fig)

# --- TAB 2: SISTEM REKOMENDASI ---
else:
    st.title("🎯 Rekomendasi Wisata Untuk Anda")
    st.write(f"Menganalisis kemiripan User **{target_user}** dengan pengguna lainnya...")

    if st.button("✨ Tampilkan Rekomendasi"):
        with st.spinner("Menghitung skor kecocokan..."):
            rekomendasi = recommend_places(target_user, n_rec)
            
            if not rekomendasi.empty:
                st.success(f"Berhasil menemukan {len(rekomendasi)} tempat yang mungkin Anda sukai!")
                
                for _, row in rekomendasi.iterrows():
                    st.markdown(f"""
                        <div class="rec-card">
                            <h3>{row['Place_Name']}</h3>
                            <p>📍 {row['City']} | 🏷️ <b>{row['Category']}</b></p>
                            <hr style="opacity: 0.2; margin: 10px 0;">
                            <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                                <span>💰 <b>Tiket:</b> Rp {int(row['Price']):,}</span>
                                <span>⭐ <b>Rating:</b> {row['Rating']}</span>
                                <span style="color: #4CAF50;">🔮 <b>Prediksi: {row['Predicted_Rating']:.2f}</b></span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Maaf, data tidak cukup untuk memberikan rekomendasi.")

st.markdown("---")
st.caption("Final Project - Universitas Amikom Yogyakarta")
