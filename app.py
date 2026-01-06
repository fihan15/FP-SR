import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Crypto Market Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================
# STYLE
# =====================
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1, h2, h3, h4 { color: white; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #1e88e5; }
</style>
""", unsafe_allow_html=True)

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    df = pd.read_csv("crypto_top1000_dataset.csv")
    return df

df = load_data()

# =====================
# COMPREHENSIVE DATA PROCESSING
# =====================

# 1. EXPLORATORY DATA ANALYSIS SECTION
st.sidebar.title("🔧 Data Processing Settings")
show_data_stats = st.sidebar.checkbox("Show Data Processing Stats", False)

# 2. MISSING VALUE HANDLING - MULTI STRATEGY
def handle_missing_values(df):
    """Handle missing values dengan strategi berbeda per kolom"""
    
    original_shape = df.shape
    missing_before = df.isnull().sum().sum()
    
    # Strategi 1: Drop rows dengan missing value kritis
    critical_cols = [
        "name", "symbol", "current_price", 
        "market_cap", "market_cap_rank"
    ]
    critical_cols = [col for col in critical_cols if col in df.columns]
    df = df.dropna(subset=critical_cols)
    
    # Strategi 2: Imputation dengan median untuk numerik
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_to_impute = [
        col for col in numeric_cols 
        if col not in critical_cols and df[col].isnull().sum() > 0
    ]
    
    if numeric_cols_to_impute:
        imputer = SimpleImputer(strategy='median')
        df[numeric_cols_to_impute] = imputer.fit_transform(df[numeric_cols_to_impute])
    
    # Strategi 3: Fill dengan nilai spesifik
    fill_dict = {
        'total_supply': df['circulating_supply'] if 'circulating_supply' in df.columns else np.nan,
        'max_supply': -1,  # -1 untuk menandai unlimited supply
        'fully_diluted_valuation': 0,
    }
    
    for col, fill_value in fill_dict.items():
        if col in df.columns:
            if callable(fill_value):
                df[col] = df[col].fillna(fill_value)
            else:
                df[col] = df[col].fillna(fill_value)
    
    # Strategi 4: Flag untuk missing values penting
    df["has_max_supply"] = df["max_supply"].notna().astype(int)
    df["has_1y_history"] = df["price_change_percentage_1y"].notna().astype(int)
    df["has_ath"] = df["ath"].notna().astype(int)
    
    missing_after = df.isnull().sum().sum()
    
    if show_data_stats:
        st.sidebar.info(f"""
        **Missing Value Stats:**
        - Sebelum: {missing_before} missing values
        - Setelah: {missing_after} missing values
        - % Reduced: {((missing_before-missing_after)/missing_before*100 if missing_before>0 else 0):.1f}%
        - Rows: {original_shape[0]} → {df.shape[0]}
        """)
    
    return df

# 3. OUTLIER DETECTION & HANDLING
def handle_outliers(df):
    """Deteksi dan handle outliers dengan multiple methods"""
    
    outlier_cols = st.sidebar.multiselect(
        "Select columns for outlier handling:",
        options=df.select_dtypes(include=[np.number]).columns.tolist(),
        default=["market_cap", "total_volume", "current_price", "price_change_percentage_24h"]
    )
    
    method = st.sidebar.selectbox(
        "Outlier handling method:",
        ["IQR Capping", "Z-Score", "Percentile", "None"]
    )
    
    if method == "None" or not outlier_cols:
        return df
    
    df_outliers = df.copy()
    outlier_stats = {}
    
    for col in outlier_cols:
        if col in df_outliers.columns:
            original_data = df_outliers[col].copy()
            
            if method == "IQR Capping":
                # IQR Method
                Q1 = df_outliers[col].quantile(0.25)
                Q3 = df_outliers[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identifikasi outliers
                outliers = df_outliers[(df_outliers[col] < lower_bound) | (df_outliers[col] > upper_bound)]
                outlier_count = len(outliers)
                
                # Cap outliers
                df_outliers[col] = df_outliers[col].clip(lower_bound, upper_bound)
                
            elif method == "Z-Score":
                # Z-Score Method
                mean = df_outliers[col].mean()
                std = df_outliers[col].std()
                z_scores = np.abs((df_outliers[col] - mean) / std)
                
                # Cap pada 3 sigma
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                outlier_count = (z_scores > 3).sum()
                
                df_outliers[col] = df_outliers[col].clip(lower_bound, upper_bound)
                
            elif method == "Percentile":
                # Percentile Method
                lower_bound = df_outliers[col].quantile(0.01)
                upper_bound = df_outliers[col].quantile(0.99)
                outlier_count = len(df_outliers[(df_outliers[col] < lower_bound) | (df_outliers[col] > upper_bound)])
                
                df_outliers[col] = df_outliers[col].clip(lower_bound, upper_bound)
            
            # Hitung perubahan
            change_pct = ((df_outliers[col] - original_data).abs().sum() / original_data.abs().sum() * 100 
                         if original_data.abs().sum() > 0 else 0)
            
            outlier_stats[col] = {
                'outliers': outlier_count,
                'change_pct': change_pct,
                'method': method
            }
    
    if show_data_stats and outlier_stats:
        st.sidebar.info(f"""
        **Outlier Handling ({method}):**
        {''.join([f"- {col}: {stats['outliers']} outliers ({stats['change_pct']:.1f}% change)\\n" 
                 for col, stats in outlier_stats.items()])}
        """)
    
    return df_outliers

# 4. FEATURE SCALING
def scale_features(df):
    """Feature scaling dengan multiple methods"""
    
    scale_method = st.sidebar.selectbox(
        "Scaling method:",
        ["RobustScaler", "MinMaxScaler", "StandardScaler", "Log Transform", "None"]
    )
    
    scale_cols = st.sidebar.multiselect(
        "Select columns to scale:",
        options=df.select_dtypes(include=[np.number]).columns.tolist(),
        default=["market_cap", "total_volume", "current_price"]
    )
    
    if scale_method == "None" or not scale_cols:
        return df
    
    df_scaled = df.copy()
    
    if scale_method == "RobustScaler":
        scaler = RobustScaler()
        df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
        
    elif scale_method == "MinMaxScaler":
        scaler = MinMaxScaler()
        df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
        
    elif scale_method == "Log Transform":
        # Log transform untuk data dengan skew tinggi
        for col in scale_cols:
            if df_scaled[col].min() > 0:  # Log hanya untuk positif
                df_scaled[col] = np.log1p(df_scaled[col])
    
    if show_data_stats:
        st.sidebar.info(f"""
        **Scaling Applied:**
        - Method: {scale_method}
        - Columns: {', '.join(scale_cols[:3])}{'...' if len(scale_cols) > 3 else ''}
        """)
    
    return df_scaled

# 5. FEATURE ENGINEERING
def create_features(df):
    """Create new features untuk analisis"""
    
    # Volatility features
    df["volatility_24h"] = (df["high_24h"] - df["low_24h"]) / df["current_price"]
    df["price_range_percentage"] = ((df["high_24h"] - df["low_24h"]) / df["low_24h"]) * 100
    
    # Volume analysis
    df["volume_marketcap_ratio"] = df["total_volume"] / df["market_cap"]
    df["volume_price_ratio"] = df["total_volume"] / df["current_price"]
    
    # Supply analysis
    if 'circulating_supply' in df.columns and 'max_supply' in df.columns:
        df["supply_utilization"] = df["circulating_supply"] / df["max_supply"].replace(-1, np.nan)
        df["supply_inflation_risk"] = 1 - df["supply_utilization"].fillna(0)
    
    # Valuation ratios
    if 'fully_diluted_valuation' in df.columns:
        df["fdv_mc_ratio"] = df["fully_diluted_valuation"] / df["market_cap"]
        df["fdv_mc_ratio"] = df["fdv_mc_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Performance metrics
    df["performance_score"] = (
        df["price_change_percentage_24h"] * 0.4 +
        df["price_change_percentage_7d"] * 0.3 +
        df["price_change_percentage_30d"] * 0.3
    ) if all(col in df.columns for col in ["price_change_percentage_24h", "price_change_percentage_7d", "price_change_percentage_30d"]) else 0
    
    # Risk score
    df["risk_score"] = (
        df["volatility_24h"].rank(pct=True) * 0.5 +
        (df["price_change_percentage_24h"] < 0).astype(int) * 0.3 +
        df["supply_inflation_risk"].fillna(0).rank(pct=True) * 0.2
    )
    
    # Liquidity score
    if 'total_volume' in df.columns and 'market_cap' in df.columns:
        df["liquidity_score"] = (df["total_volume"] / df["market_cap"]).rank(pct=True)
    
    return df

# =====================
# APPLY DATA PROCESSING PIPELINE
# =====================
processing_steps = st.sidebar.multiselect(
    "Data Processing Steps:",
    ["Missing Value Handling", "Outlier Handling", "Feature Scaling", "Feature Engineering"],
    default=["Missing Value Handling", "Outlier Handling", "Feature Engineering"]
)

# Pipeline eksekusi
df_processed = df.copy()

if "Missing Value Handling" in processing_steps:
    df_processed = handle_missing_values(df_processed)

if "Outlier Handling" in processing_steps:
    df_processed = handle_outliers(df_processed)

if "Feature Scaling" in processing_steps:
    df_processed = scale_features(df_processed)

if "Feature Engineering" in processing_steps:
    df_processed = create_features(df_processed)

# =====================
# DISPLAY PROCESSING SUMMARY
# =====================
if show_data_stats:
    with st.expander("📊 Data Processing Summary", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Original Rows", df.shape[0])
            st.metric("Processed Rows", df_processed.shape[0])
        
        with col2:
            missing_before = df.isnull().sum().sum()
            missing_after = df_processed.isnull().sum().sum()
            st.metric("Missing Values", f"{missing_before} → {missing_after}")
        
        with col3:
            num_cols_before = len(df.select_dtypes(include=[np.number]).columns)
            num_cols_after = len(df_processed.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", f"{num_cols_before} → {num_cols_after}")
        
        with col4:
            new_features = len([col for col in df_processed.columns if col not in df.columns])
            st.metric("New Features", new_features)
        
        # Show data sample
        st.write("**Sample of Processed Data:**")
        st.dataframe(df_processed.head(5))

# =====================
# HELPER FUNCTIONS
# =====================
def categorize(rank):
    if rank <= 10: return "Big Cap"
    if rank <= 50: return "Mid Cap"
    return "Small Cap"

# =====================
# MAIN DASHBOARD (SAMA SEBELUMNYA)
# =====================
# [Bagian utama dashboard sama seperti sebelumnya...]
# ... (sisanya sama dengan script sebelumnya, tapi gunakan df_processed bukan df)

# Untuk melanjutkan, bagian filter, sidebar, dan visualisasi akan menggunakan df_processed
df_filtered = df_processed.copy()

# Apply filters, categories, dan visualisasi sama seperti sebelumnya...
# ... (copy semua kode visualisasi dari script sebelumnya, ganti df dengan df_processed)
