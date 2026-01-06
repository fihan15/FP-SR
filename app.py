import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# =====================
# PAGE CONFIG & THEME
# =====================
st.set_page_config(
    page_title="📊 Crypto Market Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",  # Sidebar collapsed untuk lebih fokus
    page_icon="📈"
)

# =====================
# CUSTOM STYLING - PROFESSIONAL & CLEAN
# =====================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Title styling */
    .main-title {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem !important;
        padding: 20px 0;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.2rem !important;
        color: #6c757d !important;
        text-align: center;
        margin-bottom: 2rem !important;
        font-weight: 300;
    }
    
    /* Card styling for metrics */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
    
    /* Chart container styling */
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin-bottom: 25px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #2c3e50 !important;
        margin-top: 30px !important;
        margin-bottom: 20px !important;
        padding-bottom: 10px;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: white;
        border-radius: 10px 10px 0 0;
        padding: 10px 25px;
        font-weight: 500;
        color: #6c757d;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: #667eea !important;
        color: white !important;
        border-color: #667eea !important;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom info boxes */
    .info-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #28a745;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #ffc107;
    }
    
    /* Animation for highlights */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .highlight {
        animation: pulse 2s infinite;
        display: inline-block;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# =====================
# LOAD DATA WITH SAMPLE FOR PRESENTATION
# =====================
@st.cache_data
def load_sample_data():
    """Load data dengan sample untuk presentasi"""
    try:
        df = pd.read_csv("crypto_top1000_dataset.csv")
    except:
        # Create sample data for presentation if file not found
        st.warning("⚠️ Using sample data for presentation")
        np.random.seed(42)
        n = 200
        data = {
            'name': [f'Crypto{i}' for i in range(n)],
            'symbol': [f'CRYPTO{i:03d}' for i in range(n)],
            'market_cap': np.random.exponential(10000, n) * 1e6,
            'current_price': np.random.lognormal(0, 2, n),
            'price_change_percentage_24h': np.random.normal(0, 15, n),
            'price_change_percentage_7d': np.random.normal(5, 20, n),
            'total_volume': np.random.exponential(1000, n) * 1e6,
            'market_cap_rank': range(1, n+1),
            'high_24h': np.random.uniform(1.1, 2, n),
            'low_24h': np.random.uniform(0.8, 0.95, n),
            'circulating_supply': np.random.exponential(1e6, n),
            'max_supply': np.random.exponential(2e6, n),
        }
        df = pd.DataFrame(data)
        df['volatility_24h'] = (df['high_24h'] - df['low_24h']) / df['current_price']
        df['volume_marketcap_ratio'] = df['total_volume'] / df['market_cap']
    
    return df

df = load_sample_data()

# =====================
# SIMPLE DATA PROCESSING FOR PRESENTATION
# =====================
def process_data_for_presentation(df):
    """Simple data processing yang mudah dipahami"""
    
    # 1. Handle missing values secara sederhana
    df_clean = df.copy()
    
    # Isi missing values dengan median untuk numerik
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # 2. Simple outlier handling (cap extreme values)
    for col in ['price_change_percentage_24h', 'price_change_percentage_7d']:
        if col in df_clean.columns:
            q1 = df_clean[col].quantile(0.05)
            q3 = df_clean[col].quantile(0.95)
            df_clean[col] = df_clean[col].clip(q1, q3)
    
    # 3. Buat kategori sederhana
    df_clean['market_cap_category'] = pd.qcut(
        df_clean['market_cap'], 
        q=[0, 0.1, 0.5, 1], 
        labels=['Small Cap', 'Mid Cap', 'Large Cap']
    )
    
    # 4. Buat score sederhana untuk presentasi
    if all(col in df_clean.columns for col in ['price_change_percentage_24h', 'volatility_24h']):
        df_clean['performance_score'] = (
            df_clean['price_change_percentage_24h'].rank(pct=True) * 0.7 -
            df_clean['volatility_24h'].rank(pct=True) * 0.3
        )
    
    # 5. Format currency untuk display
    df_clean['market_cap_formatted'] = df_clean['market_cap'].apply(
        lambda x: f'${x/1e9:.1f}B' if x >= 1e9 else f'${x/1e6:.0f}M'
    )
    
    return df_clean

df_processed = process_data_for_presentation(df)

# =====================
# HERO SECTION - TITLE & KEY METRICS
# =====================
st.markdown('<h1 class="main-title">📈 Crypto Market Intelligence Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time Analysis & Insights for Informed Decision Making</p>', unsafe_allow_html=True)

# Create a top metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_market_cap = df_processed['market_cap'].sum()
        st.metric(
            label="💰 **Total Market Cap**",
            value=f"${total_market_cap/1e12:.1f}T",
            delta="+2.3%"
        )
        st.caption("Total cryptocurrency market valuation")
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_change = df_processed['price_change_percentage_24h'].mean()
        st.metric(
            label="📊 **24h Avg Change**",
            value=f"{avg_change:+.1f}%",
            delta_color="inverse" if avg_change < 0 else "normal"
        )
        st.caption("Average daily price movement")
        st.markdown('</div>', unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        gainers = (df_processed['price_change_percentage_24h'] > 0).sum()
        total_coins = len(df_processed)
        st.metric(
            label="📈 **Market Sentiment**",
            value=f"{(gainers/total_coins*100):.0f}%",
            delta="Bullish" if gainers/total_coins > 0.5 else "Bearish"
        )
        st.caption(f"{gainers}/{total_coins} coins positive")
        st.markdown('</div>', unsafe_allow_html=True)

with col4:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        top_10_dominance = df_processed.nsmallest(10, 'market_cap_rank')['market_cap'].sum() / total_market_cap * 100
        st.metric(
            label="👑 **Top 10 Dominance**",
            value=f"{top_10_dominance:.1f}%",
            delta="-1.2%"
        )
        st.caption("Market share of top 10 cryptocurrencies")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# =====================
# MARKET OVERVIEW SECTION
# =====================
st.markdown('<h2 class="section-header">📊 Market Overview</h2>', unsafe_allow_html=True)

# Row 1: Market Distribution & Performance
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 🗺️ Market Cap Distribution")
    
    # Treemap dengan warna yang lebih menarik
    fig = px.treemap(
        df_processed.head(30),
        path=['market_cap_category', 'symbol'],
        values='market_cap',
        color='price_change_percentage_24h',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        hover_data={'market_cap': ':.2s', 'current_price': ':.2f'},
        title=""
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=0, b=0, l=0, r=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Market Health Indicators")
    
    # Buat gauge charts yang menarik
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}]],
        vertical_spacing=0.2,
        horizontal_spacing=0.2
    )
    
    # Gauge 1: Market Sentiment
    sentiment = (df_processed['price_change_percentage_24h'] > 0).mean() * 100
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=sentiment,
        domain={'x': [0, 0.45], 'y': [0.5, 1]},
        title={'text': "Bullish %"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#28a745"},
               'steps': [
                   {'range': [0, 30], 'color': "#dc3545"},
                   {'range': [30, 70], 'color': "#ffc107"},
                   {'range': [70, 100], 'color': "#28a745"}
               ]}
    ), row=1, col=1)
    
    # Gauge 2: Volatility
    avg_vol = df_processed['volatility_24h'].mean() * 100 if 'volatility_24h' in df_processed.columns else 5
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=avg_vol,
        domain={'x': [0.55, 1], 'y': [0.5, 1]},
        title={'text': "Avg Volatility %"},
        gauge={'axis': {'range': [0, 20]},
               'bar': {'color': "#dc3545"},
               'steps': [
                   {'range': [0, 5], 'color': "#28a745"},
                   {'range': [5, 15], 'color': "#ffc107"},
                   {'range': [15, 20], 'color': "#dc3545"}
               ]}
    ), row=1, col=2)
    
    # Gauge 3: Liquidity
    avg_volume_ratio = df_processed['volume_marketcap_ratio'].mean() * 100 if 'volume_marketcap_ratio' in df_processed.columns else 3
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=avg_volume_ratio,
        domain={'x': [0, 0.45], 'y': [0, 0.5]},
        title={'text': "Liquidity %"},
        gauge={'axis': {'range': [0, 10]},
               'bar': {'color': "#007bff"},
               'steps': [
                   {'range': [0, 3], 'color': "#dc3545"},
                   {'range': [3, 7], 'color': "#ffc107"},
                   {'range': [7, 10], 'color': "#28a745"}
               ]}
    ), row=2, col=1)
    
    # Gauge 4: Concentration Risk
    concentration = top_10_dominance
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=concentration,
        domain={'x': [0.55, 1], 'y': [0, 0.5]},
        title={'text': "Top 10 Share %"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#6f42c1"},
               'steps': [
                   {'range': [0, 40], 'color': "#28a745"},
                   {'range': [40, 70], 'color': "#ffc107"},
                   {'range': [70, 100], 'color': "#dc3545"}
               ]}
    ), row=2, col=2)
    
    fig.update_layout(
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Row 2: Top Performers & Distribution
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.markdown("### 🏆 Top Performers Analysis")

col1, col2 = st.columns(2)

with col1:
    # Top 5 Gainers
    top_gainers = df_processed.nlargest(5, 'price_change_percentage_24h')
    fig = px.bar(
        top_gainers,
        x='symbol',
        y='price_change_percentage_24h',
        color='price_change_percentage_24h',
        color_continuous_scale='Greens',
        title="🚀 Top 5 Gainers (24h)",
        text_auto='+.1f'
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        yaxis_title="Price Change %",
        xaxis_title=""
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Top 5 Losers
    top_losers = df_processed.nsmallest(5, 'price_change_percentage_24h')
    fig = px.bar(
        top_losers,
        x='symbol',
        y='price_change_percentage_24h',
        color='price_change_percentage_24h',
        color_continuous_scale='Reds',
        title="📉 Top 5 Losers (24h)",
        text_auto='+.1f'
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        yaxis_title="Price Change %",
        xaxis_title=""
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# =====================
# INSIGHTS & ANALYSIS SECTION
# =====================
st.markdown('<h2 class="section-header">🔍 Key Insights & Analysis</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 📈 Price vs Market Cap Correlation")
    
    fig = px.scatter(
        df_processed.head(50),
        x='current_price',
        y='market_cap',
        color='market_cap_category',
        size='total_volume',
        hover_name='symbol',
        log_x=True,
        log_y=True,
        size_max=40,
        opacity=0.7,
        color_discrete_map={
            'Large Cap': '#1f77b4',
            'Mid Cap': '#ff7f0e',
            'Small Cap': '#2ca02c'
        }
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_title="Price (Log Scale)",
        yaxis_title="Market Cap (Log Scale)",
        legend_title="Market Cap Size"
    )
    
    # Add trendline
    x_log = np.log(df_processed.head(50)['current_price'] + 1)
    y_log = np.log(df_processed.head(50)['market_cap'] + 1)
    z = np.polyfit(x_log, y_log, 1)
    p = np.poly1d(z)
    
    fig.add_trace(go.Scatter(
        x=df_processed.head(50)['current_price'],
        y=np.exp(p(x_log)),
        mode='lines',
        name='Trend Line',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Market Distribution Analysis")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("📦 Market Cap Distribution", "📈 Daily Returns Distribution"),
        vertical_spacing=0.15
    )
    
    # Histogram 1: Market Cap Distribution
    fig.add_trace(
        go.Histogram(
            x=np.log10(df_processed['market_cap']),
            nbinsx=30,
            marker_color='#667eea',
            opacity=0.7,
            name="Market Cap (Log10)"
        ),
        row=1, col=1
    )
    
    # Histogram 2: Returns Distribution
    fig.add_trace(
        go.Histogram(
            x=df_processed['price_change_percentage_24h'],
            nbinsx=30,
            marker_color='#764ba2',
            opacity=0.7,
            name="24h Returns"
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Market Cap (Log10 $)", row=1, col=1)
    fig.update_xaxes(title_text="24h Price Change %", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    # Add mean lines
    fig.add_vline(
        x=np.log10(df_processed['market_cap'].mean()),
        row=1, col=1,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: ${df_processed['market_cap'].mean():.2e}"
    )
    
    fig.add_vline(
        x=df_processed['price_change_percentage_24h'].mean(),
        row=2, col=1,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {df_processed['price_change_percentage_24h'].mean():.1f}%"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =====================
# RISK ASSESSMENT SECTION
# =====================
st.markdown('<h2 class="section-header">⚠️ Risk Assessment & Opportunities</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Risk Distribution")
    
    risk_counts = df_processed['market_cap_category'].value_counts()
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        color=risk_counts.index,
        color_discrete_map={
            'Large Cap': '#28a745',
            'Mid Cap': '#ffc107',
            'Small Cap': '#dc3545'
        },
        hole=0.4,
        title=""
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>"
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 🔥 High Volatility Coins")
    
    high_vol = df_processed.nlargest(5, 'volatility_24h') if 'volatility_24h' in df_processed.columns else df_processed.head(5)
    
    fig = px.bar(
        high_vol,
        y='symbol',
        x='volatility_24h' if 'volatility_24h' in high_vol.columns else 'price_change_percentage_24h',
        orientation='h',
        color='volatility_24h' if 'volatility_24h' in high_vol.columns else 'price_change_percentage_24h',
        color_continuous_scale='Reds',
        title=""
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        xaxis_title="Volatility" if 'volatility_24h' in high_vol.columns else "24h Change",
        yaxis_title="",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 💎 High Potential Coins")
    
    if 'performance_score' in df_processed.columns:
        high_potential = df_processed.nlargest(5, 'performance_score')
        
        fig = px.bar(
            high_potential,
            y='symbol',
            x='performance_score',
            orientation='h',
            color='performance_score',
            color_continuous_scale='Greens',
            title=""
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            xaxis_title="Performance Score",
            yaxis_title="",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback to top gainers
        top_gainers_small = df_processed[df_processed['market_cap_category'] == 'Small Cap'].nlargest(5, 'price_change_percentage_24h')
        
        fig = px.bar(
            top_gainers_small,
            y='symbol',
            x='price_change_percentage_24h',
            orientation='h',
            color='price_change_percentage_24h',
            color_continuous_scale='Greens',
            title="Top Small Cap Gainers"
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            xaxis_title="24h Gain %",
            yaxis_title="",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =====================
# KEY INSIGHTS & RECOMMENDATIONS
# =====================
st.markdown("---")
st.markdown('<h2 class="section-header">🎯 Key Takeaways & Recommendations</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📈 **Market Outlook**")
        st.markdown("""
        - ✅ **Bullish sentiment** with {:.0f}% coins in green
        - 📊 Average daily return: **{:.1f}%**
        - 💰 Total market cap: **${:.1f}T**
        - 🎯 **Recommendation**: Consider gradual accumulation
        """.format(
            (df_processed['price_change_percentage_24h'] > 0).mean() * 100,
            df_processed['price_change_percentage_24h'].mean(),
            df_processed['market_cap'].sum() / 1e12
        ))
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("### ⚠️ **Risk Factors**")
        st.markdown("""
        - 📉 **High concentration**: Top 10 coins control {:.1f}% of market
        - 🔥 **Volatility**: Average {:.1f}% daily price swings
        - 📊 **Distribution**: Market heavily skewed towards large caps
        - 🛡️ **Recommendation**: Diversify across market caps
        """.format(
            top_10_dominance,
            df_processed['volatility_24h'].mean() * 100 if 'volatility_24h' in df_processed.columns else 5
        ))
        st.markdown('</div>', unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 💡 **Opportunities**")
        st.markdown("""
        - 🚀 **Emerging coins**: {:.0f} small caps showing >10% gains
        - 📈 **Momentum**: {:.0f}% coins outperforming market average
        - 💎 **Value**: Multiple mid-caps with strong fundamentals
        - 🎯 **Recommendation**: Focus on high-volume, low-volatility coins
        """.format(
            len(df_processed[(df_processed['market_cap_category'] == 'Small Cap') & 
                           (df_processed['price_change_percentage_24h'] > 10)]),
            (df_processed['price_change_percentage_24h'] > df_processed['price_change_percentage_24h'].mean()).mean() * 100
        ))
        st.markdown('</div>', unsafe_allow_html=True)

# =====================
# DATA TABLE FOR REFERENCE
# =====================
with st.expander("📋 **View Detailed Data Table**", expanded=False):
    st.markdown("### Detailed Cryptocurrency Data")
    
    # Select columns untuk display
    display_cols = ['name', 'symbol', 'market_cap_formatted', 
                   'current_price', 'price_change_percentage_24h',
                   'market_cap_category']
    
    # Filter columns yang ada
    available_cols = [col for col in display_cols if col in df_processed.columns]
    
    if available_cols:
        display_df = df_processed[available_cols].head(20).copy()
        
        # Apply styling
        def color_returns(val):
            if isinstance(val, (int, float)):
                return 'color: green' if val > 0 else 'color: red'
            elif isinstance(val, str) and '%' in val:
                return 'color: green' if '+' in val else 'color: red'
            return ''
        
        styled_df = display_df.style.applymap(color_returns, 
                                             subset=['price_change_percentage_24h'])
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400,
            column_config={
                'market_cap_formatted': st.column_config.TextColumn("Market Cap"),
                'price_change_percentage_24h': st.column_config.ProgressColumn(
                    "24h Change %",
                    format="%.1f%%",
                    min_value=float(df_processed['price_change_percentage_24h'].min()),
                    max_value=float(df_processed['price_change_percentage_24h'].max())
                )
            }
        )

# =====================
# FOOTER & CREDITS
# =====================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

with footer_col1:
    st.markdown("""
    <div style="text-align: center;">
        <p style="color: #6c757d; font-size: 0.9rem;">
        📅 Last Updated<br>
        <span style="font-weight: 600;">Today</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

with footer_col2:
    st.markdown("""
    <div style="text-align: center;">
        <p style="color: #6c757d; font-size: 0.9rem;">
        📊 <span style="font-weight: 600;">Crypto Market Intelligence Dashboard</span><br>
        Data Analysis & Visualization Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

with footer_col3:
    st.markdown("""
    <div style="text-align: center;">
        <p style="color: #6c757d; font-size: 0.9rem;">
        🔍 Data Points<br>
        <span style="font-weight: 600;">{:,}</span> cryptocurrencies
        </p>
    </div>
    """.format(len(df_processed)), unsafe_allow_html=True)

# =====================
# PRESENTATION MODE BUTTON
# =====================
if st.button("🎯 Enter Presentation Mode", use_container_width=True):
    st.balloons()
    st.success("Presentation mode activated! Press 'F' for fullscreen.")
