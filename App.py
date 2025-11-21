import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
import random

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="LifeSync: Relocation Risk Engine",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
    <style>
    .big-font { font-size:30px !important; font-weight: bold; }
    .risk-score { font-size: 48px; font-weight: 800; }
    .metric-card { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

@st.cache_resource(show_spinner=False)
def train_or_load_medical_model():
    """Train medical model if not exists, else load"""
    if os.path.exists("medical_model.pkl"):
        return joblib.load("medical_model.pkl")
    st.info("Training medical model (one-time)...")
    
    df = pd.read_csv("Data_Spark_dataset/medical_insurance.csv")
    
    # Ensure required columns exist
    for col in ['children','smoker','sex']:
        if col not in df.columns:
            if col == 'children':
                df[col] = 0
            else:
                df[col] = 0  # default encoding
    
    # Encode categorical columns
    df['smoker'] = df['smoker'].map({'Yes':1, 'No':0}).fillna(0)
    df['sex'] = df['sex'].map({'Male':0, 'Female':1}).fillna(0)
    
    df = pd.get_dummies(df, columns=['region'], drop_first=True)
    
    feature_cols = ['age','bmi','children','smoker','sex'] + [c for c in df.columns if c.startswith('region_')]
    X = df[feature_cols]
    y = df['annual_medical_cost']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, "medical_model.pkl")
    return model

@st.cache_resource(show_spinner=False)
def load_housing_data():
    """Load Airbnb median rents per borough"""
    if os.path.exists("housing_data.pkl"):
        return joblib.load("housing_data.pkl")
    
    st.info("Processing housing data (one-time)...")
    df = pd.read_csv("Data_Spark_dataset/Airbnb_Open_Data.csv", low_memory=False)
    
    # Detect correct borough column
    if 'neighbourhood_group' in df.columns:
        boro_col = 'neighbourhood_group'
    elif 'borough' in df.columns:
        boro_col = 'borough'
    else:
        boro_col = df.columns[0]  # fallback
    
    # Ensure price column exists and is numeric
    if 'price' not in df.columns:
        df['price'] = 100  # default value
    
    # Convert price to numeric, coerce errors
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(100)
    
    # Compute median per borough
    boro_stats = df.groupby(boro_col)['price'].median().to_dict()
    joblib.dump({'borough_stats': boro_stats}, 'housing_data.pkl')
    return {'borough_stats': boro_stats}


@st.cache_data(show_spinner=False)
def get_recipes(category=None, n=5):
    df = pd.read_csv("Data_Spark_dataset/1_Recipe_csv.csv")
    if category:
        df = df[df['category'].str.contains(category, case=False)]
    if len(df) < n:
        n = len(df)
    return df.sample(n).to_dict(orient='records')

@st.cache_data(show_spinner=False)
def get_playlist(genre=None, n=5):
    df = pd.read_csv("Data_Spark_dataset/dataset.csv")
    if genre:
        df = df[df['track_genre'].str.contains(genre, case=False)]
    if len(df) < n:
        n = len(df)
    return df.sample(n)[['artists','track_name']].rename(columns={'artists':'artist','track_name':'track'}).to_dict(orient='records')

def get_local_analysis(profile):
    med_model = train_or_load_medical_model()
    housing_data = load_housing_data()
    
    # Prepare medical input
    input_data = pd.DataFrame([[
        profile['age'],
        profile['bmi'],
        0,  # children default
        1 if profile['is_smoker'] else 0,
        0   # sex=male default
    ]], columns=['age','bmi','children','smoker','sex'])
    
    # Add missing region columns
    region_cols = [c for c in med_model.feature_names_in_ if c.startswith('region_')]
    for col in region_cols:
        if col not in input_data.columns:
            input_data[col] = 0

    pred_medical = med_model.predict(input_data)[0]
    
    # Rent estimate
    est_rent = housing_data['borough_stats'].get(profile['borough'], 100) * 30
    
    # Risk calculation
    remaining_budget = profile['budget'] - pred_medical
    risk_status = "CRITICAL" if remaining_budget < est_rent else "STABLE"
    risk_score = min(100, max(0, int((est_rent / max(1, remaining_budget)) * 100)))
    
    # Recommendations
    playlist = get_playlist(profile['genre'])
    recipes = get_recipes()
    
    return {
        "predicted_medical": round(pred_medical,2),
        "estimated_rent": round(est_rent,2),
        "risk_score": risk_score,
        "risk_status": risk_status,
        "analysis_text": "Your relocation risk analysis is ready.",
        "affordable_neighborhoods": ["Queens","Bronx","Staten Island"],
        "expensive_neighborhoods": ["Manhattan","Brooklyn","Upper East Side"],
        "playlist": playlist,
        "recipes": recipes
    }

# ==========================================
# UI: SIDEBAR
# ==========================================
with st.sidebar:
    st.title("⚡ LifeSync")
    st.caption("Relocation Risk Engine")
    
    st.header("1. Health Profile")
    age = st.slider("Age", 18, 65, 30)
    bmi = st.slider("BMI", 15.0, 50.0, 24.0)
    is_smoker = st.checkbox("Smoker?")
    
    st.header("2. Financials")
    budget = st.number_input("Monthly Budget ($)", value=5000, step=100)
    
    st.header("3. Lifestyle")
    borough = st.selectbox("Target NYC Borough", 
        ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'])
    genre = st.selectbox("Music Vibe", 
        ['Jazz', 'Pop', 'Rock', 'Hip-Hop', 'Classical', 'Electronic'])

    run_btn = st.button("Calculate Risk", type="primary", use_container_width=True)

# ==========================================
# UI: MAIN CONTENT
# ==========================================
if run_btn:
    with st.spinner("Running Actuarial Models & Market Analysis..."):
        profile = {
            "age": age, "bmi": bmi, "is_smoker": is_smoker,
            "budget": budget, "borough": borough, "genre": genre
        }

        result = get_local_analysis(profile)

        # UI DISPLAY
        st.markdown(f"## Relocation Report: <span style='color:#0ea5e9'>{borough}</span>", unsafe_allow_html=True)
        st.markdown("Analysis based on current market data and actuarial health predictions.")
        st.divider()

        # METRICS ROW
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Monthly Budget", f"${budget:,.0f}")
        c2.metric("Predicted Medical", f"${result['predicted_medical']:,.0f}", 
                  delta="-High Risk" if is_smoker else "Standard", delta_color="inverse")
        c3.metric(f"Est. Rent ({borough})", f"${result['estimated_rent']:,.0f}")
        
        with c4:
            color = "red" if result['risk_status'] == "CRITICAL" else "green"
            st.markdown(f"""
            <div style="text-align: center; border: 2px solid {color}; border-radius: 10px; padding: 5px;">
                <div style="font-size: 12px; font-weight: bold; color: gray;">RISK SCORE</div>
                <div style="font-size: 32px; font-weight: bold; color: {color};">{result['risk_score']}%</div>
                <div style="font-size: 14px; font-weight: bold; background-color: {color}; color: white; border-radius: 4px;">{result['risk_status']}</div>
            </div>
            """, unsafe_allow_html=True)

        # EXECUTIVE SUMMARY
        st.subheader("📊 Executive Summary")
        if result['risk_status'] == 'CRITICAL':
            st.error(result['analysis_text'])
        else:
            st.success(result['analysis_text'])

        # COLUMNS FOR DETAILS
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### 💰 Financial Reality")
            real_budget = budget - result['predicted_medical']
            gap = real_budget - result['estimated_rent']
            st.info(f"""
            **Real Rent Budget:** ${real_budget:,.0f}  
            *(Budget - Medical Costs)*
            
            **Market Gap:** {"+" if gap > 0 else ""}${gap:,.0f}
            """)
            
            st.markdown("#### 📍 Market Context")
            st.write("**Affordable Zones:** " + ", ".join(result['affordable_neighborhoods']))
            st.write("**Expensive Zones:** " + ", ".join(result['expensive_neighborhoods']))

        with col_right:
            st.markdown("#### 🎵 Stress Relief Playlist")
            for song in result['playlist']:
                st.text(f"♪ {song['track']} - {song['artist']}")

            st.markdown("#### 🥦 Diet Recommendations")
            for recipe in result['recipes']:
                st.text(f"• {recipe['recipe_title']} ({recipe['num_ingredients']} ingredients)")

else:
    st.markdown("""
    <div style="text-align: center; padding-top: 50px; color: #6c757d;">
        <h1>🏙️</h1>
        <h2>Ready to Relocate?</h2>
        <p>Adjust the parameters in the sidebar and hit "Calculate Risk" <br>
        to see if your move to NYC is financially viable.</p>
    </div>
    """, unsafe_allow_html=True)
