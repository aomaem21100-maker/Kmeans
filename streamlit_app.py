import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# ตั้งค่าหน้า
st.set_page_config(
    page_title="K-Means Clustering App",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS สำหรับความสวยงาม
st.markdown("""
    <style>
    /* ซ่อน Sidebar และเมนูทั้งหมด */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .css-1d391kg {display: none;}
    section[data-testid="stSidebar"] {display: none;}
    
    /* Main background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem auto;
        max-width: 1200px;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Content container */
    .content-container {
        background: white;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem auto;
        max-width: 1200px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.05);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        width: 100%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        transform: translateY(-2px);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        margin-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🔮 K-Means Clustering App</h1>
        <p class="header-subtitle">Interactive Machine Learning Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

# เนื้อหาหลัก
st.markdown('<div class="content-container">', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Manual Prediction", "📁 Batch Prediction", "ℹ️ Model Information"])

# ================= TAB 1: Manual Prediction =================
with tab1:
    st.markdown("### 🎯 Enter Feature Values")
    
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.5, step=0.1)
        sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1)
    with col2:
        petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.0, step=0.1)
        petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.5, step=0.1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ปุ่ม Predict
    if st.button("🔮 Predict Cluster", use_container_width=True):
        input_features = [sepal_length, sepal_width, petal_length, petal_width]
        input_data = np.array([input_features])
        feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        
        # โหลด Iris dataset และฝึกโมเดล
        iris = load_iris()
        X = iris.data
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        input_scaled = scaler.transform(input_data)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        
        prediction = int(kmeans.predict(input_scaled)[0])
        cluster_name = iris.target_names[prediction] if prediction < len(iris.target_names) else f"Cluster {prediction}"
        
        # คำนวณระยะห่างเพื่อหา Confidence
        distances = np.linalg.norm(kmeans.cluster_centers_ - input_scaled, axis=1)
        confidence_calc = (1 / (distances[prediction] + 1e-10)) / sum(1 / (distances + 1e-10)) * 100
        
        # --- แสดงผลแบบสวยงาม (Prediction Result UI) ---
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 1. Header ผลลัพธ์
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <div style="font-size: 1.2rem; opacity: 0.9;">🔮 Prediction Result</div>
                <h1 style="font-size: 3.5rem; margin: 0.5rem 0; font-weight: bold;">Cluster {prediction}</h1>
                <div style="font-size: 1.3rem; font-weight: 500;">({cluster_name})</div>
            </div>
            """, unsafe_allow_html=True)
            
        # 2. Info Cards (3 ใบ)
        st.markdown(f"""
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-bottom: 2rem;">
                <div style="background: white; padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.08); border-top: 4px solid #667eea;">
                    <div style="color: #667eea; font-size: 0.85rem; font-weight: 700; margin-bottom: 0.5rem; letter-spacing: 1px;">INPUT FEATURES</div>
                    <div style="font-size: 2rem; font-weight: bold; color: #333;">{len(feature_names)}</div>
                </div>
                <div style="background: white; padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.08); border-top: 4px solid #764ba2;">
                    <div style="color: #764ba2; font-size: 0.85rem; font-weight: 700; margin-bottom: 0.5rem; letter-spacing: 1px;">CLUSTER ASSIGNED</div>
                    <div style="font-size: 2rem; font-weight: bold; color: #333;">{prediction}</div>
                </div>
                <div style="background: white; padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.08); border-top: 4px solid #f093fb;">
                    <div style="color: #f093fb; font-size: 0.85rem; font-weight: 700; margin-bottom: 0.5rem; letter-spacing: 1px;">CONFIDENCE</div>
                    <div style="font-size: 2rem; font-weight: bold; color: #333;">{confidence_calc:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # 3. Distances Table
        st.markdown("### 📏 Distances to All Cluster Centers")
        distances_df = pd.DataFrame({
            'Cluster': [f'Cluster {i}' for i in range(len(distances))],
            'Distance': [f"{d:.4f}" for d in distances],
            'Status': ['✅ Nearest (Assigned)' if i == prediction else '❌' for i in range(len(distances))]
        })
        st.dataframe(distances_df, use_container_width=True, hide_index=True)
        
        # 4. Radar Chart Visualization
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Feature Visualization (Radar Chart)")
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        num_vars = len(feature_names)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        # Normalize ค่าสำหรับวาดกราฟ (สเกล 0-5 เพื่อความสวยงาม)
        max_vals = X.max(axis=0)
        values = [v / m * 5 for v, m in zip(input_features, max_vals)]
        values += values[:1]
        
        center_vals = [v / m * 5 for v, m in zip(kmeans.cluster_centers_[prediction], max_vals)]
        center_vals += center_vals[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label='Your Input', color='#667eea')
        ax.fill(angles, values, alpha=0.25, color='#667eea')
        
        ax.plot(angles, center_vals, 'o-', linewidth=2.5, label=f'Cluster {prediction} Center', color='#f093fb')
        ax.fill(angles, center_vals, alpha=0.25, color='#f093fb')
        
        ax.set_ylim(0, 5.5)
        ax.set_thetagrids(np.degrees(angles[:-1]), feature_names, fontsize=11)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        st.pyplot(fig)

# ================= TAB 2: Batch Prediction =================
with tab2:
    st.markdown("### 📁 Batch Prediction")
    st.info("อัปโหลดไฟล์ CSV ที่มีคอลัมน์: sepal_length, sepal_width, petal_length, petal_width")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🔮 Predict All Clusters"):
                st.info("กำลังประมวลผล... (ฟีเจอร์นี้พร้อมสำหรับการพัฒนาต่อยอด)")
        except Exception as e:
            st.error(f"Error: {e}")

# ================= TAB 3: Model Information =================
with tab3:
    st.markdown("### ℹ️ Model Information")
    st.markdown("""
    - **Algorithm:** K-Means Clustering
    - **Dataset:** Iris Dataset (Built-in)
    - **Features:** Sepal Length, Sepal Width, Petal Length, Petal Width
    - **Number of Clusters:** 3
    - **Preprocessing:** Standard Scaler (Z-score normalization)
    """)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>🎓 Machine Learning for Python Programming Course</p>
        <p>Built with ❤️ using Streamlit | © 2026</p>
    </div>
    """, unsafe_allow_html=True)