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
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS สำหรับความสวยงาม
st.markdown("""
    <style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
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
    
    /* Sidebar styling */
    .sidebar-container {
        background: linear-gradient(180deg, #f0f4ff 0%, #e8ecff 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem;
    }
    
    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Content container */
    .content-container {
        background: white;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem auto;
        max-width: 1200px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
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
    
    /* Slider styling */
    .stSlider > div > div {
        color: #667eea;
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
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255,255,255,0.8);
        margin-top: 3rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
        <div class="sidebar-container">
            <h3>📋 About</h3>
            <div class="info-box">
                <p>This application uses a trained K-Means clustering model to predict cluster assignments based on input features.</p>
                
                <h4 style="color: #667eea; margin-top: 1rem;">Model Details:</h4>
                <ul>
                    <li>Algorithm: K-Means</li>
                    <li>Dataset: Iris</li>
                    <li>Features: 4</li>
                </ul>
            </div>
            
            <h3>🎯 How to Use</h3>
            <div class="info-box">
                <ol>
                    <li><strong>Manual Input:</strong> Enter feature values using sliders</li>
                    <li><strong>CSV Upload:</strong> Upload a CSV file for batch predictions</li>
                    <li><strong>View Results:</strong> See cluster assignments and visualizations</li>
                </ol>
            </div>
            
            <button onclick="window.location.reload()" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 10px 20px; border-radius: 8px; width: 100%; cursor: pointer; font-weight: 600;">
                🔄 Reset All
            </button>
        </div>
        """, unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-container">
        <h1 class="header-title"> K-Means Clustering App</h1>
        <p class="header-subtitle">Interactive Machine Learning Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

# เนื้อหาหลัก
st.markdown('<div class="content-container">', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Manual Prediction", "📁 Batch Prediction", "️ Model Information"])

# Tab 1: Manual Prediction
with tab1:
    st.markdown("### 🎯 Enter Feature Values")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.slider(
            "Sepal Length (cm)",
            min_value=4.0,
            max_value=8.0,
            value=5.5,
            step=0.1,
            help="The length of the sepal in centimeters"
        )
        
        sepal_width = st.slider(
            "Sepal Width (cm)",
            min_value=2.0,
            max_value=4.5,
            value=3.0,
            step=0.1,
            help="The width of the sepal in centimeters"
        )
    
    with col2:
        petal_length = st.slider(
            "Petal Length (cm)",
            min_value=1.0,
            max_value=7.0,
            value=4.0,
            step=0.1,
            help="The length of the petal in centimeters"
        )
        
        petal_width = st.slider(
            "Petal Width (cm)",
            min_value=0.1,
            max_value=2.5,
            value=1.5,
            step=0.1,
            help="The width of the petal in centimeters"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ปุ่ม Predict
    if st.button("🔮 Predict Cluster", use_container_width=True):
        # สร้าง input data
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # โหลด Iris dataset และฝึกโมเดล (หรือโหลดโมเดลที่ฝึกไว้)
        iris = load_iris()
        X = iris.data
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        input_scaled = scaler.transform(input_data)
        
        # ฝึก K-Means
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        
        # ทำนาย
        prediction = kmeans.predict(input_scaled)[0]
        cluster_name = iris.target_names[prediction] if prediction < len(iris.target_names) else f"Cluster {prediction}"
        
        # แสดงผล
        st.markdown("<br>", unsafe_allow_html=True)
        st.success(f"### 🎉 Predicted Cluster: **{cluster_name}** (Cluster {prediction})")
        
        # แสดงข้อมูล input
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sepal Length", f"{sepal_length} cm")
        with col2:
            st.metric("Sepal Width", f"{sepal_width} cm")
        with col3:
            st.metric("Petal Length", f"{petal_length} cm")
        with col4:
            st.metric("Petal Width", f"{petal_width} cm")
        
        # Visualization
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Cluster Visualization")
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Sepal
        scatter1 = ax[0].scatter(X_scaled[:, 0], X_scaled[:, 1], 
                                 c=kmeans.labels_, cmap='viridis', alpha=0.6)
        ax[0].scatter(input_scaled[0, 0], input_scaled[0, 1], 
                     c='red', s=200, marker='X', label='Your Input', edgecolors='black', linewidth=2)
        ax[0].set_xlabel('Sepal Length (scaled)')
        ax[0].set_ylabel('Sepal Width (scaled)')
        ax[0].set_title('Sepal Features')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        
        # Plot 2: Petal
        scatter2 = ax[1].scatter(X_scaled[:, 2], X_scaled[:, 3], 
                                 c=kmeans.labels_, cmap='viridis', alpha=0.6)
        ax[1].scatter(input_scaled[0, 2], input_scaled[0, 3], 
                     c='red', s=200, marker='X', label='Your Input', edgecolors='black', linewidth=2)
        ax[1].set_xlabel('Petal Length (scaled)')
        ax[1].set_ylabel('Petal Width (scaled)')
        ax[1].set_title('Petal Features')
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

# Tab 2: Batch Prediction
with tab2:
    st.markdown("### 📁 Batch Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV file with columns: sepal_length, sepal_width, petal_length, petal_width"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.dataframe(df.head())
            
            if st.button("🔮 Predict All Clusters"):
                # ประมวลผล batch prediction
                st.info("Processing...")
                # เพิ่มโค้ดสำหรับ batch prediction ที่นี่
        except Exception as e:
            st.error(f"Error: {e}")

# Tab 3: Model Information
with tab3:
    st.markdown("### ℹ️ Model Information")
    
    st.markdown("""
    **Algorithm:** K-Means Clustering
    
    **Dataset:** Iris Dataset
    
    **Features:**
    - Sepal Length (cm)
    - Sepal Width (cm)
    - Petal Length (cm)
    - Petal Width (cm)
    
    **Number of Clusters:** 3
    
    **Performance Metrics:**
    - Inertia: (จะแสดงหลังจากฝึกโมเดล)
    - Silhouette Score: (จะแสดงหลังจากฝึกโมเดล)
    """)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>🎓 Machine Learning for Python Programming Course</p>
        <p>Built with ❤️ using Streamlit | © 2026</p>
    </div>
    """, unsafe_allow_html=True)