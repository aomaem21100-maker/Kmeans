import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ตั้งค่าหน้า
st.set_page_config(
    page_title="K-Means Clustering App",
    page_icon="🎯",
    layout="wide"
)

# CSS สำหรับความสวยงาม
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ส่วนหัว
st.markdown('<p class="main-header">🎯 K-Means Clustering</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ระบบจัดกลุ่มข้อมูลอัตโนมัติ</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("️ การตั้งค่า")

# โหลดโมเดล
@st.cache_resource
def load_model():
    try:
        with open('models/kmeans_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, feature_names
    except FileNotFoundError:
        return None, None, None

model, scaler, feature_names = load_model()

# อัพโหลดไฟล์
st.sidebar.markdown("### 📁 อัพโหลดข้อมูล")
uploaded_file = st.sidebar.file_uploader(
    "เลือกไฟล์ CSV หรือ Excel",
    type=['csv', 'xlsx']
)

# ฟังก์ชันโหลดข้อมูล
@st.cache_data
def load_data(file):
    if file is not None:
        try:
            if file.name.endswith('.csv'):
                return pd.read_csv(file)
            else:
                return pd.read_excel(file)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
            return None
    return None

df = load_data(uploaded_file)

# แสดงผลถ้ามีข้อมูล
if df is not None:
    st.sidebar.success("✅ อัพโหลดสำเร็จ!")
    st.sidebar.write(f"จำนวนแถว: {df.shape[0]}")
    st.sidebar.write(f"จำนวนคอลัมน์: {df.shape[1]}")
    
    # เลือก features
    st.sidebar.markdown("### 🎯 เลือก Features")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        feature1 = st.sidebar.selectbox("Feature 1", numeric_cols, index=0 if 'Annual Income (k$)' not in numeric_cols else numeric_cols.index('Annual Income (k$)') if 'Annual Income (k$)' in numeric_cols else 0)
        feature2 = st.sidebar.selectbox("Feature 2", numeric_cols, index=1 if 'Spending Score (1-100)' not in numeric_cols else numeric_cols.index('Spending Score (1-100)') if 'Spending Score (1-100)' in numeric_cols else 1)
        
        n_clusters = st.sidebar.slider("จำนวนClusters (K)", 2, 10, 5)
        
        # เตรียมข้อมูล
        X = df[[feature1, feature2]].values
        
        # Standardize
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            scaler_temp = StandardScaler()
            X_scaled = scaler_temp.fit_transform(X)
        
        # ทำนาย
        if model is not None:
            labels = model.predict(X_scaled)
        else:
            # สร้างโมเดลใหม่ถ้าไม่มี
            kmeans_temp = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans_temp.fit_predict(X_scaled)
        
        # เพิ่มคอลัมน์ cluster ให้ dataframe
        df['Cluster'] = labels
        
        # แสดงข้อมูล
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 ตัวอย่างข้อมูล")
            st.dataframe(df.head(10))
        
        with col2:
            st.markdown("### 📈 สถิติ")
            st.write(df.describe())
        
        # พล็อต
        st.markdown("### 🎨 ผลการจัดกลุ่ม")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df[feature1], df[feature2], 
                            c=df['Cluster'], 
                            cmap='viridis', 
                            alpha=0.6,
                            s=100,
                            edgecolors='black',
                            linewidth=0.5)
        
        plt.xlabel(feature1, fontsize=12)
        plt.ylabel(feature2, fontsize=12)
        plt.title(f'K-Means Clustering (K={len(set(labels))})', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # แสดงข้อมูลแต่ละ cluster
        st.markdown("### 📋 ข้อมูลแต่ละ Cluster")
        cluster_info = df.groupby('Cluster')[[feature1, feature2]].mean()
        st.dataframe(cluster_info.style.background_gradient(cmap='Blues'))
        
        # ดาวน์โหลดผล
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 ดาวน์โหลดผลลัพธ์ (CSV)",
            data=csv,
            file_name='kmeans_clustering_result.csv',
            mime='text/csv'
        )
        
    else:
        st.warning("⚠️ ข้อมูลต้องมีอย่างน้อย 2 คอลัมน์ที่เป็นตัวเลข")
else:
    st.info("👆 กรุณาอัพโหลดไฟล์ CSV หรือ Excel เพื่อเริ่มต้น")
    
    # ตัวอย่าง
    st.markdown("### 📝 รูปแบบไฟล์ที่รองรับ:")
    st.markdown("""
    - **CSV**: ไฟล์ .csv
    - **Excel**: ไฟล์ .xlsx, .xls
    - ต้องมีคอลัมน์ที่เป็นตัวเลขอย่างน้อย 2 คอลัมน์
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>สร้างด้วย ❤️ โดยใช้ Streamlit และ Scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)