import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ตั้งค่าหน้า
st.set_page_config(page_title="Prediction Result", layout="wide")

# CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    .result-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .result-header h1 {
        font-size: 3rem;
        margin: 0.5rem 0;
    }
    
    .info-cards {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .card-title {
        color: #667eea;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .card-value {
        font-size: 2rem;
        font-weight: bold;
        color: #333;
    }
    
    .section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ตัวอย่างข้อมูล (แทนที่ด้วยค่าจริงจากการ predict)
input_features = [5.2, 3.6, 1.5, 0.4]  # sepal_length, sepal_width, petal_length, petal_width
feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
predicted_cluster = 1
confidence = 28.77

# โหลดข้อมูลและฝึกโมเดล
iris = load_iris()
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# คำนวณระยะห่างถึง cluster centers
input_scaled = scaler.transform([input_features])
distances = np.linalg.norm(kmeans.cluster_centers_ - input_scaled, axis=1)
confidence_calc = (1 / (distances[predicted_cluster] + 1e-10)) / sum(1 / (distances + 1e-10)) * 100

# Header
st.markdown("""
    <div class="result-header">
        <div style="font-size: 1.2rem;">🔮 Prediction Result</div>
        <h1>Cluster """ + str(predicted_cluster) + """</h1>
        <div style="opacity: 0.9; margin-top: 0.5rem;">Data point classification result</div>
    </div>
    """, unsafe_allow_html=True)

# Info Cards
st.markdown("""
    <div class="info-cards">
        <div class="card">
            <div class="card-title">INPUT FEATURES</div>
            <div class="card-value">""" + str(len(feature_names)) + """ values</div>
        </div>
        <div class="card">
            <div class="card-title">CLUSTER ASSIGNED</div>
            <div class="card-value">""" + str(predicted_cluster) + """</div>
        </div>
        <div class="card">
            <div class="card-title">CONFIDENCE</div>
            <div class="card-value">""" + f"{confidence_calc:.2f}" + """%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Distances to All Cluster Centers
st.markdown("### 📏 Distances to All Cluster Centers")

distances_df = pd.DataFrame({
    'Cluster': [f'Cluster {i}' for i in range(len(distances))],
    'Distance': distances,
    'Closest': ['✅' if i == predicted_cluster else '❌' for i in range(len(distances))]
})

st.dataframe(distances_df, use_container_width=True, hide_index=True)

# Feature Visualization (Radar Chart)
st.markdown("### 📊 Feature Visualization")

# สร้าง radar chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

# จำนวน features
num_vars = len(feature_names)

# คำนวณมุม
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # ทำให้วงปิด

# ค่า input features (normalize)
values = input_features
values += values[:1]

# ค่าเฉลี่ยของแต่ละ feature จากข้อมูลทั้งหมด
means = X.mean(axis=0)
means_scaled = means / means.max() * 5  # normalize
means_scaled = means_scaled.tolist()
means_scaled += means_scaled[:1]

# วาดกราฟ
ax.plot(angles, values, 'o-', linewidth=2, label='Input Sample', color='#667eea')
ax.fill(angles, values, alpha=0.25, color='#667eea')

ax.plot(angles, means_scaled, 'o-', linewidth=2, label='Cluster Center', color='#f093fb')
ax.fill(angles, means_scaled, alpha=0.25, color='#f093fb')

# ตั้งค่า
ax.set_ylim(0, max(max(values), max(means_scaled)) * 1.2)
ax.set_thetagrids(np.degrees(angles[:-1]), feature_names)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

st.pyplot(fig)

# แสดงรายละเอียดเพิ่มเติม
with st.expander("📋 Detailed Information"):
    st.write("**Input Feature Values:**")
    for name, value in zip(feature_names, input_features):
        st.write(f"- {name}: {value}")
    
    st.write("\n**Cluster Center Values:**")
    for i, center in enumerate(kmeans.cluster_centers_[predicted_cluster]):
        st.write(f"- {feature_names[i]}: {center:.2f}")