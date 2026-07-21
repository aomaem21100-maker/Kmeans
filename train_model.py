import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt

# 1. โหลดข้อมูล
print("กำลังโหลดข้อมูล...")
df = pd.read_csv('Mall_Customers.csv')  # หรือใช้ไฟล์ของคุณ

# 2. เลือก features สำหรับ clustering
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# 3. Standardize ข้อมูล
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. หาจำนวน clusters ที่ดีที่สุด (Elbow Method)
print("กำลังหาจำนวน clusters ที่ดีที่สุด...")
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# พล็อต Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal K')
plt.grid(True)
plt.savefig('elbow_curve.png')
plt.show()

# 5. สร้างโมเดล K-Means ด้วย K ที่เหมาะสม (สมมติ K=5)
print("กำลังฝึกโมเดล...")
optimal_k = 5  # เปลี่ยนตามผลลัพธ์จาก Elbow Method
kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_model.fit(X_scaled)

# 6. บันทึกโมเดล
print("กำลังบันทึกโมเดล...")
with open('models/kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans_model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(features, f)

print("✅ ฝึกโมเดลและบันทึกเรียบร้อย!")
print(f"📊 จำนวน clusters: {optimal_k}")
print(f"📁 ไฟล์ที่สร้าง: kmeans_model.pkl, scaler.pkl, feature_names.pkl")
