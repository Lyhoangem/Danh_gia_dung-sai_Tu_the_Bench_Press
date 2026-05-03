import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

# ==========================================
# 1. TẢI DỮ LIỆU GÓC VẬT LÝ
# ==========================================
print("🔄 Đang tải dữ liệu từ data_angles_v2.csv...")
df = pd.read_csv("data_angles_v2.csv")

# Ép kiểu sang np.array để xóa bỏ các cảnh báo đỏ của Pylance
X = np.array(df[['left_armpit', 'right_armpit']]) 
y = np.array(df['label'])

# ==========================================
# 2. THIẾT LẬP ĐÁNH GIÁ CHÉO (K-FOLD)
# ==========================================
# Chia dữ liệu thành 5 phần (Fold). Mỗi lần sẽ học 4 phần, thi 1 phần.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Sử dụng Random Forest - Thuật toán mạnh mẽ cho dữ liệu dạng bảng
model = RandomForestClassifier(n_estimators=100, random_state=42)

# ==========================================
# 3. CHẠY KIỂM TRA & TÍNH TOÁN METRICS
# ==========================================
print("🧪 Đang chạy K-Fold Cross-Validation...")
# Dự đoán nhãn dựa trên cơ chế xoay vòng 5 lần
y_pred = cross_val_predict(model, X, y, cv=kf)

print("\n" + "="*40)
print("📊 BÁO CÁO ĐÁNH GIÁ MÔ HÌNH (GÓC ĐỘ)")
print("="*40)

# Accuracy: Tỷ lệ đoán đúng trên tổng số frame
acc = accuracy_score(y, y_pred)
print(f"✅ Accuracy (Độ chính xác tổng): {acc*100:.2f}%")

# Classification Report: Precision (P), Recall (R), F1-Score
# target_names giúp phân biệt 0 là Đúng form, 1 là Sai form
report = classification_report(y, y_pred, target_names=['Dung Form', 'Sai Form'])
print("\n📝 Chỉ số chi tiết (Precision, Recall, F1):")
print(report)

# ==========================================
# 4. HUẤN LUYỆN LẠI TOÀN BỘ & LƯU BỘ NÃO
# ==========================================
model.fit(X, y)
with open("model_angles_v2.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n💾 Đã lưu bộ não dựa trên số đo góc: model_angles_v2.pkl")