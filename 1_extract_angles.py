import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np

# --- HÀM TOÁN HỌC: TÍNH GÓC GIỮA 3 ĐIỂM ---
def calculate_angle(a, b, c):
    # Chuyển danh sách tọa độ thành mảng Numpy để tính toán vector nhanh hơn
    a, b, c = np.array(a), np.array(b), np.array(c)
    
    # Tạo 2 vector: ba (từ Vai đến Hông) và bc (từ Vai đến Khuỷu tay)
    ba, bc = a - b, c - b
    
    # Tính Cosin của góc dựa trên công thức Tích vô hướng (Dot Product)
    # Cos(theta) = (ba . bc) / (|ba| * |bc|)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Dùng hàm Arccos để chuyển giá trị Cosin về đơn vị Radian, sau đó đổi sang Độ (Degrees)
    # np.clip để giới hạn giá trị trong khoảng [-1, 1], tránh lỗi toán học nhỏ do làm tròn
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# --- KHỞI TẠO CÔNG CỤ NHẬN DIỆN ---
mp_pose = mp.solutions.pose#type: ignore
# static_image_mode=False: Tối ưu cho việc xử lý video (theo dõi liên tục)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

data_list = [] # Nơi chứa toàn bộ dữ liệu để sau này xuất ra CSV
dataset_path = "dataset/Dataset" # Đường dẫn đến thư mục gốc chứa train/test

# --- VÒNG LẶP DUYỆT THƯ MỤC ---
# Lần lượt đi vào tập "train" rồi đến tập "test"
for split in ["train", "test"]:
    split_path = os.path.join(dataset_path, split)
    
    # Duyệt qua các folder: Len_Dung, Len_Sai, Xuong_Dung, Xuong_Sai
    for folder_name in os.listdir(split_path):
        folder_path = os.path.join(split_path, folder_name)
        if not os.path.isdir(folder_path): continue
        
        # GÁN NHÃN (Labeling):
        # Nếu tên folder có chữ "Dung" -> Gán nhãn 0 (Đúng form)
        # Nếu tên folder có chữ "Sai" -> Gán nhãn 1 (Sai form/Loe nách)
        label = 0 if "Dung" in folder_name else 1
        
        # PHÂN LOẠI GIAI ĐOẠN (Phase):
        # "Len" tương ứng với giai đoạn Đẩy lên (Up), ngược lại là Xuống (Down)
        phase = "up" if "Len" in folder_name else "down"
        
        # Duyệt từng file video trong folder đó
        for video_name in os.listdir(folder_path):
            video_full_path = os.path.join(folder_path, video_name)
            cap = cv2.VideoCapture(video_full_path) # Mở video
            
            while cap.isOpened():
                ret, frame = cap.read() # Đọc từng khung hình (frame)
                if not ret: break # Hết video thì dừng
                
                # MediaPipe yêu cầu ảnh màu RGB, trong khi OpenCV mặc định là BGR
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame) # AI quét tìm khung xương
                
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    
                    # --- LOGIC TỰ ĐỘNG CHỐNG NGƯỢC CAMERA ---
                    # Tính tọa độ Y trung bình của 2 vai và 2 hông
                    y_shoulder = (lm[11].y + lm[12].y) / 2
                    y_hip = (lm[23].y + lm[24].y) / 2
                    
                    # Trong ảnh, tọa độ Y càng lớn là càng ở dưới. 
                    # Nếu Vai nằm dưới Hông (Y_vai > Y_hong) -> Video đang bị lộn ngược đầu
                    if y_shoulder > y_hip:
                        frame = cv2.rotate(frame, cv2.ROTATE_180) # Xoay lại 180 độ
                        # Quét lại ảnh đã xoay để có tọa độ chuẩn
                        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        if not results.pose_landmarks: continue
                        lm = results.pose_landmarks.landmark

                    # --- TRÍCH XUẤT GÓC VẬT LÝ ---
                    # Góc nách trái: Điểm đỉnh là Vai trái (11), hai cạnh là Hông trái (23) và Cùi chỏ trái (13)
                    angle_l = calculate_angle([lm[23].x, lm[23].y], [lm[11].x, lm[11].y], [lm[13].x, lm[13].y])
                    # Góc nách phải: Tương tự cho bên phải (24, 12, 14)
                    angle_r = calculate_angle([lm[24].x, lm[24].y], [lm[12].x, lm[12].y], [lm[14].x, lm[14].y])
                    
                    # Lưu các con số góc này vào danh sách kèm theo nhãn phân loại
                    data_list.append({
                        "left_armpit": angle_l,
                        "right_armpit": angle_r,
                        "label": label,
                        "phase": phase,
                        "split": split # Lưu lại để sau này chia Train/Test cho chuẩn
                    })
            cap.release() # Đóng video hiện tại để chuyển sang video tiếp theo
            print(f"Xong video: {video_name}")

# --- XUẤT DỮ LIỆU ---
df = pd.DataFrame(data_list) # Chuyển danh sách thành bảng (DataFrame)
df.to_csv("data_angles_v2.csv", index=False) # Lưu thành file Excel/CSV