import cv2             # OpenCV: Thư viện xử lý luồng Video và giao diện
import mediapipe as mp # MediaPipe: AI lõi trích xuất 33 tọa độ xương khớp
import mediapipe.python.solutions.pose as mp_pose_module
import numpy as np     # Numpy: Thư viện toán học xử lý vector
import pickle          # Pickle: Nạp mô hình học máy (.pkl)
import os              # OS: Tương tác hệ thống tệp

# ==========================================
# 1. KHỐI TOÁN HỌC KHÔNG GIAN
# ==========================================
def calculate_angle(a, b, c):
    """Tính góc vật lý tại đỉnh B bằng tích vô hướng vector 2D."""
    a, b, c = np.array(a), np.array(b), np.array(c) 
    ba, bc = a - b, c - b                           
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# ==========================================
# 2. LỚP ĐIỀU KHIỂN CHÍNH: AI COACH
# ==========================================
class BenchPressCoach:
    def __init__(self, model_path):
        """Khởi tạo AI và định tuyến bộ khung xương."""
        self.mp_pose = mp.solutions.pose # type: ignore
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
            
        self.target_points = [11, 12, 13, 14, 15, 16, 23, 24]
        self.connections = [
            (11, 13), (12, 14), # Vai - Khuỷu
            (13, 15), (14, 16), # Khuỷu - Cổ tay
            (11, 23), (12, 24), # Vai - Hông 
            (11, 12), (23, 24)  # Ngang người
        ]
        self.manual_flip = False 

    def process_frame(self, frame):
        """Pipeline xử lý từng khung hình video theo thuật toán Trục Y."""
        orig_h, orig_w, _ = frame.shape 
        
        if self.manual_flip:
            frame = cv2.rotate(frame, cv2.ROTATE_180) 
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame) 
        
        status_text = "DANG CHO NGUOI TAP..."
        color = (200, 200, 200) # Khởi tạo mặc định là màu Xám
        
        angle_l = angle_r = arm_bend_l = arm_bend_r = 0
        y_ratio_l = y_ratio_r = 0
        state_info = "Chua xac dinh"

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark 
            
            # --- BƯỚC 1: ĐO ĐẠC GÓC ---
            # Góc khuỷu tay
            arm_bend_l = calculate_angle([lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y])
            arm_bend_r = calculate_angle([lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y])
            
            # Góc nách
            angle_l = calculate_angle([lm[23].x, lm[23].y], [lm[11].x, lm[11].y], [lm[13].x, lm[13].y])
            angle_r = calculate_angle([lm[24].x, lm[24].y], [lm[12].x, lm[12].y], [lm[14].x, lm[14].y])
            
            # --- BƯỚC 2: THUẬT TOÁN ĐO ĐỘ LỆCH TRỤC Y (Y-AXIS TRACKING) ---
            # Tính khoảng cách DỌC (chỉ lấy trục Y) từ Vai đến Cổ tay
            y_dist_wrist_l = abs(lm[15].y - lm[11].y)
            y_dist_wrist_r = abs(lm[16].y - lm[12].y)
            
            # Tính khoảng cách DỌC của thân người để làm thước đo chuẩn
            y_torso_l = abs(lm[23].y - lm[11].y) + 1e-6
            y_torso_r = abs(lm[24].y - lm[12].y) + 1e-6
            
            # Tỷ lệ dịch chuyển: Nếu tay duỗi thẳng hướng lên trần, tỷ lệ này sẽ cực kỳ nhỏ (< 0.25)
            y_ratio_l = y_dist_wrist_l / y_torso_l
            y_ratio_r = y_dist_wrist_r / y_torso_r
            
            # --- BƯỚC 3: MÁY TRẠNG THÁI DỨT KHOÁT ---
            # LÊN TẠ (UP): Góc khuỷu lớn (> 140) VÀ cổ tay không bị hạ thấp (Tỷ lệ Y < 0.25)
            is_up = (arm_bend_l > 140 or arm_bend_r > 140) and (y_ratio_l < 0.25 and y_ratio_r < 0.25)
            
            if is_up:
                # TRẠNG THÁI LÊN TẠ: ĐÚNG YÊU CẦU LÀ XÁM VÀ CHƯA XÁC ĐỊNH
                state_info = "Pha day len (Up)"
                status_text = "CHUA XAC DINH"
                color = (150, 150, 150) # Ép toàn bộ thành màu xám
            else:
                # TRẠNG THÁI HẠ TẠ: ÁP DỤNG LUẬT BẮT LỖI GÓC NÁCH
                state_info = "Pha ha ta (Down)"
                
                # Luật do User chỉ định: Chỉ đúng khi <= 75 độ. Lớn hơn là Sai.
                if angle_l > 75 or angle_r > 75: 
                    status_text = "SAI FORM! GOC NACH > 75 DO"
                    color = (0, 0, 255) # Màu Đỏ
                else:
                    status_text = "FORM CHUAN!"
                    color = (0, 255, 0) # Màu Xanh

            # --- BƯỚC 4: RENDER BỘ KHUNG XƯƠNG ---
            for p1, p2 in self.connections:
                pt1 = (int(lm[p1].x * orig_w), int(lm[p1].y * orig_h)) 
                pt2 = (int(lm[p2].x * orig_w), int(lm[p2].y * orig_h))
                cv2.line(frame, pt1, pt2, color, thickness=4) 
            
            for point_idx in self.target_points:
                cx, cy = int(lm[point_idx].x * orig_w), int(lm[point_idx].y * orig_h)
                cv2.circle(frame, (cx, cy), radius=8, color=(0, 255, 255), thickness=-1) 

        # --- BƯỚC 5: RENDER GIAO DIỆN DASHBOARD ---
        dash_h = 135 
        pad_w = max(0, 480 - orig_w) 
        
        # Mở rộng Canvas xuống đáy
        frame_padded = cv2.copyMakeBorder(frame, 0, dash_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(15, 15, 15))
        box_y1 = orig_h 
        
        # In các dòng thông số chẩn đoán
        cv2.putText(frame_padded, f"Trang thai : {state_info}", (15, box_y1 + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        
        cv2.putText(frame_padded, f"Goc Nach   : Trai {int(angle_l):03d} do | Phai {int(angle_r):03d} do", (15, box_y1 + 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # In thêm Tỷ lệ Trục Y để quan sát độ "ảo" của camera
        cv2.putText(frame_padded, f"Ty le Truc Y: Trai {y_ratio_l:.2f} | Phai {y_ratio_r:.2f}", (15, box_y1 + 115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # LỆNH IN TEXT DUY NHẤT: Tránh hiện tượng chồng chữ
        # Căn text to, rõ ràng ở phần đầu video
        cv2.putText(frame_padded, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        
        return frame_padded 

    def run(self, source):
        """Khởi chạy I/O Stream Video."""
        cap = cv2.VideoCapture(source) 
        while cap.isOpened():
            ret, frame = cap.read() 
            if not ret: break       
            
            # Cố định chiều cao 700px để giao diện luôn ổn định
            h_orig, w_orig = frame.shape[:2]
            target_height = 700 
            target_width = int(w_orig * (target_height / h_orig)) 
            frame = cv2.resize(frame, (target_width, target_height))
            
            processed_frame = self.process_frame(frame)
            cv2.imshow('AI Bench Press Coach (Y-Axis Tracking)', processed_frame) 
            
            key = cv2.waitKey(1) & 0xFF 
            if key == ord('q'): break                   
            elif key == ord('f'): self.manual_flip = not self.manual_flip 
            
        cap.release()          
        cv2.destroyAllWindows()

if __name__ == "__main__":
    coach = BenchPressCoach("model_angles_v2.pkl")
    
    print("\n" + "="*50 + "\n PHẦN MỀM HỖ TRỢ TẬP GYM - AI COACH\n" + "="*50)
    print("1. Chạy từ file Video (.mp4)\n2. Chạy từ Camera (Webcam/DroidCam)\n" + "-"*50)
    
    choice = input("👉 Mời bạn chọn chế độ (1/2): ")
    if choice == "1":
        while True:
            path = input("📁 Nhập tên video (VD: dung.mp4) hoặc gõ 'q' để thoát: ")
            if path.lower() == 'q': break 
            
            if os.path.exists(path):
                coach.run(path)
                break           
            else: 
                print(f"❌ Không tìm thấy file: {path} trong thư mục.")
    else:
        ip = input("📷 Nhập IP DroidCam (VD: 192.168.1.5:4747) hoặc gõ 0 cho Webcam: ")
        coach.run(int(ip) if ip == "0" else ip)