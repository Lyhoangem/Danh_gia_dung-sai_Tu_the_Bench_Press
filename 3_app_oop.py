import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import tempfile
import os
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase

# ==========================================
# CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="AI Bench Press Coach", page_icon="🏋️‍♂️", layout="wide")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==========================================
# 1. KHỐI TOÁN HỌC KHÔNG GIAN
# ==========================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c) 
    ba, bc = a - b, c - b                           
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def calculate_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# ==========================================
# 2. LỚP ĐIỀU KHIỂN CHÍNH (AI LÕI)
# ==========================================
class BenchPressCoach:
    def __init__(self):
        # Trở về cách gọi chuẩn mực của MediaPipe
        self.mp_pose = mp.solutions.pose # type: ignore
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        if os.path.exists("model_angles_v2.pkl"):
            with open("model_angles_v2.pkl", "rb") as f:
                self.model = pickle.load(f)
                
        self.target_points = [11, 12, 13, 14, 15, 16, 23, 24]
        self.connections = [
            (11, 13), (12, 14), (13, 15), (14, 16), 
            (11, 23), (12, 24), (11, 12), (23, 24)
        ]

    def process_frame(self, frame):
        orig_h, orig_w, _ = frame.shape 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame) 
        
        status_text = "DANG CHO NGUOI TAP..."
        color = (200, 200, 200) 
        angle_l = angle_r = arm_bend_l = arm_bend_r = 0
        state_info = "Chua xac dinh"

        if getattr(results, 'pose_landmarks', None): 
            lm = results.pose_landmarks.landmark # type: ignore
            
            # 1. ĐO GÓC NÁCH
            angle_l = calculate_angle([lm[23].x, lm[23].y], [lm[11].x, lm[11].y], [lm[13].x, lm[13].y])
            angle_r = calculate_angle([lm[24].x, lm[24].y], [lm[12].x, lm[12].y], [lm[14].x, lm[14].y])
            
            # 2. ĐO GÓC KHUỶU TAY 
            arm_bend_l = calculate_angle([lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y])
            arm_bend_r = calculate_angle([lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y])
            
            # 3. MÁY TRẠNG THÁI REAL-TIME
            is_up = (arm_bend_l > 140) and (arm_bend_r > 140)
            
            if is_up:
                state_info = "Pha day len (Up)"
                status_text = "KHONG XAC DINH"
                color = (150, 150, 150) 
            else:
                state_info = "Pha ha ta (Down)"
                if angle_l <= 75 and angle_r <= 75: 
                    status_text = "FORM CHUAN"
                    color = (0, 255, 0) 
                else:
                    status_text = "FORM SAI"
                    color = (0, 0, 255) 

            # VẼ KHUNG XƯƠNG
            for p1, p2 in self.connections:
                pt1 = (int(lm[p1].x * orig_w), int(lm[p1].y * orig_h)) 
                pt2 = (int(lm[p2].x * orig_w), int(lm[p2].y * orig_h))
                cv2.line(frame, pt1, pt2, color, thickness=4) 
            
            for point_idx in self.target_points:
                cx, cy = int(lm[point_idx].x * orig_w), int(lm[point_idx].y * orig_h)
                cv2.circle(frame, (cx, cy), radius=8, color=(0, 255, 255), thickness=-1) 

        # VẼ DASHBOARD
        dash_h = 135 
        pad_w = max(0, 480 - orig_w) 
        frame = cv2.copyMakeBorder(frame, 0, dash_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(15, 15, 15))
        box_y1 = orig_h 
        
        cv2.putText(frame, f"Trang thai : {state_info}", (15, box_y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"Goc Nach   : Trai {int(angle_l):03d} | Phai {int(angle_r):03d} (Do)", (15, box_y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Goc Khuyu  : Trai {int(arm_bend_l):03d} | Phai {int(arm_bend_r):03d} (Do)", (15, box_y1 + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        
        return frame

coach = BenchPressCoach()

# ==========================================
# 3. LỚP XỬ LÝ WEBRTC
# ==========================================
class PoseProcessor(VideoProcessorBase):
    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame": # type: ignore
        img = frame.to_ndarray(format="bgr24")
        img_processed = coach.process_frame(img)
        return av.VideoFrame.from_ndarray(img_processed, format="bgr24") # type: ignore

# ==========================================
# 4. GIAO DIỆN NGƯỜI DÙNG STREAMLIT
# ==========================================
def main():
    st.sidebar.title("🏋️‍♂️ Bảng Điều Khiển")
    
    app_mode = st.sidebar.radio(
        "Chọn chức năng:",
        ["📖 Hướng dẫn sử dụng", "📱 Camera Trực tiếp (Điện thoại)", "🎞️ Upload Video Đánh giá"]
    )

    if app_mode == "📖 Hướng dẫn sử dụng":
        st.title("🏋️‍♂️ Phần Mềm AI Coach - Đánh Giá Tư Thế Bench Press")
        
        st.markdown(r"""
        ### Chào mừng bạn đến với hệ thống AI Coach!
        Hệ thống sử dụng Trí tuệ nhân tạo (Computer Vision) để tự động phân tích và sửa lỗi tư thế đẩy ngực của bạn.
        
        #### 📌 Cách hoạt động:
        1. **Pha Lên Tạ (Lockout):** Hệ thống sẽ hiện **KHÔNG XÁC ĐỊNH (Màu xám)** vì đây là pha an toàn.
        2. **Pha Hạ Tạ (Eccentric):** AI sẽ đo góc mở của nách.
           * Góc nách $\le 75^\circ$: **FORM CHUẨN (Màu xanh)**.
           * Góc nách $> 75^\circ$: **FORM SAI (Màu đỏ)** - Cảnh báo nguy cơ chấn thương khớp vai!
        
        #### ⚙️ Các chức năng chính (Xem ở menu bên trái):
        * **Camera Trực tiếp:** Dùng điện thoại hoặc laptop mở trình duyệt lên, cho phép truy cập Camera. Đặt điện thoại ở góc quay ngang (Side-view) hoặc chéo để AI chấm điểm Real-time.
        * **Upload Video:** Tải lên video bạn nhờ bạn bè quay lại ở phòng Gym để phân tích chuyên sâu.
        """)
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Bench_press_1.gif/400px-Bench_press_1.gif", caption="Tư thế Bench Press chuẩn")

    elif app_mode == "📱 Camera Trực tiếp (Điện thoại)":
        st.title("📱 Phân Tích Bằng Camera Điện Thoại")
        st.warning("Vui lòng cấp quyền truy cập Camera cho trình duyệt. Nhấn 'START' để bắt đầu.")
        
        webrtc_streamer(
            key="bench-press-camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=PoseProcessor,
            media_stream_constraints={
                "video": {"facingMode": "environment"}, 
                "audio": False
            },
            async_processing=True
        )

    elif app_mode == "🎞️ Upload Video Đánh giá":
        st.title("🎞️ Hệ Thống Chấm Điểm Video")
        uploaded_file = st.file_uploader("Tải lên video tập luyện của bạn (.mp4, .mov)", type=["mp4", "mov", "avi"])

        if uploaded_file is not None:
            st.success("✅ Đã tải video thành công!")
            if st.button("🚀 Bắt đầu Phân tích"):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                
                cap = cv2.VideoCapture(tfile.name)
                st_video = st.empty()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.info("✅ Đã phân tích xong toàn bộ video!")
                        break
                    
                    h_orig, w_orig = frame.shape[:2]
                    target_height = 600
                    target_width = int(w_orig * (target_height / h_orig))
                    frame = cv2.resize(frame, (target_width, target_height))
                    
                    processed_frame = coach.process_frame(frame)
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    st_video.image(processed_frame_rgb, channels="RGB")
                    
                cap.release()
                os.remove(tfile.name)

if __name__ == "__main__":
    main()
