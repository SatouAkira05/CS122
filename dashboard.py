import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import platform
from PIL import Image
import os

# ==========================================
# PHẦN 1: CẤU HÌNH & VÁ LỖI HỆ ĐIỀU HÀNH
# ==========================================

# 1.1 Vá lỗi WindowsPath khi chạy trên Linux/Colab/MacOS
# (Mô hình huấn luyện trên Windows yêu cầu cấu trúc này khi load)
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

# 1.2 Cấu hình trang Streamlit (Tiêu đề tab, bố cục rộng)
st.set_page_config(
    page_title="AI Safety Driver Monitor",
    page_icon="🛡️",
    layout="wide"
)

# ==========================================
# PHẦN 2: HÀM CỐT LÕI (MODEL & XỬ LÝ)
# ==========================================

# 2.1 Hàm tải mô hình YOLOv5 (Dùng cache để tránh load lại khi rerun)
@st.cache_resource
def load_yolov5_model(weights_path):
    if not os.path.exists(weights_path):
        st.error(f"❌ Không tìm thấy file trọng số: {weights_path}")
        return None
    
    with st.spinner('Đang tải mô hình AI... Vui lòng đợi.'):
        try:
            # Load model custom
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
            return model
        except Exception as e:
            st.error(f"❌ Lỗi khi load mô hình: {e}")
            return None

# 2.2 Hàm chạy nhận diện và xử lý thống kê
def get_detection_statistics(model, image):
    # Chạy nhận diện
    results = model(image)
    
    # Trích xuất nhãn
    labels = results.xyxyn[0][:, -1].tolist()
    # Strip whitespace và chuyển về chữ thường để đồng bộ
    detected_raw = [results.names[int(i)].strip().lower() for i in labels]
    
    # 2.3 Sắp xếp lại và thống kê theo thứ tự mong muốn (giống labels.jpg)
    target_order = ['awake', 'drowsy', 'phone', 'smoking', 'yawn']
    
    if detected_raw:
        df_count = pd.Series(detected_raw).value_counts()
        # Reindex để đảm bảo đủ 5 class và đúng thứ tự, class nào thiếu gán bằng 0
        df_count = df_count.reindex(target_order, fill_value=0)
        return df_count, results
    else:
        # Trả về Series trống nhưng đúng cấu trúc nếu không thấy gì
        return pd.Series(0, index=target_order), results

# ==========================================
# PHẦN 3: GIAO DIỆN DASHBOARD (UI/UX)
# ==========================================

# 3.1 Tiêu đề chính
st.title("🛡️ Hệ Thống Giám Sát An Toàn Lái Xe")
st.markdown("Dashboard hiển thị kết quả phân tích hành vi lái xe real-time từ mô hình AI.")
st.write("---")

# 3.2 THANH BÊN (SIDEBAR) - Nơi chứa các điều khiển
with st.sidebar:
    st.header("Cấu Hình Hệ Thống")
    
    # Đường dẫn file model
    model_path = st.text_input("Đường dẫn file mô hình (.pt)", value="last.pt")
    
    # Tải ảnh test
    uploaded_file = st.file_uploader("Tải ảnh người lái lên để kiểm tra", type=['jpg', 'jpeg', 'png'])
    
    st.write("---")
    st.info("AI nhận diện 5 trạng thái: awake (tỉnh táo), drowsy (ngủ gật), phone (điện thoại), smoking (hút thuốc), yawn (ngáp).")

# 3.3 TẢI MODEL
model = load_yolov5_model(model_path)

if model:
    # Xác định ảnh nguồn
    if uploaded_file is not None:
        # Nếu người dùng upload ảnh mới
        input_image = Image.open(uploaded_file)
        image_name = uploaded_file.name
    else:
        # Sử dụng ảnh mặc định nếu có, nếu không thì cảnh báo
        default_img_path = "train_batch0.jpg"
        if os.path.exists(default_img_path):
            input_image = Image.open(default_img_path)
            image_name = default_img_path # (Mặc định)
            st.sidebar.caption(f"Đang dùng ảnh mặc định: {default_img_path}")
        else:
            st.warning("⚠️ Vui lòng tải ảnh lên từ Thanh bên (Sidebar) để bắt đầu phân tích.")
            input_image = None

    # 3.4 HIỂN THỊ KẾT QUẢ
    if input_image:
        # Chạy phân tích
        df_count, results = get_detection_statistics(model, input_image)
        
        # --- PHẦN 3.4.1: CÁC CHỈ SỐ KPI CHÍNH (METRICS) ---
        col_m1, col_m2, col_m3 = st.columns(3)
        
        danger_classes = ["drowsy", "phone", "smoking", "yawn"]
        total_detections = df_count.sum()
        safe_count = df_count.get('awake', 0)
        danger_count = df_count.reindex(danger_classes).sum()

        with col_m1:
            st.metric(label="Tổng số hành vi phát hiện", value=total_detections)
        with col_m2:
            st.metric(label="✅ Trạng thái an toàn", value=safe_count)
        with col_m3:
            # Dùng markdown để đổi màu metric Danger sang đỏ
            st.metric(label="🚨 Hành vi NGUY HIỂM", value=danger_count, delta_color="inverse")
            if danger_count > 0:
                st.error("⚠️ CẢNH BÁO: Phát hiện hành vi nguy hiểm!")
            else:
                st.success("✅ Người lái đang tỉnh táo.")
        
        st.write("---")

        # --- PHẦN 3.4.2: BIỂU ĐỒ & ẢNH NGUỒN ---
        col_img, col_chart = st.columns([1, 1]) # Chia tỷ lệ 1:1

        with col_img:
            st.subheader(f"Ảnh đầu vào: {image_name}")
            st.image(input_image, use_container_width=True)

        with col_chart:
            st.subheader("Sơ đồ thống kê hành vi cột dọc")
            
            # Kiểm tra xem có dữ liệu để vẽ không
            if total_detections > 0:
                # --- LOGIC VẼ BIỂU ĐỒ CỘT DỌC (METHOD 2 UPGRADE) ---
                
                # Thiết lập màu: Đỏ cảnh báo, Xanh an toàn
                # Sử dụng trực tiếp df_count.index (đã reindex đúng thứ tự)
                colors = ['#ff4d4d' if x in danger_classes else '#2ecc71' for x in df_count.index]

                # Tạo biểu đồ Matplotlib: Changed from ax.barh to ax.bar
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Vẽ cột dọc (bar) thay vì thanh ngang (barh)
                bars = ax.bar(df_count.index, df_count.values, color=colors, width=0.6)

                # Trang trí biểu đồ cho chuyên nghiệp
                ax.set_title("PHÂN TÍCH HÀNH VI LÁI XE", fontsize=16, fontweight='bold', pad=15)
                ax.set_ylabel("Số lần phát hiện (Detections)", fontsize=12)
                # X-label tự động lấy tên behaviors nên không cần set_xlabel rối mắt
                
                # Loại bỏ spines không cần thiết
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Thiết lập giới hạn trục Y để có khoảng trống cho số
                ax.set_ylim(0, df_count.max() + (df_count.max() * 0.1) + 0.5)

                # Thêm số liệu vào TRÊN đầu thanh cột (Changed text placement)
                for bar in bars:
                    height = bar.get_height()
                    # Chỉ vẽ số nếu giá trị lớn hơn 0
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, # Vị trí X giữa cột
                                height + (df_count.max() * 0.02),   # Vị trí Y trên đầu cột một chút
                                f'{int(height)}', 
                                va='bottom', ha='center', fontsize=12, fontweight='bold')

                plt.tight_layout()
                
                # HIỂN THỊ BIỂU ĐỒ MATPLOTLIB TRÊN STREAMLIT
                st.pyplot(fig)
            else:
                st.info("AI quét ảnh này nhưng không phát hiện bất kỳ hành vi nào trong 5 nhóm class đã học.")
