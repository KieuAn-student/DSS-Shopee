import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Cấu hình trang
st.set_page_config(page_title="Hệ Hỗ Trợ Ra Quyết Định - Cửa hàng Chút Chíu", layout="wide")

st.title("📊 Hệ Hỗ Trợ Ra Quyết Định (DSS) - Phân Tích Sản Phẩm Shopee")
st.markdown("Ứng dụng xếp hạng các sản phẩm của **cuahangchutchiu** để đưa ra chiến lược kinh doanh tối ưu dựa trên phương pháp **TOPSIS**.")

@st.cache_data
def load_data():
    file_path = 'Quản lý sản phẩm shopee cuahangchutchiu.xlsx'
    try:
        # Load the raw data
        df_raw = pd.read_excel(file_path, sheet_name='Danh mục hàng hóa', header=1)
        
        # Select relevant columns and clean
        df = df_raw[['Mã hàng', 'Tên quy cách', 'Giá nhập', 'Giá bán', 'Xuất hiện tại', 'Tồn', 'Tổng doanh thu']].copy()
        
        # Chuyển về dạng số và loại bỏ các dòng bị rỗng dữ liệu
        for col in ['Giá nhập', 'Giá bán', 'Xuất hiện tại', 'Tồn', 'Tổng doanh thu']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Giá nhập', 'Giá bán', 'Xuất hiện tại', 'Tồn', 'Tổng doanh thu'])
        
        # Calculate Profit
        df['Lợi nhuận (VNĐ)'] = df['Giá bán'] - df['Giá nhập']
        
        # Rename for clarity
        df = df.rename(columns={
            'Tên quy cách': 'Sản phẩm',
            'Xuất hiện tại': 'Đã bán (SL)',
            'Tồn': 'Tồn kho (SL)',
            'Tổng doanh thu': 'Doanh thu (VNĐ)',
            'Giá nhập': 'Giá nhập (VNĐ)'
        })
        
        # Filter products with reasonable data (optional) to avoid infinity
        df = df[df['Giá nhập (VNĐ)'] > 0]
        
        # Reset index
        df = df.reset_index(drop=True)
        return df
        
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # ================= SIDEBAR: TRỌNG SỐ =================
    st.sidebar.header("⚙️ Bảng Điều Khiển Trọng Số")
    st.sidebar.markdown("Điều chỉnh mức độ quan trọng của các tiêu chí kinh doanh.")

    w_profit = st.sidebar.slider("Lợi nhuận trên 1 SP (Càng cao càng tốt)", 1, 10, 8)
    w_sold = st.sidebar.slider("Số lượng đã bán (Càng cao càng tốt)", 1, 10, 7)
    w_revenue = st.sidebar.slider("Tổng doanh thu (Càng cao càng tốt)", 1, 10, 6)
    w_inventory = st.sidebar.slider("Tồn kho (Càng thấp càng tốt)", 1, 10, 5)
    w_cost = st.sidebar.slider("Giá nhập (Càng thấp càng tốt)", 1, 10, 4)

    weights = np.array([w_profit, w_sold, w_revenue, w_inventory, w_cost], dtype=float)
    weights = weights / weights.sum()

    # 1: Benefit (Maximize), 0: Cost (Minimize)
    criteria_types = np.array([1, 1, 1, 0, 0])
    
    # Cột dùng để tính toán
    calc_cols = ['Lợi nhuận (VNĐ)', 'Đã bán (SL)', 'Doanh thu (VNĐ)', 'Tồn kho (SL)', 'Giá nhập (VNĐ)']

    # ================= GIAO DIỆN CHÍNH =================
    st.subheader("1. Dữ liệu Danh mục Sản phẩm")
    st.dataframe(df[['Mã hàng', 'Sản phẩm'] + calc_cols], use_container_width=True)

    # 3. Thuật toán TOPSIS
    def run_topsis(data_df, cols, w, ctypes):
        matrix = data_df[cols].values.astype(float)
        
        # Xử lý các giá trị 0 hoặc âm trong ma trận để tránh lỗi chia cho 0
        matrix = np.where(matrix == 0, 1e-9, matrix)
        
        # Bước 1: Chuẩn hóa Vector
        norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
        
        # Bước 2: Nhân trọng số
        weighted_matrix = norm_matrix * w
        
        # Bước 3: Tìm A+ và A-
        ideal_best = np.zeros(len(cols))
        ideal_worst = np.zeros(len(cols))
        
        for i in range(len(cols)):
            if ctypes[i] == 1: # Benefit
                ideal_best[i] = np.max(weighted_matrix[:, i])
                ideal_worst[i] = np.min(weighted_matrix[:, i])
            else: # Cost
                ideal_best[i] = np.min(weighted_matrix[:, i])
                ideal_worst[i] = np.max(weighted_matrix[:, i])
                
        # Bước 4: Tính khoảng cách
        dist_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
        dist_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
        
        # Bước 5: Tính Closeness Score
        scores = dist_worst / (dist_best + dist_worst + 1e-9) # tránh chia 0
        return scores

    scores = run_topsis(df, calc_cols, weights, criteria_types)
    
    df_results = df[['Mã hàng', 'Sản phẩm'] + calc_cols].copy()
    df_results["Điểm TOPSIS"] = scores
    df_results["Xếp hạng"] = df_results["Điểm TOPSIS"].rank(ascending=False, na_option='bottom').fillna(9999).astype(int)

    def get_recommendation(score):
        if score >= 0.7: return "Đẩy mạnh Marketing"
        elif score >= 0.4: return "Duy trì bán hàng"
        else: return "Xả hàng / Cắt giảm"

    df_results["Khuyến nghị"] = df_results["Điểm TOPSIS"].apply(get_recommendation)
    df_results = df_results.sort_values(by="Xếp hạng")

    st.subheader("2. Bảng Xếp Hạng & Khuyến Nghị Sản Phẩm")
    st.dataframe(df_results.style.background_gradient(subset=['Điểm TOPSIS'], cmap='YlGnBu'), use_container_width=True)

    st.subheader("3. Phân Tích Trực Quan")
    col1, col2 = st.columns([2, 1])

    with col1:
        # Biểu đồ Top 15 sản phẩm
        top_15 = df_results.head(15)
        fig1 = px.bar(top_15, x='Sản phẩm', y='Điểm TOPSIS', color='Điểm TOPSIS',
                     title="Top 15 Sản phẩm Đáng đầu tư nhất",
                     color_continuous_scale='YlGnBu', text_auto='.3f')
        fig1.update_xaxes(tickangle=45)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        labels = ["Lợi nhuận", "Số lượng bán", "Doanh thu", "Tồn kho (Cost)", "Giá nhập (Cost)"]
        fig2 = px.pie(values=weights, names=labels, title="Trọng số Tiêu chí", hole=0.3)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("4. Gợi ý từ Hệ thống")
    top_product = df_results.iloc[0]
    st.success(f"🌟 **Sản phẩm xuất sắc nhất:** [{top_product['Mã hàng']}] - **{top_product['Sản phẩm']}** \n\n"
               f"Đạt điểm TOPSIS **{top_product['Điểm TOPSIS']:.4f}**. Hệ thống đề xuất tăng ngân sách quảng cáo và ưu tiên nhập thêm hàng cho sản phẩm này.")
    
    worst_product = df_results.iloc[-1]
    st.warning(f"⚠️ **Sản phẩm rủi ro cao:** [{worst_product['Mã hàng']}] - **{worst_product['Sản phẩm']}** \n\n"
               f"Đạt điểm TOPSIS **{worst_product['Điểm TOPSIS']:.4f}**. Hệ thống đề xuất xả kho, khuyến mãi giảm giá và hạn chế nhập thêm.")
