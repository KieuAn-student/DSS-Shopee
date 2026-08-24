# -*- coding: utf-8 -*-
# Giao diện Streamlit: gợi ý nhập hàng cho cửa hàng Shopee "Chút Chíu".
# Luồng: Excel -> load_data() -> dss_model (AHP + TOPSIS) -> gợi ý -> giao diện.
# Phần lý thuyết, công thức: xem Tai_lieu_thuyet_trinh_TOPSIS.docx

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import dss_model as M

DATA_FILE = Path(__file__).resolve().parent / "Quản lý sản phẩm shopee cuahangchutchiu.xlsx"
SHEET_NAME = "Danh mục hàng hóa"

REVENUE_OPTIONS = {
    "Doanh thu trong kỳ (nên dùng)": "Doanh thu hiện tại",
    "Doanh thu cộng dồn từ trước tới nay": "Tổng doanh thu",
}

GROUP_COLORS = {
    "NÊN NHẬP": "#2E7D32",
    "NHẬP VỪA": "#F9A825",
    "KHÔNG NÊN NHẬP": "#C62828",
}

# Material Symbols có sẵn trong Streamlit; đi kèm màu để đọc được ngay ý nghĩa.
GROUP_ICONS = {
    "NÊN NHẬP": ":green[:material/check_circle:]",
    "NHẬP VỪA": ":orange[:material/pause_circle:]",
    "KHÔNG NÊN NHẬP": ":red[:material/cancel:]",
}
GROUP_ICON_PLAIN = {
    "NÊN NHẬP": ":material/check_circle:",
    "NHẬP VỪA": ":material/pause_circle:",
    "KHÔNG NÊN NHẬP": ":material/cancel:",
}


# ---------------------------------------------------------------------------
# Dữ liệu
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Đang đọc dữ liệu bán hàng...")
def load_data(revenue_source: str = "Doanh thu hiện tại"):
    meta = {"file": str(DATA_FILE), "revenue_source": revenue_source}

    if not DATA_FILE.exists():
        return pd.DataFrame(), {**meta, "error": f"Không tìm thấy file {DATA_FILE.name}"}

    try:
        # header=1 vì dòng đầu sheet là tiêu đề trang trí, tên cột thật ở dòng 2.
        df_raw = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME, header=1)
    except Exception as e:
        return pd.DataFrame(), {**meta, "error": f"Không đọc được file Excel: {e}"}

    meta["n_raw"] = len(df_raw)

    needed = ["Mã hàng", "Tên quy cách", "Giá nhập", "Giá bán",
              "Xuất hiện tại", "Tồn", revenue_source]
    missing = [c for c in needed if c not in df_raw.columns]
    if missing:
        return pd.DataFrame(), {**meta, "error": f"File thiếu cột: {missing}"}

    df = df_raw[needed].copy()

    num_cols = ["Giá nhập", "Giá bán", "Xuất hiện tại", "Tồn", revenue_source]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=num_cols)
    df = df[df["Giá nhập"] > 0]
    meta["n_dropped"] = n_before - len(df)

    # Đếm trước rồi mới sửa để còn báo lại cho người dùng.
    meta["n_negative_stock"] = int((df["Tồn"] < 0).sum())
    meta["min_stock_before_fix"] = float(df["Tồn"].min()) if len(df) else 0.0
    # Tồn âm là lỗi ghi sổ; để nguyên thì TOPSIS coi đó là mức tồn lý tưởng
    # (tồn kho là tiêu chí càng thấp càng tốt) và đẩy nhầm sản phẩm lên hạng 1.
    df["Tồn"] = df["Tồn"].clip(lower=0)

    df["Lợi nhuận (VNĐ)"] = df["Giá bán"] - df["Giá nhập"]

    df = df.rename(columns={
        "Tên quy cách": "Sản phẩm",
        "Xuất hiện tại": "Đã bán (SL)",
        "Tồn": "Tồn kho (SL)",
        revenue_source: "Doanh thu (VNĐ)",
        "Giá nhập": "Giá nhập (VNĐ)",
        "Giá bán": "Giá bán (VNĐ)",
    }).reset_index(drop=True)

    meta["n_final"] = len(df)
    meta["n_zero_sold"] = int((df["Đã bán (SL)"] == 0).sum())
    meta["n_zero_profit"] = int((df["Lợi nhuận (VNĐ)"] == 0).sum())
    return df, meta


# ---------------------------------------------------------------------------
# Thanh bên
# ---------------------------------------------------------------------------

SAATY_VALUES = [1 / v for v in range(9, 1, -1)] + [float(v) for v in range(1, 10)]


def _fmt_saaty(v: float) -> str:
    return str(int(round(v))) if v >= 1 else f"1/{int(round(1 / v))}"


def _weights_chart(weights):
    d = pd.DataFrame({
        "Tiêu chí": M.CRITERIA_LABELS,
        "Tỷ lệ": weights * 100,
    }).sort_values("Tỷ lệ")
    fig = px.bar(d, x="Tỷ lệ", y="Tiêu chí", orientation="h", text_auto=".0f")
    fig.update_traces(marker_color="#1565C0", textposition="outside",
                      cliponaxis=False, hovertemplate="%{y}: %{x:.1f}%<extra></extra>")
    fig.update_layout(height=185, margin=dict(l=0, r=18, t=4, b=0),
                      yaxis_title=None, xaxis_visible=False,
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def render_sidebar():
    st.sidebar.header("Thiết lập", divider="gray")

    st.sidebar.markdown(":material/flag: **Kỳ này bạn ưu tiên điều gì?**")
    choice = st.sidebar.selectbox(
        "Mục tiêu kinh doanh",
        list(M.BUSINESS_PRESETS.keys()) + ["Tự chỉnh mức quan trọng…"],
        label_visibility="collapsed",
    )

    if choice in M.BUSINESS_PRESETS:
        st.sidebar.caption(M.PRESET_HINTS[choice])
        upper = M.BUSINESS_PRESETS[choice]
    else:
        upper = dict(M.DEFAULT_PAIRWISE_UPPER)
        with st.sidebar.expander("So sánh từng cặp tiêu chí", expanded=True):
            st.caption("Vế trái quan trọng gấp mấy lần vế phải. "
                       "Chọn 1 nếu hai bên ngang nhau.")
            n = len(M.CRITERIA)
            for i in range(n):
                for j in range(i + 1, n):
                    d = M.DEFAULT_PAIRWISE_UPPER.get((i, j), 1.0)
                    idx = int(np.argmin([abs(v - d) for v in SAATY_VALUES]))
                    upper[(i, j)] = st.select_slider(
                        f"{M.CRITERIA_LABELS[i]} so với {M.CRITERIA_LABELS[j]}",
                        options=SAATY_VALUES, value=SAATY_VALUES[idx],
                        format_func=_fmt_saaty, key=f"ahp_{i}_{j}",
                    )

    # Tính ngay tại đây để người dùng thấy trọng số đúng chỗ mình vừa chỉnh.
    ahp = M.ahp_weights(M.build_pairwise_matrix(upper, len(M.CRITERIA)))

    st.sidebar.markdown("")
    st.sidebar.markdown(":material/balance: **Mức quan trọng đang áp dụng**")
    st.sidebar.plotly_chart(_weights_chart(ahp["weights"]),
                            use_container_width=True, key="sb_weights")
    if not ahp["is_consistent"]:
        st.sidebar.caption(":red[Các mức quan trọng đang mâu thuẫn với nhau.]")

    st.sidebar.markdown("")
    st.sidebar.markdown(":material/inventory_2: **Độ dài danh sách nên nhập**")
    top_pct = st.sidebar.slider(
        "Độ dài danh sách nên nhập",
        min_value=5, max_value=40, value=20, step=5,
        label_visibility="collapsed",
        help="Vốn có hạn nên không thể nhập tất cả. Kéo thấp để chọn lọc gắt hơn.",
    )
    # Điền sau khi chạy mô hình để hiện số sản phẩm thật thay vì chỉ phần trăm.
    count_slot = st.sidebar.empty()

    with st.sidebar.expander("Tùy chọn nâng cao", icon=":material/tune:"):
        only_sold = st.checkbox(
            "Bỏ qua sản phẩm chưa bán được đơn nào",
            help="Hàng chưa bán thường có tồn kho bằng 0 nên vô tình được chấm "
                 "điểm cao dù chưa chứng minh được gì.",
        )
        rev_label = st.selectbox(
            "Tính doanh thu theo", list(REVENUE_OPTIONS.keys()),
            help="Doanh thu trong kỳ cùng mốc thời gian với số lượng đã bán.",
        )

    st.sidebar.divider()
    st.sidebar.caption("Hệ thống chỉ đưa ra gợi ý. "
                       "Người quyết định cuối cùng vẫn là bạn.")

    return {
        "ahp": ahp,
        "top_pct": top_pct / 100,
        "top_pct_raw": top_pct,
        "count_slot": count_slot,
        "only_sold": only_sold,
        "revenue_source": REVENUE_OPTIONS[rev_label],
    }


# ---------------------------------------------------------------------------
# Tab: gợi ý nhập hàng
# ---------------------------------------------------------------------------

def tab_recommend(res):
    st.markdown(":material/star: **Năm sản phẩm đáng nhập nhất**")

    for _, r in res.head(5).iterrows():
        with st.container(border=True):
            c1, c2, c3 = st.columns([5, 2, 2])
            c1.markdown(
                f"**{int(r['Xếp hạng'])}. {r['Sản phẩm']}**  \n"
                f"{GROUP_ICONS[r['Khuyến nghị']]} {r['Khuyến nghị']} "
                f"· Mã {r['Mã hàng']}"
            )
            c2.metric("Lời mỗi đơn", f"{r['Lợi nhuận (VNĐ)']:,.0f} đ")
            c3.metric("Đã bán / Tồn",
                      f"{r['Đã bán (SL)']:.0f} / {r['Tồn kho (SL)']:.0f}")

    st.markdown("")
    st.markdown(":material/list: **Toàn bộ danh mục**")

    c1, c2 = st.columns([3, 2])
    with c1:
        pick = st.multiselect(
            "Lọc nhóm", M.RECOMMEND_LABELS, default=M.RECOMMEND_LABELS,
            label_visibility="collapsed", placeholder="Chọn nhóm muốn xem",
        )
    with c2:
        kw = st.text_input("Tìm sản phẩm", placeholder="Tìm theo tên hoặc mã",
                           label_visibility="collapsed", icon=":material/search:")

    view = res[res["Khuyến nghị"].isin(pick)] if pick else res
    if kw.strip():
        k = kw.strip().lower()
        view = view[view["Sản phẩm"].str.lower().str.contains(k, na=False)
                    | view["Mã hàng"].astype(str).str.lower().str.contains(k, na=False)]

    if view.empty:
        st.info("Không có sản phẩm nào khớp. Thử bỏ bớt bộ lọc hoặc đổi từ khóa.",
                icon=":material/search_off:")
    else:
        st.dataframe(
            view[["Xếp hạng", "Mã hàng", "Sản phẩm", "Lợi nhuận (VNĐ)",
                  "Đã bán (SL)", "Tồn kho (SL)", "Điểm", "Khuyến nghị",
                  "Nên làm gì"]],
            use_container_width=True, hide_index=True, height=340,
            column_config={
                "Xếp hạng": st.column_config.NumberColumn("Hạng", width="small"),
                "Lợi nhuận (VNĐ)": st.column_config.NumberColumn(
                    "Lời/đơn", format="%,.0f đ"),
                "Đã bán (SL)": st.column_config.NumberColumn("Đã bán", format="%.0f"),
                "Tồn kho (SL)": st.column_config.NumberColumn("Tồn", format="%.0f"),
                "Điểm": st.column_config.ProgressColumn(
                    "Điểm", min_value=0.0, max_value=1.0, format="%.3f",
                    help="Càng gần 1 càng đáng nhập."),
            },
        )
        st.caption(f"Đang hiện {len(view)} trên tổng {len(res)} sản phẩm.")

    c1, _ = st.columns([1, 3])
    with c1:
        st.download_button(
            "Tải danh sách", icon=":material/download:",
            data=res.to_csv(index=False).encode("utf-8-sig"),
            file_name="goi_y_nhap_hang.csv", mime="text/csv",
            use_container_width=True,
        )

    st.markdown("")
    fig = px.bar(
        res.head(12).iloc[::-1], x="Điểm", y="Sản phẩm", orientation="h",
        color="Khuyến nghị", color_discrete_map=GROUP_COLORS,
        title="12 sản phẩm đứng đầu", text_auto=".3f",
    )
    fig.update_layout(height=430, yaxis_title=None, xaxis_title="Điểm",
                      legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("")
    st.markdown(":material/scatter_plot: **Toàn cảnh danh mục**")
    st.caption("Mỗi chấm là một sản phẩm. Chấm càng to thì doanh thu càng lớn. "
               "Góc trên bên phải là vùng vừa lời nhiều vừa bán chạy.")
    fig2 = px.scatter(
        res, x="Đã bán (SL)", y="Lợi nhuận (VNĐ)",
        size="Doanh thu (VNĐ)", color="Khuyến nghị",
        color_discrete_map=GROUP_COLORS, hover_name="Sản phẩm",
        hover_data={"Mã hàng": True, "Tồn kho (SL)": ":.0f", "Điểm": ":.3f",
                    "Đã bán (SL)": ":.0f", "Lợi nhuận (VNĐ)": ":,.0f",
                    "Doanh thu (VNĐ)": ":,.0f"},
        size_max=32, opacity=0.75,
    )
    fig2.update_layout(height=460, legend_title_text="",
                       xaxis_title="Số lượng đã bán trong kỳ",
                       yaxis_title="Lợi nhuận mỗi đơn (VNĐ)")
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: lựa chọn an toàn
# ---------------------------------------------------------------------------

def tab_safe(df, weights, topk=10):
    st.markdown(":material/verified: **Những sản phẩm đáng nhập dù bạn đổi ưu tiên**")
    st.caption("Hệ thống thử lại nhiều mức ưu tiên khác nhau và giữ lại những "
               "sản phẩm gần như luôn nằm trong nhóm đầu.")

    X = df[M.CRITERIA_KEYS].to_numpy(dtype=float)
    base_rank = M.rank_scores(M.topsis(X, weights, M.CRITERIA_TYPES)["scores"])

    # Quét trọng số từng tiêu chí 5%-55% cho cả 5 tiêu chí = 60 kịch bản. Dừng ở
    # 55% vì dồn quá nửa mức quan trọng vào một tiêu chí là tình huống phi thực tế.
    grid = np.round(np.arange(0.05, 0.56, 0.05), 2)
    STABLE_PCT = 80

    with st.spinner("Đang thử các mức ưu tiên khác nhau..."):
        tables = [M.sensitivity_one_at_a_time(X, weights, M.CRITERIA_TYPES, c, grid)
                  for c in range(len(M.CRITERIA))]
    stab = M.stability_top_k(pd.concat(tables, ignore_index=True), k=topk)
    rock = stab[stab >= STABLE_PCT].sort_values(ascending=False).index.tolist()

    if not rock:
        st.warning(
            f"Không sản phẩm nào trụ vững ở nhóm {topk} đầu. Thứ hạng phụ thuộc "
            f"nhiều vào mục tiêu bạn chọn, nên cân nhắc kỹ trước khi nhập.",
            icon=":material/warning:",
        )
        return

    st.success(
        f"{len(rock)} sản phẩm vẫn nằm trong nhóm {topk} đầu ở hầu hết các mức "
        f"ưu tiên. Đây là những lựa chọn an toàn nhất.",
        icon=":material/verified:",
    )

    safe = pd.DataFrame({
        "Hạng": [base_rank[i] for i in rock],
        "Mã hàng": [df.loc[i, "Mã hàng"] for i in rock],
        "Sản phẩm": [df.loc[i, "Sản phẩm"] for i in rock],
        "Độ ổn định": [stab[i] for i in rock],
        "Lời mỗi đơn": [df.loc[i, "Lợi nhuận (VNĐ)"] for i in rock],
        "Đã bán": [df.loc[i, "Đã bán (SL)"] for i in rock],
    }).sort_values("Hạng")

    st.dataframe(
        safe, use_container_width=True, hide_index=True,
        column_config={
            "Độ ổn định": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.0f%%",
                help="Tỷ lệ số lần sản phẩm vẫn nằm trong nhóm đầu."),
            "Lời mỗi đơn": st.column_config.NumberColumn(format="%,.0f đ"),
            "Đã bán": st.column_config.NumberColumn(format="%.0f"),
        },
    )

    top_stab = stab.sort_values(ascending=False).head(12)
    bar = pd.DataFrame({
        "Sản phẩm": [f"{df.loc[i, 'Mã hàng']} · {df.loc[i, 'Sản phẩm'][:24]}"
                     for i in top_stab.index],
        "Độ ổn định": top_stab.to_numpy(),
    })
    fig = px.bar(bar.iloc[::-1], x="Độ ổn định", y="Sản phẩm", orientation="h",
                 range_x=[0, 100], text_auto=".0f",
                 color="Độ ổn định", color_continuous_scale="Greens")
    fig.update_layout(height=420, yaxis_title=None, coloraxis_showscale=False,
                      xaxis_title="Tỷ lệ vẫn nằm trong nhóm đầu (%)")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: dữ liệu
# ---------------------------------------------------------------------------

def tab_data(df, meta, weights):
    st.markdown(":material/database: **Nguồn dữ liệu**")
    st.caption(f"Lấy từ file quản lý bán hàng của cửa hàng - {DATA_FILE.name}, "
               f"trang {SHEET_NAME}. File gốc không bị sửa đổi.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Dòng đọc từ file", meta.get("n_raw", 0))
    c2.metric("Sản phẩm đưa vào tính", meta.get("n_final", 0))
    c3.metric("Dòng bỏ vì thiếu dữ liệu", meta.get("n_dropped", 0))

    st.markdown("")
    st.markdown(":material/tune: **Tiêu chí chấm điểm và mức quan trọng đang áp dụng**")
    st.dataframe(
        pd.DataFrame({
            "Tiêu chí": M.CRITERIA_LABELS,
            "Chiều tốt": ["Càng cao càng tốt" if c["type"] == 1
                          else "Càng thấp càng tốt" for c in M.CRITERIA],
            "Mức quan trọng": weights * 100,
            "Nghĩa là gì": [c["help"] for c in M.CRITERIA],
        }),
        use_container_width=True, hide_index=True,
        column_config={
            "Mức quan trọng": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f%%"),
        },
    )

    st.markdown("")
    st.markdown(":material/table_chart: **Bảng dữ liệu đang dùng**")
    st.dataframe(df[["Mã hàng", "Sản phẩm"] + M.CRITERIA_KEYS],
                 use_container_width=True, hide_index=True, height=300)

    notes = [
        "Lợi nhuận là phần chênh giữa giá bán và giá nhập, "
        "chưa trừ phí sàn Shopee, phí vận chuyển và tiền quảng cáo.",
        f"{meta.get('n_zero_sold', 0)} sản phẩm chưa bán được đơn nào trong kỳ. "
        f"Có thể bỏ qua nhóm này trong Tùy chọn nâng cao.",
    ]
    if meta.get("n_negative_stock", 0):
        notes.append(
            f"{meta['n_negative_stock']} sản phẩm có tồn kho âm do lỗi ghi sổ, "
            f"đã được đưa về 0 trước khi chấm điểm."
        )
    st.info("Cần lưu ý:\n\n" + "\n".join(f"- {n}" for n in notes),
            icon=":material/info:")


# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Gợi ý nhập hàng Shopee", layout="wide",
                       page_icon=":material/storefront:",
                       initial_sidebar_state="expanded")

    st.title("Trợ lý nhập hàng - Cửa hàng Chút Chíu")
    st.caption("Chấm điểm toàn bộ sản phẩm đang bán theo lợi nhuận, sức bán, "
               "doanh thu, tồn kho và giá nhập, rồi gợi ý kỳ tới nên nhập thêm "
               "gì, giữ nguyên gì và nên dừng gì.")

    cfg = render_sidebar()
    df, meta = load_data(cfg["revenue_source"])

    if "error" in meta:
        st.error(meta["error"], icon=":material/error:")
        st.info(f"Hãy đặt file Excel vào cùng thư mục với ứng dụng: "
                f"{DATA_FILE.parent}", icon=":material/folder:")
        st.stop()
    if df.empty:
        st.error("Không còn dòng dữ liệu hợp lệ nào sau khi dọn dẹp.",
                 icon=":material/error:")
        st.stop()

    if cfg["only_sold"]:
        df = df[df["Đã bán (SL)"] > 0].reset_index(drop=True)
        if df.empty:
            st.error("Không còn sản phẩm nào sau khi lọc. Hãy tắt bớt bộ lọc.",
                     icon=":material/error:")
            st.stop()

    ahp = cfg["ahp"]
    weights = ahp["weights"]

    X = df[M.CRITERIA_KEYS].to_numpy(dtype=float)
    try:
        out = M.topsis(X, weights, M.CRITERIA_TYPES)
    except ValueError as e:
        st.error(f"Không tính được kết quả: {e}", icon=":material/error:")
        st.stop()

    scores = out["scores"]
    labels, _ = M.classify_by_percentile(scores, cfg["top_pct"], 0.50)

    res = df[["Mã hàng", "Sản phẩm"] + M.CRITERIA_KEYS].copy()
    res["Điểm"] = scores
    res["Xếp hạng"] = M.rank_scores(scores)
    res["Khuyến nghị"] = labels
    res["Nên làm gì"] = [M.RECOMMEND_ACTIONS[l] for l in labels]
    res = res.sort_values("Xếp hạng").reset_index(drop=True)

    # Trọng số tự mâu thuẫn thì xếp hạng không đáng tin, phải báo cho người dùng.
    if not ahp["is_consistent"]:
        st.warning(
            "Các mức quan trọng bạn đặt đang mâu thuẫn với nhau nên kết quả có "
            "thể không đáng tin. Hãy chỉnh lại, hoặc chọn một mục tiêu có sẵn.",
            icon=":material/warning:",
        )

    counts = res["Khuyến nghị"].value_counts()
    cfg["count_slot"].caption(
        f"{int(counts.get('NÊN NHẬP', 0))} trong tổng {len(res)} sản phẩm "
        f"({cfg['top_pct_raw']}% danh mục)."
    )

    c1, c2, c3 = st.columns(3)
    for col, lb, text in [(c1, "NÊN NHẬP", "Nên nhập thêm"),
                          (c2, "NHẬP VỪA", "Nhập cầm chừng"),
                          (c3, "KHÔNG NÊN NHẬP", "Nên dừng nhập")]:
        col.metric(f"{GROUP_ICONS[lb]} {text}",
                   f"{int(counts.get(lb, 0))} sản phẩm")

    t1, t2, t3 = st.tabs([
        ":material/shopping_cart: Gợi ý nhập hàng",
        ":material/verified: Lựa chọn an toàn",
        ":material/database: Dữ liệu",
    ])
    with t1:
        tab_recommend(res)
    with t2:
        tab_safe(df, weights)
    with t3:
        tab_data(df, meta, weights)


if __name__ == "__main__":
    main()
