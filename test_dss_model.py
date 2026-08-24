# -*- coding: utf-8 -*-
"""
test_dss_model.py - KIỂM THỬ MODULE MÔ HÌNH CỦA THÀNH VIÊN 3
=============================================================

Mục đích: chứng minh bằng số rằng phần AHP và TOPSIS được cài đặt ĐÚNG công
thức, thay vì chỉ nói miệng "em làm theo lý thuyết". Khi giảng viên hỏi
"làm sao biết code đúng?", chạy file này là câu trả lời.

Cách chạy:
    python test_dss_model.py
(hoặc: pytest test_dss_model.py -v)

Tất cả các ca kiểm thử đều so với kết quả TÍNH TAY được, hoặc với tính chất
toán học bắt buộc phải đúng của phương pháp.
"""

import sys
import io

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd

import dss_model as M


# =============================================================================
# NHÓM 1 - KIỂM THỬ AHP
# =============================================================================

def test_ahp_ma_tran_nhat_quan_tuyet_doi():
    """Nếu dựng ma trận từ chính bộ trọng số (a_ij = w_i / w_j) thì đó là ma
    trận nhất quán tuyệt đối: lambda_max = n, CI = 0, CR = 0, và AHP phải khôi
    phục lại đúng bộ trọng số ban đầu."""
    w_that = np.array([0.4, 0.3, 0.2, 0.1])
    A = w_that[:, None] / w_that[None, :]

    r = M.ahp_weights(A)
    assert abs(r["lambda_max"] - 4) < 1e-9, r["lambda_max"]
    assert abs(r["CI"]) < 1e-9
    assert abs(r["CR"]) < 1e-9
    assert r["is_consistent"]
    assert np.allclose(r["weights"], w_that, atol=1e-9), r["weights"]


def test_ahp_hai_phuong_phap_cho_ket_qua_gan_nhau():
    """Vector riêng và trung bình nhân là hai cách tính trọng số khác nhau;
    với ma trận đủ nhất quán, chúng phải gần trùng nhau."""
    A = M.build_pairwise_matrix(M.DEFAULT_PAIRWISE_UPPER, 5)
    w_eig = M.ahp_weights(A, method="eigenvector")["weights"]
    w_geo = M.ahp_weights(A, method="geometric")["weights"]
    assert np.max(np.abs(w_eig - w_geo)) < 0.01, (w_eig, w_geo)


def test_ahp_ma_tran_mac_dinh_cua_nhom_dat_CR():
    """Ma trận so sánh cặp mặc định mà nhóm dùng phải đạt CR < 0.1.
    Đây là con số phải thuộc lòng khi thuyết trình."""
    A = M.build_pairwise_matrix(M.DEFAULT_PAIRWISE_UPPER, 5)
    r = M.ahp_weights(A)

    assert r["n"] == 5
    assert abs(r["RI"] - 1.12) < 1e-9          # RI tra bảng Saaty với n = 5
    assert r["CR"] < 0.1, f"CR = {r['CR']:.4f} không đạt"
    assert abs(r["weights"].sum() - 1.0) < 1e-9

    # Trọng số phải giữ đúng thứ tự ưu tiên nhóm đã thống nhất:
    # Lợi nhuận > (Đã bán ~ Doanh thu) > Tồn kho > Giá nhập
    w = r["weights"]
    assert w[0] == max(w), "Lợi nhuận phải có trọng số cao nhất"
    assert w[4] == min(w), "Giá nhập phải có trọng số thấp nhất"
    assert w[3] > w[4]

    print(f"    -> CR = {r['CR']:.4f} | lambda_max = {r['lambda_max']:.4f} "
          f"| CI = {r['CI']:.4f}")
    print("    -> Trọng số:", dict(zip(M.CRITERIA_LABELS, np.round(w, 4))))


def test_moi_muc_tieu_kinh_doanh_deu_dat_CR():
    """Giao diện cho người dùng chọn 'mục tiêu kinh doanh' bằng tiếng Việt thay
    vì tự cho điểm 10 cặp tiêu chí. Mỗi mục tiêu quy đổi ra một ma trận AHP, nên
    TẤT CẢ đều bắt buộc phải đạt CR < 0.1 - nếu không, người dùng vừa chọn xong
    là app đã báo lỗi ngay."""
    assert len(M.BUSINESS_PRESETS) >= 3
    for ten, upper in M.BUSINESS_PRESETS.items():
        A = M.build_pairwise_matrix(upper, len(M.CRITERIA))
        r = M.ahp_weights(A)
        assert r["CR"] < 0.1, f"Mục tiêu '{ten}' có CR = {r['CR']:.4f} - không đạt"
        assert abs(r["weights"].sum() - 1.0) < 1e-9
        assert ten in M.PRESET_HINTS, f"Mục tiêu '{ten}' thiếu câu mô tả"
        print(f"    -> {ten[:38]:40s} CR = {r['CR']:.4f}")


def test_muc_tieu_kinh_doanh_uu_tien_dung_tieu_chi():
    """Mỗi mục tiêu phải thực sự đẩy trọng số về đúng tiêu chí mà tên nó hứa hẹn,
    nếu không thì nhãn tiếng Việt đang nói dối người dùng."""
    ky_vong = {
        "Kiếm lời nhiều nhất trên mỗi đơn": 0,   # Lợi nhuận
        "Chạy theo hàng bán chạy": 1,            # Số lượng bán
        "Tránh ôm thêm hàng đang tồn": 3,        # Tồn kho
        "Ít vốn, xoay vòng nhanh": 4,            # Giá nhập
    }
    for ten, idx in ky_vong.items():
        w = M.ahp_weights(M.build_pairwise_matrix(M.BUSINESS_PRESETS[ten], 5))["weights"]
        assert int(np.argmax(w)) == idx, (
            f"'{ten}' đáng lẽ ưu tiên '{M.CRITERIA_LABELS[idx]}' "
            f"nhưng lại ưu tiên '{M.CRITERIA_LABELS[int(np.argmax(w))]}'"
        )


def test_vi_tri_tung_tieu_chi():
    """criterion_positions phải cho 1 với phương án tốt nhất của mỗi tiêu chí và
    0 với phương án tệ nhất - kể cả tiêu chí CHI PHÍ (nơi 'tốt nhất' là nhỏ nhất)."""
    # Cột 0 là lợi ích, cột 1 là chi phí.
    X = np.array([[10.0, 5.0],
                  [5.0, 10.0],
                  [1.0, 1.0]])
    out = M.topsis(X, np.array([0.5, 0.5]), np.array([1, 0]))
    pos = M.criterion_positions(out)

    assert pos.shape == X.shape
    assert (pos >= 0).all() and (pos <= 1).all()

    # Tiêu chí lợi ích: giá trị 10 là tốt nhất -> 1; giá trị 1 là tệ nhất -> 0.
    assert abs(pos[0, 0] - 1.0) < 1e-9
    assert abs(pos[2, 0] - 0.0) < 1e-9
    # Tiêu chí chi phí: giá trị 1 mới là tốt nhất -> 1; giá trị 10 là tệ -> 0.
    assert abs(pos[2, 1] - 1.0) < 1e-9
    assert abs(pos[1, 1] - 0.0) < 1e-9


def test_vi_tri_tieu_chi_khi_moi_phuong_an_bang_nhau():
    """Tiêu chí mà mọi sản phẩm bằng nhau thì không phân biệt được ai hơn ai,
    phải trả 0.5 chứ không được chia cho 0."""
    X = np.array([[5.0, 7.0], [9.0, 7.0], [2.0, 7.0]])
    out = M.topsis(X, np.array([0.5, 0.5]), np.array([1, 1]))
    pos = M.criterion_positions(out)
    assert np.allclose(pos[:, 1], 0.5)
    assert not np.isnan(pos).any()


def test_ahp_ma_tran_khong_nhat_quan_bi_bat_loi():
    """Ma trận mâu thuẫn (A hơn B, B hơn C, nhưng C lại hơn A rất nhiều)
    phải cho CR lớn -> hệ thống phải báo KHÔNG ĐẠT."""
    A = M.build_pairwise_matrix({(0, 1): 9, (0, 2): 1 / 9, (1, 2): 9}, 3)
    r = M.ahp_weights(A)
    assert r["CR"] > 0.1, r["CR"]
    assert not r["is_consistent"]


def test_ahp_tinh_nghich_dao():
    """Ma trận dựng ra phải thỏa a_ji = 1 / a_ij và đường chéo bằng 1."""
    A = M.build_pairwise_matrix(M.DEFAULT_PAIRWISE_UPPER, 5)
    assert np.allclose(np.diag(A), 1.0)
    assert np.allclose(A * A.T, 1.0)


# =============================================================================
# NHÓM 2 - KIỂM THỬ TOPSIS
# =============================================================================

def test_topsis_vi_du_tinh_tay():
    """Ví dụ tính tay được hoàn toàn.

    X = [[4,4],[2,2],[1,1]], cả 2 tiêu chí đều BENEFIT, trọng số bằng nhau.

      Chuẩn cột      = sqrt(4^2 + 2^2 + 1^2) = sqrt(21)  (giống nhau cả 2 cột)
      V              = 0.5 * X / sqrt(21)
      A+ = (2/sqrt21, 2/sqrt21)   A- = (0.5/sqrt21, 0.5/sqrt21)

      P.án 1 trùng A+  -> D+ = 0            -> C* = 1
      P.án 3 trùng A-  -> D- = 0            -> C* = 0
      P.án 2: D+ = sqrt(2/21), D- = sqrt(0.5/21)
              C* = sqrt(0.5) / (sqrt(2) + sqrt(0.5)) = 1/3
    """
    X = np.array([[4.0, 4.0], [2.0, 2.0], [1.0, 1.0]])
    w = np.array([0.5, 0.5])
    t = np.array([1, 1])

    out = M.topsis(X, w, t)
    expected = np.array([1.0, 1.0 / 3.0, 0.0])
    assert np.allclose(out["scores"], expected, atol=1e-12), out["scores"]

    # Kiểm tra luôn các bước trung gian so với tính tay
    s21 = np.sqrt(21.0)
    assert np.allclose(out["normalized"][:, 0], np.array([4, 2, 1]) / s21)
    assert np.allclose(out["ideal_best"], [2 / s21, 2 / s21])
    assert np.allclose(out["ideal_worst"], [0.5 / s21, 0.5 / s21])
    print("    -> Điểm C* tính bằng code khớp tuyệt đối với tính tay:", out["scores"])


def test_topsis_diem_luon_trong_khoang_0_1():
    """Theo định nghĩa C* = D- / (D+ + D-) với D+, D- >= 0 nên C* luôn thuộc [0,1]."""
    rng = np.random.default_rng(42)          # cố định seed -> tái lập được
    for _ in range(200):
        m = rng.integers(2, 30)
        n = rng.integers(2, 8)
        X = rng.random((m, n)) * 1000
        w = rng.random(n)
        t = rng.integers(0, 2, n)
        sc = M.topsis(X, w, t)["scores"]
        assert (sc >= -1e-12).all() and (sc <= 1 + 1e-12).all(), sc


def test_topsis_huong_tieu_chi_chi_phi():
    """Với DUY NHẤT một tiêu chí CHI PHÍ, phương án có giá trị nhỏ nhất phải
    xếp hạng 1. Đây chính là chỗ dễ cài sai dấu nhất."""
    X = np.array([[10.0], [50.0], [30.0]])
    sc = M.topsis(X, np.array([1.0]), np.array([0]))["scores"]
    assert M.rank_scores(sc)[0] == 1, sc
    assert sc[0] > sc[2] > sc[1]

    # Đảo thành tiêu chí LỢI ÍCH thì thứ tự phải lật ngược lại.
    sc_b = M.topsis(X, np.array([1.0]), np.array([1]))["scores"]
    assert M.rank_scores(sc_b)[1] == 1, sc_b


def test_topsis_bat_bien_khi_nhan_trong_so_voi_hang_so():
    """Trọng số [2,3,5] và [4,6,10] mô tả CÙNG một ưu tiên, nên điểm phải y hệt
    (vì hàm tự chuẩn hóa tổng trọng số về 1)."""
    rng = np.random.default_rng(7)
    X = rng.random((12, 3)) * 100
    t = np.array([1, 0, 1])
    a = M.topsis(X, np.array([2.0, 3.0, 5.0]), t)["scores"]
    b = M.topsis(X, np.array([4.0, 6.0, 10.0]), t)["scores"]
    assert np.allclose(a, b)


def test_topsis_cot_hang_so_khong_anh_huong_xep_hang():
    """Một tiêu chí mà mọi sản phẩm đều bằng nhau thì không phân biệt được ai
    hơn ai -> thêm cột đó vào không được làm đổi thứ hạng."""
    rng = np.random.default_rng(3)
    X = rng.random((10, 2)) * 50
    t = np.array([1, 0])
    r1 = M.rank_scores(M.topsis(X, np.array([0.5, 0.5]), t)["scores"])

    X2 = np.hstack([X, np.full((10, 1), 7.0)])
    t2 = np.array([1, 0, 1])
    r2 = M.rank_scores(M.topsis(X2, np.array([0.45, 0.45, 0.10]), t2)["scores"])
    assert (r1 == r2).all(), (r1, r2)


def test_topsis_chan_du_lieu_khong_hop_le():
    """Các lá chắn đầu vào phải hoạt động - không được âm thầm cho ra số sai."""
    w = np.array([0.5, 0.5]); t = np.array([1, 0])

    for X, mo_ta in [
        (np.array([[1.0, -2.0], [3.0, 4.0]]), "giá trị âm"),
        (np.array([[1.0, np.nan], [3.0, 4.0]]), "giá trị NaN"),
        (np.array([[1.0, 2.0]]), "chỉ có 1 phương án"),
    ]:
        try:
            M.topsis(X, w, t)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Phải báo lỗi với trường hợp: {mo_ta}")

    # Sai số chiều trọng số
    try:
        M.topsis(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0]), t)
    except ValueError:
        pass
    else:
        raise AssertionError("Phải báo lỗi khi số trọng số không khớp số tiêu chí")


def test_rank_scores_xu_ly_diem_bang_nhau():
    """Hai sản phẩm cùng điểm phải nhận cùng hạng NGUYÊN (không ra hạng .5)."""
    r = M.rank_scores(np.array([0.9, 0.5, 0.5, 0.1]))
    assert r.tolist() == [1, 2, 2, 4], r.tolist()
    assert r.dtype.kind in "iu"


# =============================================================================
# NHÓM 3 - KIỂM THỬ NGƯỠNG KHUYẾN NGHỊ
# =============================================================================

def test_phan_vi_chia_dung_ty_le():
    """Chế độ phân vị phải cho ra đúng khoảng 20% / 50% / 30%."""
    scores = np.linspace(0, 1, 200)
    labels, info = M.classify_by_percentile(scores, top_pct=0.20, mid_pct=0.50)
    vc = pd.Series(labels).value_counts()
    assert abs(vc["NÊN NHẬP"] - 40) <= 2, vc.to_dict()
    assert abs(vc["NHẬP VỪA"] - 100) <= 2, vc.to_dict()
    assert abs(vc["KHÔNG NÊN NHẬP"] - 60) <= 2, vc.to_dict()
    assert info["mode"] == "percentile"


def test_nguong_tuyet_doi_dung_moc():
    labels, _ = M.classify_by_threshold(np.array([0.9, 0.5, 0.1]), 0.7, 0.4)
    assert labels.tolist() == ["NÊN NHẬP", "NHẬP VỪA", "KHÔNG NÊN NHẬP"]


def test_moi_nhan_deu_co_hanh_dong_kem_theo():
    for lb in M.RECOMMEND_LABELS:
        assert lb in M.RECOMMEND_ACTIONS and M.RECOMMEND_ACTIONS[lb]


# =============================================================================
# NHÓM 4 - KIỂM THỬ PHÂN TÍCH ĐỘ NHẠY
# =============================================================================

def test_do_nhay_trong_so_luon_tong_bang_1():
    """Khi quét trọng số của một tiêu chí, tổng trọng số vẫn phải bằng 1 và
    tỉ lệ giữa các tiêu chí còn lại phải giữ nguyên."""
    rng = np.random.default_rng(11)
    X = rng.random((15, 5)) * 100
    w0 = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
    tbl = M.sensitivity_one_at_a_time(X, w0, M.CRITERIA_TYPES, criterion_index=3,
                                      grid=np.array([0.1, 0.3, 0.5]))
    assert tbl.shape == (3, 15)
    # Mỗi kịch bản phải là một hoán vị hạng hợp lệ 1..15
    for _, row in tbl.iterrows():
        assert sorted(row.tolist()) == list(range(1, 16))


def test_do_nhay_phat_hien_duoc_thay_doi_xep_hang():
    """Sản phẩm A tốt về tiêu chí 0 nhưng tệ về tiêu chí 1, sản phẩm B ngược lại.
    Khi dồn trọng số sang tiêu chí 1 thì B phải vượt lên trên A."""
    X = np.array([[100.0, 1.0], [1.0, 100.0], [50.0, 50.0]])
    w0 = np.array([0.5, 0.5])
    tbl = M.sensitivity_one_at_a_time(X, w0, np.array([1, 1]), criterion_index=1,
                                      grid=np.array([0.1, 0.9]))
    assert tbl.loc[0.1, 0] < tbl.loc[0.1, 1], "Khi tiêu chí 0 nặng, A phải trên B"
    assert tbl.loc[0.9, 1] < tbl.loc[0.9, 0], "Khi tiêu chí 1 nặng, B phải trên A"


def test_stability_top_k():
    tbl = pd.DataFrame({0: [1, 1, 2], 1: [2, 3, 1], 2: [3, 2, 3]})
    st = M.stability_top_k(tbl, k=2)
    assert st[0] == 100.0            # luôn trong top 2
    assert abs(st[1] - 200 / 3) < 1e-9
    assert abs(st[2] - 100 / 3) < 1e-9


def test_kiem_tra_dao_hang_chay_duoc():
    rng = np.random.default_rng(5)
    X = rng.random((10, 3)) * 100
    out = M.check_rank_reversal(X, np.array([0.5, 0.3, 0.2]),
                                np.array([1, 0, 1]), drop_index=0)
    assert len(out) == 9
    assert set(out.columns) == {"index_goc", "hang_truoc", "hang_sau", "thay_doi"}


def test_tom_tat_dao_hang_phan_biet_vun_va_dang_ke():
    """Bản tóm tắt phải tách được 'xáo trộn vụn ở giữa bảng' khỏi 'thay đổi
    thành phần Top-k' - hai chuyện có ý nghĩa rất khác nhau khi ra quyết định."""
    # Top-2 giữ nguyên {0, 1}, chỉ hai phương án cuối đổi chỗ cho nhau.
    rr = pd.DataFrame({
        "index_goc":  [0, 1, 2, 3],
        "hang_truoc": [1, 2, 3, 4],
        "hang_sau":   [1, 2, 4, 3],
    })
    rr["thay_doi"] = rr["hang_sau"] - rr["hang_truoc"]
    s = M.summarize_rank_reversal(rr, k=2)
    assert s["n_changed"] == 2
    assert s["topk_stable"] is True
    assert s["topk_in"] == set() and s["topk_out"] == set()
    assert s["max_shift"] == 1

    # Trường hợp Top-2 thực sự đổi: phương án 2 chen lên, phương án 1 rớt xuống.
    rr2 = pd.DataFrame({
        "index_goc":  [0, 1, 2, 3],
        "hang_truoc": [1, 2, 3, 4],
        "hang_sau":   [1, 3, 2, 4],
    })
    rr2["thay_doi"] = rr2["hang_sau"] - rr2["hang_truoc"]
    s2 = M.summarize_rank_reversal(rr2, k=2)
    assert s2["topk_stable"] is False
    assert s2["topk_in"] == {2} and s2["topk_out"] == {1}


# =============================================================================
# NHÓM 5 - KIỂM THỬ TRÊN DỮ LIỆU THẬT CỦA ĐỒ ÁN
# =============================================================================

def test_chay_tren_du_lieu_that():
    """Chạy toàn bộ chuỗi AHP -> TOPSIS -> khuyến nghị trên file Excel thật."""
    try:
        from topsis_app import load_data
    except Exception as e:                                   # pragma: no cover
        print(f"    (bỏ qua: không import được load_data - {e})")
        return

    df, meta = load_data()
    if df.empty:                                             # pragma: no cover
        print("    (bỏ qua: không đọc được file dữ liệu)")
        return

    # Sau tiền xử lý, ma trận quyết định phải sạch: không âm, không NaN.
    X = df[M.CRITERIA_KEYS].to_numpy(dtype=float)
    assert not np.isnan(X).any(), "Còn NaN sau tiền xử lý"
    assert (X >= 0).all(), "Còn giá trị âm sau tiền xử lý"

    A = M.build_pairwise_matrix(M.DEFAULT_PAIRWISE_UPPER, 5)
    ahp = M.ahp_weights(A)
    assert ahp["is_consistent"]

    out = M.topsis(X, ahp["weights"], M.CRITERIA_TYPES)
    sc = out["scores"]
    assert len(sc) == len(df)
    assert (sc >= 0).all() and (sc <= 1).all()

    labels, info = M.classify_by_percentile(sc)
    vc = pd.Series(labels).value_counts()
    # Điểm mấu chốt: cả BA nhóm khuyến nghị đều phải có sản phẩm.
    for lb in M.RECOMMEND_LABELS:
        assert vc.get(lb, 0) > 0, f"Nhóm '{lb}' rỗng - ngưỡng khuyến nghị vô nghĩa"

    print(f"    -> {len(df)} sản phẩm | C* trong [{sc.min():.4f}, {sc.max():.4f}]")
    print(f"    -> Phân nhóm: {vc.to_dict()}")
    print(f"    -> Số dòng bị loại khi làm sạch: {meta['n_dropped']}, "
          f"số dòng tồn kho âm đã xử lý: {meta['n_negative_stock']}")


# =============================================================================
# BỘ CHẠY ĐƠN GIẢN (không cần cài pytest)
# =============================================================================

def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    print("=" * 72)
    print(f"CHẠY {len(tests)} CA KIỂM THỬ CHO MODULE MÔ HÌNH DSS (TV3)")
    print("=" * 72)

    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            failed.append((name, e))
            print(f"[ TRƯỢT ] {name}\n    {type(e).__name__}: {e}")
        else:
            print(f"[  ĐẠT  ] {name}")

    print("-" * 72)
    if failed:
        print(f"KẾT QUẢ: {len(tests) - len(failed)}/{len(tests)} đạt, "
              f"{len(failed)} trượt.")
        return 1
    print(f"KẾT QUẢ: TẤT CẢ {len(tests)}/{len(tests)} CA KIỂM THỬ ĐỀU ĐẠT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
