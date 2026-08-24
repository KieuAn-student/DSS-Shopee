# -*- coding: utf-8 -*-
# Lõi tính toán DSS: AHP (trọng số) + TOPSIS (xếp hạng) + ngưỡng khuyến nghị
# + phân tích độ nhạy. Tách khỏi giao diện để kiểm thử độc lập.
# Lý thuyết và công thức đầy đủ: xem Tai_lieu_thuyet_trinh_TOPSIS.docx

from __future__ import annotations

import numpy as np
import pandas as pd

# type: 1 = benefit (càng lớn càng tốt), 0 = cost (càng nhỏ càng tốt)
CRITERIA = [
    {"key": "Lợi nhuận (VNĐ)", "label": "Lợi nhuận/SP", "type": 1,
     "help": "Giá bán trừ giá nhập, tính trên một sản phẩm (chưa trừ phí sàn)."},
    {"key": "Đã bán (SL)", "label": "Số lượng bán", "type": 1,
     "help": "Số lượng bán ra trong kỳ, thể hiện sức bán."},
    {"key": "Doanh thu (VNĐ)", "label": "Doanh thu", "type": 1,
     "help": "Doanh thu của kỳ đang xét."},
    {"key": "Tồn kho (SL)", "label": "Tồn kho", "type": 0,
     "help": "Tồn càng nhiều càng đọng vốn nên càng thấp càng tốt."},
    {"key": "Giá nhập (VNĐ)", "label": "Giá nhập", "type": 0,
     "help": "Vốn bỏ ra cho mỗi sản phẩm, càng thấp càng dễ xoay vòng."},
]

CRITERIA_KEYS = [c["key"] for c in CRITERIA]
CRITERIA_LABELS = [c["label"] for c in CRITERIA]
CRITERIA_TYPES = np.array([c["type"] for c in CRITERIA])


# ---------------------------------------------------------------------------
# AHP
# ---------------------------------------------------------------------------

# Random Index của Saaty, tra theo cỡ ma trận n.
RANDOM_INDEX = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

SAATY_SCALE = {
    1: "1 - Quan trọng như nhau",
    2: "2 - Giữa 1 và 3",
    3: "3 - Quan trọng hơn một chút",
    4: "4 - Giữa 3 và 5",
    5: "5 - Quan trọng hơn rõ rệt",
    6: "6 - Giữa 5 và 7",
    7: "7 - Quan trọng hơn nhiều",
    8: "8 - Giữa 7 và 9",
    9: "9 - Quan trọng hơn tuyệt đối",
}

# Tam giác trên của ma trận so sánh cặp mặc định. CR = 0.0074.
DEFAULT_PAIRWISE_UPPER = {
    (0, 1): 2.0, (0, 2): 2.0, (0, 3): 3.0, (0, 4): 4.0,
    (1, 2): 1.0, (1, 3): 2.0, (1, 4): 3.0,
    (2, 3): 2.0, (2, 4): 3.0, (3, 4): 2.0,
}

# Mỗi mục tiêu kinh doanh được quy đổi sẵn thành một ma trận so sánh cặp AHP,
# để người dùng không phải trả lời 10 câu so sánh mới xem được kết quả.
# Cả 5 bộ đều đạt CR < 0.1 (test_moi_muc_tieu_kinh_doanh_deu_dat_CR).
BUSINESS_PRESETS = {
    "Cân bằng - nhìn tổng thể mọi mặt": DEFAULT_PAIRWISE_UPPER,
    "Kiếm lời nhiều nhất trên mỗi đơn": {
        (0, 1): 2.0, (0, 2): 3.0, (0, 3): 4.0, (0, 4): 6.0,
        (1, 2): 1.0, (1, 3): 2.0, (1, 4): 3.0,
        (2, 3): 1.0, (2, 4): 2.0, (3, 4): 2.0,
    },
    "Chạy theo hàng bán chạy": {
        (0, 1): 1 / 2, (0, 2): 1.0, (0, 3): 2.0, (0, 4): 3.0,
        (1, 2): 2.0, (1, 3): 3.0, (1, 4): 5.0,
        (2, 3): 2.0, (2, 4): 3.0, (3, 4): 2.0,
    },
    "Tránh ôm thêm hàng đang tồn": {
        (0, 1): 1.0, (0, 2): 1.0, (0, 3): 1 / 3, (0, 4): 1.0,
        (1, 2): 2.0, (1, 3): 1 / 2, (1, 4): 2.0,
        (2, 3): 1 / 3, (2, 4): 1.0, (3, 4): 3.0,
    },
    "Ít vốn, xoay vòng nhanh": {
        (0, 1): 1.0, (0, 2): 2.0, (0, 3): 1.0, (0, 4): 1 / 2,
        (1, 2): 1.0, (1, 3): 1.0, (1, 4): 1 / 3,
        (2, 3): 1.0, (2, 4): 1 / 3, (3, 4): 1 / 3,
    },
}

PRESET_HINTS = {
    "Cân bằng - nhìn tổng thể mọi mặt":
        "Cân nhắc cả 5 mặt, nghiêng nhẹ về lợi nhuận.",
    "Kiếm lời nhiều nhất trên mỗi đơn":
        "Ưu tiên sản phẩm lời nhiều trên từng đơn, kể cả khi bán chậm hơn.",
    "Chạy theo hàng bán chạy":
        "Ưu tiên sản phẩm ra đơn đều để giữ dòng tiền chạy liên tục.",
    "Tránh ôm thêm hàng đang tồn":
        "Ưu tiên thứ đã bán hết, đẩy hàng còn tồn nhiều xuống nhóm nên dừng nhập.",
    "Ít vốn, xoay vòng nhanh":
        "Ưu tiên hàng giá nhập rẻ, hợp khi vốn eo hẹp.",
}


def build_pairwise_matrix(upper: dict, n: int) -> np.ndarray:
    A = np.ones((n, n), dtype=float)
    for (i, j), v in upper.items():
        if i >= j:
            raise ValueError(f"Chỉ khai báo phần tam giác trên (i<j), gặp ({i},{j}).")
        if v <= 0:
            raise ValueError(f"Giá trị so sánh phải dương, gặp a[{i}][{j}]={v}.")
        A[i, j] = float(v)
        A[j, i] = 1.0 / float(v)   # ma trận AHP có tính nghịch đảo
    return A


def ahp_weights(A: np.ndarray, method: str = "eigenvector") -> dict:
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Ma trận so sánh cặp phải vuông.")

    if method == "geometric":
        w = np.prod(A, axis=1) ** (1.0 / n)
        w = w / w.sum()
    else:
        eigvals, eigvecs = np.linalg.eig(A)
        k = int(np.argmax(eigvals.real))
        w = np.abs(eigvecs[:, k].real)
        w = w / w.sum()

    # Tính lại lambda_max từ w để hai phương pháp cho kết quả nhất quán:
    # A·w = lambda_max·w  =>  lambda_max = trung bình của (A·w)_i / w_i
    lambda_max = float(np.mean((A @ w) / w))

    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    RI = RANDOM_INDEX.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0.0

    return {
        "weights": w,
        "lambda_max": lambda_max,
        "CI": float(CI),
        "RI": float(RI),
        "CR": float(CR),
        "is_consistent": bool(CR < 0.1),
        "n": n,
        "method": method,
    }


# ---------------------------------------------------------------------------
# TOPSIS
# ---------------------------------------------------------------------------

def topsis(matrix: np.ndarray,
           weights: np.ndarray,
           criteria_types: np.ndarray) -> dict:
    X = np.asarray(matrix, dtype=float)
    w = np.asarray(weights, dtype=float)
    ctypes = np.asarray(criteria_types)

    if X.ndim != 2:
        raise ValueError("Ma trận quyết định phải là mảng 2 chiều (m x n).")
    m, n = X.shape
    if w.shape[0] != n or ctypes.shape[0] != n:
        raise ValueError(f"Số trọng số/loại tiêu chí ({w.shape[0]}/{ctypes.shape[0]}) "
                         f"không khớp số cột ma trận ({n}).")
    if m < 2:
        raise ValueError("Cần ít nhất 2 phương án để xếp hạng.")
    if np.isnan(X).any():
        raise ValueError("Ma trận quyết định còn giá trị NaN - phải làm sạch trước.")
    # Chuẩn hóa vector giả định x_ij >= 0; giá trị âm làm r_ij đổi dấu và phá vỡ
    # ý nghĩa hình học của A+/A-, nên chặn ngay thay vì cho ra số sai.
    if (X < 0).any():
        bad = np.unique(np.where(X < 0)[1])
        raise ValueError(
            f"TOPSIS (chuẩn hóa vector) yêu cầu dữ liệu không âm. "
            f"Các cột còn giá trị âm: {bad.tolist()}."
        )
    if w.sum() <= 0:
        raise ValueError("Tổng trọng số phải dương.")
    w = w / w.sum()

    # B1: r_ij = x_ij / sqrt(sum_i x_ij^2)
    col_norm = np.sqrt((X ** 2).sum(axis=0))
    # Cột toàn 0 -> đặt chuẩn = 1 để r_ij = 0, tức tiêu chí đó không phân biệt
    # được phương án nào (đúng bản chất) thay vì chia cho 0.
    col_norm = np.where(col_norm == 0, 1.0, col_norm)
    R = X / col_norm

    # B2: v_ij = w_j * r_ij
    V = R * w

    # B3: tiêu chí benefit lấy max cho A+, cost lấy min cho A+ (đảo chiều)
    is_benefit = (ctypes == 1)
    ideal_best = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    ideal_worst = np.where(is_benefit, V.min(axis=0), V.max(axis=0))

    # B4: khoảng cách Euclid tới A+ và A-
    d_best = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

    # B5: C* = D- / (D+ + D-). denom = 0 chỉ khi mọi phương án trùng nhau hoàn
    # toàn -> trả 0.5 thay vì cộng epsilon làm lệch điểm.
    denom = d_best + d_worst
    scores = np.where(denom == 0, 0.5, d_worst / np.where(denom == 0, 1.0, denom))

    return {
        "normalized": R,
        "weighted": V,
        "ideal_best": ideal_best,
        "ideal_worst": ideal_worst,
        "d_best": d_best,
        "d_worst": d_worst,
        "scores": scores,
        "weights_used": w,
    }


def criterion_positions(out: dict) -> np.ndarray:
    """Vị trí 0-1 của từng phương án trên từng tiêu chí:
    |v_ij - A-_j| / |A+_j - A-_j|. Đã tự đúng chiều cho tiêu chí cost vì A+/A-
    được xác định theo loại tiêu chí từ bước 3."""
    V = out["weighted"]
    best, worst = out["ideal_best"], out["ideal_worst"]
    span = np.abs(best - worst)
    span_safe = np.where(span == 0, 1.0, span)
    pos = np.abs(V - worst) / span_safe
    pos = np.where(span == 0, 0.5, pos)   # tiêu chí không phân biệt được
    return np.clip(pos, 0.0, 1.0)


def rank_scores(scores: np.ndarray) -> np.ndarray:
    # method="min" để điểm bằng nhau nhận cùng hạng nguyên, tránh hạng thập
    # phân bị cắt cụt khi ép về int.
    return (pd.Series(scores)
            .rank(ascending=False, method="min")
            .astype(int)
            .to_numpy())


# ---------------------------------------------------------------------------
# Ngưỡng khuyến nghị
# ---------------------------------------------------------------------------

RECOMMEND_LABELS = ["NÊN NHẬP", "NHẬP VỪA", "KHÔNG NÊN NHẬP"]
RECOMMEND_ACTIONS = {
    "NÊN NHẬP": "Đẩy mạnh marketing, ưu tiên nhập thêm",
    "NHẬP VỪA": "Duy trì bán, nhập cầm chừng theo nhu cầu",
    "KHÔNG NÊN NHẬP": "Xả kho, giảm giá, ngừng nhập thêm",
}


def classify_by_percentile(scores: np.ndarray,
                           top_pct: float = 0.20,
                           mid_pct: float = 0.50) -> tuple[np.ndarray, dict]:
    # Dùng phân vị thay vì ngưỡng tuyệt đối: C* phụ thuộc vào chính tập phương
    # án đang xét, với 217 sản phẩm điểm dồn về dải hẹp nên mốc cứng 0.7/0.4 sẽ
    # đẩy gần như cả danh mục vào một nhóm.
    if not (0 < top_pct < 1) or not (0 <= mid_pct < 1) or top_pct + mid_pct >= 1:
        raise ValueError("Cần 0 < top_pct, 0 <= mid_pct và top_pct + mid_pct < 1.")

    s = np.asarray(scores, dtype=float)
    cut_top = float(np.quantile(s, 1.0 - top_pct))
    cut_mid = float(np.quantile(s, 1.0 - top_pct - mid_pct))

    labels = np.where(s >= cut_top, RECOMMEND_LABELS[0],
             np.where(s >= cut_mid, RECOMMEND_LABELS[1], RECOMMEND_LABELS[2]))
    return labels, {"cut_top": cut_top, "cut_mid": cut_mid, "mode": "percentile"}


def classify_by_threshold(scores: np.ndarray,
                          cut_top: float = 0.70,
                          cut_mid: float = 0.40) -> tuple[np.ndarray, dict]:
    if not (0 <= cut_mid < cut_top <= 1):
        raise ValueError("Cần 0 <= cut_mid < cut_top <= 1.")
    s = np.asarray(scores, dtype=float)
    labels = np.where(s >= cut_top, RECOMMEND_LABELS[0],
             np.where(s >= cut_mid, RECOMMEND_LABELS[1], RECOMMEND_LABELS[2]))
    return labels, {"cut_top": cut_top, "cut_mid": cut_mid, "mode": "absolute"}


# ---------------------------------------------------------------------------
# Phân tích độ nhạy
# ---------------------------------------------------------------------------

def sensitivity_one_at_a_time(matrix: np.ndarray,
                              base_weights: np.ndarray,
                              criteria_types: np.ndarray,
                              criterion_index: int,
                              grid: np.ndarray | None = None) -> pd.DataFrame:
    """Đổi trọng số của một tiêu chí, phần còn lại (1 - w_c) chia cho các tiêu
    chí khác theo đúng tỉ lệ gốc để cô lập ảnh hưởng của riêng tiêu chí đó."""
    X = np.asarray(matrix, dtype=float)
    w0 = np.asarray(base_weights, dtype=float)
    w0 = w0 / w0.sum()
    n = len(w0)
    c = int(criterion_index)
    if not (0 <= c < n):
        raise ValueError(f"criterion_index phải trong [0, {n-1}].")

    if grid is None:
        grid = np.round(np.arange(0.05, 0.81, 0.05), 2)

    others = [j for j in range(n) if j != c]
    rest_base = w0[others]
    rest_sum = rest_base.sum()

    rows = []
    for wc in grid:
        w = np.empty(n, dtype=float)
        w[c] = wc
        if rest_sum > 0:
            w[others] = rest_base / rest_sum * (1.0 - wc)
        else:
            w[others] = (1.0 - wc) / len(others)
        sc = topsis(X, w, criteria_types)["scores"]
        rows.append({"w": float(wc), "scores": sc, "ranks": rank_scores(sc)})

    ranks = pd.DataFrame([r["ranks"] for r in rows], index=[r["w"] for r in rows])
    ranks.index.name = f"Trọng số tiêu chí #{c}"
    return ranks


def stability_top_k(rank_table: pd.DataFrame, k: int = 10) -> pd.Series:
    return (rank_table <= k).mean(axis=0).mul(100.0)


def check_rank_reversal(matrix: np.ndarray,
                        weights: np.ndarray,
                        criteria_types: np.ndarray,
                        drop_index: int) -> pd.DataFrame:
    X = np.asarray(matrix, dtype=float)
    m = X.shape[0]
    if not (0 <= drop_index < m):
        raise ValueError(f"drop_index phải trong [0, {m-1}].")

    full_scores = topsis(X, weights, criteria_types)["scores"]
    keep = np.array([i for i in range(m) if i != drop_index])

    # Hạng cũ tính lại trong nhóm còn lại để so sánh công bằng với hạng mới.
    rank_before = rank_scores(full_scores[keep])
    rank_after = rank_scores(topsis(X[keep], weights, criteria_types)["scores"])

    out = pd.DataFrame({
        "index_goc": keep,
        "hang_truoc": rank_before,
        "hang_sau": rank_after,
    })
    out["thay_doi"] = out["hang_sau"] - out["hang_truoc"]
    return out


def summarize_rank_reversal(rr: pd.DataFrame, k: int = 10) -> dict:
    # Đếm thô "bao nhiêu phương án đổi hạng" gây hiểu nhầm khi hàng trăm phương
    # án gần bằng điểm nhau, nên tách riêng phần ảnh hưởng tới Top-k.
    n_changed = int((rr["thay_doi"] != 0).sum())
    n_major = int((rr["thay_doi"].abs() > k).sum())

    before = set(rr.loc[rr["hang_truoc"] <= k, "index_goc"])
    after = set(rr.loc[rr["hang_sau"] <= k, "index_goc"])

    return {
        "n_total": len(rr),
        "n_changed": n_changed,
        "n_major": n_major,
        "max_shift": int(rr["thay_doi"].abs().max()) if len(rr) else 0,
        "median_shift": float(rr["thay_doi"].abs().median()) if len(rr) else 0.0,
        "topk_before": before,
        "topk_after": after,
        "topk_in": after - before,
        "topk_out": before - after,
        "topk_stable": before == after,
        "k": k,
    }
