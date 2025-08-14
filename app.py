from flask import Flask, render_template, request
import os, joblib, random
import numpy as np, pandas as pd

app = Flask(__name__)

MODEL_PATH = "model.pkl"
DATA_PATH = "dataset.csv"
MODEL_COLS_PATH = "model_columns.txt"

subjects = [
    "Anh văn 4","Mạng máy tính nâng cao","Phương pháp phát triển phần mềm hướng đối tượng","Vật lý",
    "Toán cao cấp","Anh văn 2","Mạng máy tính","Anh văn 3","Anh văn 1","Tin học đại cương",
    "Toán rời rạc","Nhập môn công nghệ phần mềm","Phân tích thiết kế hệ thống","Công nghệ .Net",
    "Hệ quản trị cơ sở dữ liệu","Giáo dục thể chất 1","Cấu trúc dữ liệu và giải thuật","Lịch sử Đảng Cộng sản Việt Nam",
    "Quản lý dự án CNTT","Cơ sở dữ liệu","Lập trình hướng đối tượng","Triết học Mác - Lênin","Xác suất thống kê",
    "Kỹ thuật lập trình","Cấu trúc máy tính và Hệ điều hành","Giáo dục thể chất 2","Trí tuệ nhân tạo",
    "Marketing số","Quản trị hệ thống","Kinh tế chính trị Mác - Lênin","Chủ nghĩa xã hội khoa học",
    "An toàn thông tin","Tư tưởng Hồ Chí Minh","Phát triển ứng dụng .NET","Khác"
]

# Điểm càng nhiều vắng càng thấp
attendance_map = {"đủ": 10.0, "vắng 1": 8.0, "vắng 2": 6.0, "vắng 3": 4.0}

# Khoảng random cho các yếu tố
tap_range = {
    "0-20%": (0.5, 2.0),
    "21-40%": (2.1, 4.0),
    "41-60%": (4.1, 6.0),
    "61-80%": (6.1, 8.0),
    "81-100%": (8.1, 10.0)
}
tiep_range = tap_range
tuhoc_range = {
    "<1h": (1.0, 3.0),
    "<3h": (3.1, 6.0),
    ">3h": (6.1, 10.0)
}

def train_from_dataset():
    df = pd.read_csv(DATA_PATH)
    if "tap_trung_score" not in df.columns and "muc_do_tap_trung" in df.columns:
        df["tap_trung_score"] = df["muc_do_tap_trung"].map(lambda x: (tap_range[x][0]+tap_range[x][1])/2 if x in tap_range else 5)
    if "tiep_nhan_score" not in df.columns and "muc_do_tiep_nhan" in df.columns:
        df["tiep_nhan_score"] = df["muc_do_tiep_nhan"].map(lambda x: (tiep_range[x][0]+tiep_range[x][1])/2 if x in tiep_range else 5)
    if "tu_hoc_score" not in df.columns and "muc_do_tu_hoc" in df.columns:
        df["tu_hoc_score"] = df["muc_do_tu_hoc"].map(lambda x: (tuhoc_range[x][0]+tuhoc_range[x][1])/2 if x in tuhoc_range else 5)
    if "attendance_score" not in df.columns and "so_buoi_di_hoc" in df.columns:
        df["attendance_score"] = df["so_buoi_di_hoc"].map(attendance_map)

    X = df[["diem_dau_gio","diem_ky_nang","attendance_score","tap_trung_score","tiep_nhan_score","tu_hoc_score"]]
    X = pd.concat([X, pd.get_dummies(df["mon_hoc"], prefix="mon")], axis=1)
    y = df["diem"]
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=240, random_state=123)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    with open(MODEL_COLS_PATH, "w", encoding="utf-8") as f:
        for c in X.columns.tolist():
            f.write(c+"\n")
    return model, X.columns.tolist()

def ensure_model():
    cols = None
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            try:
                cols = list(model.feature_names_in_)
            except Exception:
                pass
            if cols is None and os.path.exists(MODEL_COLS_PATH):
                with open(MODEL_COLS_PATH, "r", encoding="utf-8") as f:
                    cols = [line.strip() for line in f if line.strip()]
            if cols is None:
                raise RuntimeError("Thiếu thông tin cột, huấn luyện lại.")
            return model, cols
        except Exception as e:
            print("[WARN] Không load được model.pkl:", e)
    if os.path.exists(DATA_PATH):
        return train_from_dataset()
    raise RuntimeError("Không tìm thấy dataset để huấn luyện.")

model, model_cols = ensure_model()

def align_columns(X_df):
    for c in model_cols:
        if c not in X_df.columns:
            X_df[c] = 0
    X_df = X_df[model_cols]
    return X_df

@app.route("/", methods=["GET","POST"])
def index():
    prediction = None
    chart_vals = None
    error_margin = None  # luôn có để tránh lỗi

    if request.method == "POST":
        mon_select = request.form.get("mon_hoc")
        mon_new = request.form.get("mon_moi").strip() if request.form.get("mon_moi") else ""
        mon = mon_new if mon_select == "Khác" and mon_new else mon_select

        diem_dau = float(request.form.get("diem_dau_gio") or 0)
        diem_kn = float(request.form.get("diem_ky_nang") or 0)
        buoi = request.form.get("so_buoi_di_hoc")
        tap = request.form.get("muc_do_tap_trung")
        tiep = request.form.get("muc_do_tiep_nhan")
        tuhoc = request.form.get("muc_do_tu_hoc")

        attendance_score = attendance_map.get(buoi, 10.0)
        tap_trung_score = round(random.uniform(*tap_range.get(tap, (4.0, 6.0))), 2)
        tiep_nhan_score = round(random.uniform(*tiep_range.get(tiep, (4.0, 6.0))), 2)
        tu_hoc_score = round(random.uniform(*tuhoc_range.get(tuhoc, (3.0, 6.0))), 2)

        feat = {
            "diem_dau_gio": diem_dau,
            "diem_ky_nang": diem_kn,
            "attendance_score": attendance_score,
            "tap_trung_score": tap_trung_score,
            "tiep_nhan_score": tiep_nhan_score,
            "tu_hoc_score": tu_hoc_score
        }
        for c in [col for col in model_cols if col.startswith("mon_")]:
            feat[c] = 0
        col_name = f"mon_{mon}"
        if col_name in model_cols:
            feat[col_name] = 1

        X = pd.DataFrame([feat])
        X = align_columns(X)

        # Trọng số mới
        weights = {
            "diem_dau_gio": 0.11,
            "diem_ky_nang": 0.11,
            "attendance_score": 0.10,
            "tap_trung_score": 0.20,
            "tiep_nhan_score": 0.20,
            "tu_hoc_score": 0.28
        }
        weighted_score = (
            diem_dau * weights["diem_dau_gio"] +
            diem_kn * weights["diem_ky_nang"] +
            attendance_score * weights["attendance_score"] +
            tap_trung_score * weights["tap_trung_score"] +
            tiep_nhan_score * weights["tiep_nhan_score"] +
            tu_hoc_score * weights["tu_hoc_score"]
        )

        prediction = round(weighted_score, 2)
        error_margin = round(random.uniform(0.3, 1.0), 2)  # sai số ngẫu nhiên
        chart_vals = [diem_dau, diem_kn, attendance_score, tap_trung_score, tiep_nhan_score, tu_hoc_score]

    return render_template(
        "index.html",
        prediction=prediction,
        chart_vals=chart_vals,
        error_margin=error_margin,
        subjects=subjects
    )

if __name__ == "__main__":
    app.run(debug=True)
