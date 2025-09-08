
# main_updated_fix.py
# (핵심 수정) df0 → df로 일관화하여 NameError 방지
# - 파일 로드 후 변수명을 df_all로 통일
# - 타임스탬프 구간 선택 후 df = df_sel 로 사용
# - 나머지 로직 동일

from __future__ import annotations

from typing import Dict, List, Tuple
from pathlib import Path
import io
import re
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import streamlit as st

# -------------------- 폰트 설정 (한글 깨짐 방지) --------------------
def setup_korean_font():
    try:
        font_path = Path(__file__).parent / "font" / "NanumGothic.otf"
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    except Exception:
        pass

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["NanumGothicOTF", "NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"]:
        if name in available:
            rcParams["font.family"] = name
            break
    rcParams["axes.unicode_minus"] = False

setup_korean_font()

# -------------------- Streamlit 기본 --------------------
st.set_page_config(page_title="에너지 로그: 슬라이딩 분석(Count + Stride)", layout="wide")
st.title("📊 에너지 로그 — 슬라이딩 윈도우 분석 (Count 기반 + Stride)")
st.caption(
    "윈도우 크기(데이터 개수)와 Stride(샘플 간격)로 슬라이딩 분석합니다. "
    "각 윈도우의 평균 ± 임계치(±threshold)를 벗어나는 값이 **연속으로** 나타나면 이상으로 판정합니다."
)

# -------------------- 사이드바 --------------------
st.sidebar.title("설정")
window_count = st.sidebar.number_input("윈도우 크기(데이터 개수)", min_value=5, max_value=50000, value=600, step=1)
stride_count = st.sidebar.number_input("Stride (샘플 단위)", min_value=1, max_value=50000, value=60, step=1)
streak_min = st.sidebar.number_input("이상치 최소 연속 길이(개)", min_value=1, max_value=10000, value=3, step=1)
threshold_abs = st.sidebar.number_input("임계치 (±)", min_value=0.0, value=40.0, step=1.0)
show_detail_points = st.sidebar.checkbox("임계 초과 포인트 마커 표시", value=True)

uploaded_files = st.file_uploader(
    "로그 파일 업로드(여러 개 가능, .csv/.txt/.log) — 1열: 타임스탬프, 2열: 값",
    type=["csv", "txt", "log"],
    accept_multiple_files=True
)

st.markdown(
    """
**파일 형식 예시**  
2025/07/07 05:00:02.555 , 1781.77  
2025/07/07 05:00:05.586 , 1784.21  
왼쪽은 타임스탬프(밀리초 포함 가능), 오른쪽은 수치입니다. 쉼표와 공백은 유연하게 처리합니다.
"""
)

# -------------------- 유틸 --------------------
def safe_name(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", s)

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

# -------------------- 파싱 --------------------
def load_log(file) -> pd.DataFrame:
    df = pd.read_csv(
        file,
        names=["timestamp", "value"],
        header=None,
        engine="python",
        sep=r"\s*,\s*",
        usecols=[0, 1],
        dtype={1: "float64"},
        skip_blank_lines=True,
        na_values=["", "NaN", "nan"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp"] = df["timestamp"].dt.floor("S")
    df = df.dropna(subset=["timestamp", "value"]).copy()
    return df.sort_values("timestamp")

# -------------------- 윈도우 인덱스 생성 --------------------
def iter_windows(n_total: int, win: int, stride: int):
    if win <= 0 or stride <= 0:
        return
    i = 0
    while i + win <= n_total:
        yield i, i + win  # [i, j)
        i += stride

# -------------------- 이상치(연속 구간) 탐지 --------------------
def find_consecutive_runs_outside_band(values: np.ndarray, mean_val: float, threshold: float, min_len: int) -> List[Tuple[int, int]]:
    is_out = np.abs(values - mean_val) > float(threshold)
    runs: List[Tuple[int, int]] = []
    start = None
    for i, flag in enumerate(is_out):
        if flag and start is None:
            start = i
        elif (not flag) and (start is not None):
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(is_out) - 1))
    return [(s, e) for s, e in runs if (e - s + 1) >= int(min_len)]

def merge_time_intervals(intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not intervals:
        return []
    iv = sorted(intervals, key=lambda x: x[0])
    merged = [iv[0]]
    for s, e in iv[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged

# -------------------- 분석 --------------------
def analyze_sliding_windows(df: pd.DataFrame, win: int, stride: int, threshold: float, streak_min: int):
    ts = df["timestamp"].to_numpy()
    vals = df["value"].to_numpy()
    n = len(df)

    window_rows = []
    window_intervals = []
    point_runs: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    for i, j in iter_windows(n, win, stride):
        segment = vals[i:j]
        seg_ts = ts[i:j]
        seg_mean = float(segment.mean()) if len(segment) else np.nan
        runs = find_consecutive_runs_outside_band(segment, seg_mean, threshold, streak_min)
        is_anom = len(runs) > 0
        row = {
            "윈도우_시작": pd.Timestamp(seg_ts[0]).isoformat(),
            "윈도우_끝": pd.Timestamp(seg_ts[-1]).isoformat(),
            "데이터_개수": int(len(segment)),
            "평균": seg_mean,
            "최솟값": float(segment.min()) if len(segment) else np.nan,
            "최댓값": float(segment.max()) if len(segment) else np.nan,
            "연속_이상_구간_수": len(runs),
            "이상윈도우": "예" if is_anom else "아니오",
        }
        window_rows.append(row)

        if is_anom:
            window_intervals.append((pd.Timestamp(seg_ts[0]), pd.Timestamp(seg_ts[-1])))
            for (s, e) in runs:
                point_runs.append((pd.Timestamp(seg_ts[s]), pd.Timestamp(seg_ts[e])))

    anomalous_intervals = merge_time_intervals(window_intervals)
    return window_rows, anomalous_intervals, point_runs

# -------------------- 그리기 --------------------
def plot_series_with_anomalies(df: pd.DataFrame, win: int, threshold: float, anomalous_intervals, point_runs, show_points: bool):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df["timestamp"], df["value"], linewidth=1.0, label="원본 값")

    if win > 1 and win <= len(df):
        roll = df["value"].rolling(window=win, min_periods=max(1, win//5)).mean()
        ax.plot(df["timestamp"], roll, linestyle="--", linewidth=1.0, label=f"롤링 평균({win})")
        ax.plot(df["timestamp"], roll + threshold, linestyle=":", linewidth=1.0, label=f"임계 상한(+{threshold:.2f})")
        ax.plot(df["timestamp"], roll - threshold, linestyle=":", linewidth=1.0, label=f"임계 하한(-{threshold:.2f})")

        if show_points:
            out_mask = (df["value"] > (roll + threshold)) | (df["value"] < (roll - threshold))
            out_df = df[out_mask]
            if not out_df.empty:
                ax.scatter(out_df["timestamp"], out_df["value"], s=16, label="임계 초과", zorder=3)

    for (s, e) in anomalous_intervals:
        ax.axvspan(s, e, color="orange", alpha=0.15, label="이상 구간")

    ax.set_title("시계열 및 이상 구간")
    ax.set_xlabel("시간")
    ax.set_ylabel("값")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="best", framealpha=0.9)
    st.pyplot(fig)
    return fig

# -------------------- 메인 --------------------
if not uploaded_files:
    st.info("좌측/상단에서 로그 파일을 업로드하세요.")
    st.stop()

# 파일 선택
names = [f.name for f in uploaded_files]
file_idx = st.selectbox("분석할 파일 선택", options=list(range(len(names))), format_func=lambda i: names[i])

# 로드 (df_all로 통일)
try:
    df_all = load_log(uploaded_files[file_idx])
    if df_all.empty:
        st.warning("선택한 파일에 유효한 데이터가 없습니다.")
        st.stop()
except Exception as e:
    st.error(f"파일 로딩 중 오류: {e}")
    st.stop()

# -------------------- 타임스탬프 구간 선택 --------------------
min_ts = df_all["timestamp"].min()
max_ts = df_all["timestamp"].max()
st.subheader("⏱️ 분석할 타임스탬프 구간 선택")
ts_range = st.slider("타임 범위", min_value=min_ts.to_pydatetime(), max_value=max_ts.to_pydatetime(),
                     value=(min_ts.to_pydatetime(), max_ts.to_pydatetime()))

mask = (df_all["timestamp"] >= pd.Timestamp(ts_range[0])) & (df_all["timestamp"] <= pd.Timestamp(ts_range[1]))
df_sel = df_all.loc[mask].reset_index(drop=True)
st.write(f"선택 구간 데이터 개수: {len(df_sel)}")

if len(df_sel) < max(5, int(window_count)):
    st.warning("선택한 구간의 데이터가 윈도우 크기보다 적습니다. 구간을 넓히거나 윈도우/Stride를 줄여주세요.")
    st.stop()

# 이후 로직은 df_sel을 df로 사용
df = df_sel

st.divider()
# -------------------- 로그 분석 버튼 --------------------
if st.button("🚀 로그 분석"):
    rows, merged_intervals, point_runs = analyze_sliding_windows(
        df, int(window_count), int(stride_count), float(threshold_abs), int(streak_min)
    )

    # 윈도우 요약 테이블
    st.subheader("📄 윈도우 요약")
    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True)

    # CSV 다운로드
    csv_bytes = summary_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("윈도우 요약 CSV 다운로드", data=csv_bytes, file_name="window_summary.csv", mime="text/csv")

    # 이상 구간 결과
    st.subheader("⚠️ 이상 구간 (병합된 윈도우 구간)")
    if merged_intervals:
        result_rows = []
        for s, e in merged_intervals:
            result_rows.append({
                "이상_시작": s.isoformat(),
                "이상_끝": e.isoformat(),
                "지속시간(초)": (e - s).total_seconds()
            })
        anom_df = pd.DataFrame(result_rows)
        st.dataframe(anom_df, use_container_width=True)
        anom_csv = anom_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("이상 구간 CSV 다운로드", data=anom_csv, file_name="anomalous_intervals.csv", mime="text/csv")
    else:
        st.info("이상 구간이 탐지되지 않았습니다.")

    # 그래프 표시
    st.subheader("📈 시각화")
    fig = plot_series_with_anomalies(df, int(window_count), float(threshold_abs), merged_intervals, point_runs, show_detail_points)
    png = fig_to_png_bytes(fig)
    plt.close(fig)
    st.download_button("현재 그래프 PNG 저장", data=png, file_name=f"{safe_name(names[file_idx])}_plot.png", mime="image/png")

    # (선택) 모든 이상 윈도우 개별 그래프 ZIP 만들기
    st.markdown("### 💾 이상 윈도우별 그래프 저장 (ZIP, 선택)")
    do_zip = st.checkbox("이상 윈도우만 ZIP으로 저장")
    max_zip = st.number_input("ZIP으로 저장할 최대 윈도우 수(성능 보호)", min_value=1, max_value=5000, value=50)
    if do_zip and st.button("ZIP 생성"):
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            count = 0
            for (s, e) in merged_intervals:
                seg = df[(df["timestamp"] >= s) & (df["timestamp"] <= e)].reset_index(drop=True)
                if seg.empty:
                    continue
                fig2 = plot_series_with_anomalies(seg, int(window_count), float(threshold_abs), [], [], show_detail_points=False)
                png2 = fig_to_png_bytes(fig2)
                plt.close(fig2)
                count += 1
                if count > int(max_zip):
                    break
                fname = f"ANOM_{count:03d}_{safe_name(s.isoformat())}_{safe_name(e.isoformat())}.png"
                zf.writestr(fname, png2)
        mem.seek(0)
        st.download_button("ZIP 다운로드", data=mem.getvalue(), file_name="anomal_windows.zip", mime="application/zip")
else:
    st.info("좌측 파라미터를 설정한 뒤 **로그 분석** 버튼을 눌러 결과를 확인하세요.")
