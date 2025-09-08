# main.py
# - 블록(청크)을 "데이터 개수" 단위로 분할하여 표시 (기존 10분 → 개수 기반)
# - 이상치: 같은 블록/윈도우 안에서 "평균 ± 임계치"를 벗어나는 값이 streak_min개 이상 **연속**으로 나타나는 구간
# - 이상 구간은 연한 주황색 음영(color="orange", alpha=0.15)
# - 리샘플링 없음, 타임존 처리 없음, 밀리초 제외(초 단위)
# - 그래프 저장: 개별 블록 PNG, 전체 블록 ZIP
# - 한글 폰트: font/NanumGothic.otf 우선 적용 (없으면 대체 폰트)
#
# [추가 섹션]
# - 슬라이딩 윈도우(데이터 개수 기반) + Stride(윈도우 시작 간격) 분석
# - 분석용 타임스탬프 범위 선택(슬라이딩 분석에만 적용)
# - "로그 분석 실행" 버튼으로 이상 구간(시간대) 표/CSV 및 시각화

from typing import Dict, List, Tuple, Optional
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
st.set_page_config(page_title="에너지 로그: 개수 기반 블록 뷰어", layout="wide")
st.title("📊 에너지 로그 — 개수 기반 블록(청크) 뷰어")
st.caption(
    "리샘플링 없이 원본 사용 · 밀리초 제외(초 단위) · "
    "각 블록의 **평균 ± 임계치**를 벗어나는 값이 **연속(streak_min)** 으로 나타나면 이상 구간으로 표시 · 그래프 저장 지원"
)

# -------------------- 사이드바 --------------------
st.sidebar.title("설정")
block_count = st.sidebar.number_input("블록(청크) 크기 (데이터 개수)", min_value=5, max_value=100_000, value=600, step=1)
streak_min = st.sidebar.number_input("이상치 최소 연속 길이(개)", min_value=1, max_value=10_000, value=2, step=1)
threshold_abs = st.sidebar.number_input("임계치 (±)", min_value=0.0, value=40.0, step=1.0)
show_all = st.sidebar.checkbox("모든 블록 그래프 한꺼번에 보기", value=False)
max_show = st.sidebar.number_input("한꺼번에 그릴 최대 블록 수(성능 보호)", min_value=1, max_value=300, value=30)

# [슬라이딩 분석용 파라미터 (기존 블록 뷰어와 독립)]
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 슬라이딩 분석(카운트 + Stride)")
window_count = st.sidebar.number_input("윈도우 크기(데이터 개수)", min_value=5, max_value=100_000, value=600, step=1)
stride_count = st.sidebar.number_input(
    "Stride (윈도우 시작 간격, 개수)",
    min_value=1, max_value=100_000, value=60, step=1,
    help="슬라이딩 윈도우의 '시작 지점' 사이 간격입니다. 예) win=600, stride=60 → [0:600), [60:660), ..."
)
show_detail_points = st.sidebar.checkbox("임계 초과 포인트 마커(분석 그래프)", value=True)

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
    df["timestamp"] = df["timestamp"].dt.floor("S")  # 밀리초 제거(초 단위), 타임존 변환 없음
    df = df.dropna(subset=["timestamp", "value"]).copy()
    return df.sort_values("timestamp").reset_index(drop=True)

# -------------------- 블록(청크) 분할: 개수 기반 (비중첩, 순차) --------------------
def split_into_count_blocks(df: pd.DataFrame, count: int) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    입력 데이터를 앞에서부터 count개씩 끊어서 비중첩 블록으로 나눕니다.
    딕셔너리 키는 각 블록의 시작 타임스탬프입니다.
    """
    n = len(df)
    if count <= 0 or n == 0:
        return {}
    blocks: Dict[pd.Timestamp, pd.DataFrame] = {}
    for i in range(0, n, count):
        chunk = df.iloc[i:i+count].copy()
        if chunk.empty:
            continue
        start_ts = pd.Timestamp(chunk["timestamp"].iloc[0])
        blocks[start_ts] = chunk.reset_index(drop=True)
    return blocks

# -------------------- 이상치(연속 구간) 탐지 (블록/슬라이딩 공용) --------------------
def find_consecutive_runs_outside_band(
    frame: pd.DataFrame, min_len: int, threshold: float
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    평균 ± threshold 바깥(|x - mean| > threshold)의 포인트가
    'min_len'개 이상 **연속**으로 나타나는 구간(start_ts, end_ts)을 반환.
    """
    if frame.empty:
        return []

    mean_val = float(frame["value"].mean())  # ← 블록 내부 평균
    values = frame["value"].to_numpy()
    times = frame["timestamp"].to_numpy()

    is_out = np.abs(values - mean_val) > float(threshold)  # '넘는다' 기준(동등 제외)
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

    good = [(s, e) for s, e in runs if (e - s + 1) >= int(min_len)]
    return [(pd.Timestamp(times[s]), pd.Timestamp(times[e])) for s, e in good]

def analyze_block(block_df: pd.DataFrame, min_streak: int, threshold: float) -> Dict:
    mean_val = float(block_df["value"].mean())
    intervals = find_consecutive_runs_outside_band(block_df, min_streak, threshold)
    return {
        "mean": mean_val,
        "threshold": float(threshold),
        "intervals": intervals,
        "is_anomaly": len(intervals) > 0,
        "start": block_df["timestamp"].min(),
        "end": block_df["timestamp"].max(),
        "n": int(block_df.shape[0]),
        "min": float(block_df["value"].min()),
        "max": float(block_df["value"].max()),
    }

# -------------------- 그리기(블록) --------------------
def plot_block(block_df: pd.DataFrame, info: Dict, title: str):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(block_df["timestamp"], block_df["value"], linewidth=1.0)
    ax.axhline(info["mean"], linestyle="--", linewidth=1.0)

    upper = info["mean"] + info["threshold"]
    lower = info["mean"] - info["threshold"]
    ax.axhline(upper, color="red", linestyle="--", linewidth=1.0)
    ax.axhline(lower, color="red", linestyle="--", linewidth=1.0)

    out_mask = (block_df["value"] > upper) | (block_df["value"] < lower)
    out_df = block_df[out_mask]
    if not out_df.empty:
        ax.scatter(out_df["timestamp"], out_df["value"], s=24, zorder=3)

    for (s, e) in info["intervals"]:
        ax.axvspan(s, e, color="orange", alpha=0.15)

    ax.set_title(title)
    ax.set_xlabel("시간")
    ax.set_ylabel("값")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)
    return fig

# -------------------- ZIP 생성(블록) --------------------
def render_all_blocks_to_zip(
    blocks: Dict[pd.Timestamp, pd.DataFrame],
    infos: Dict[pd.Timestamp, Dict],
    base_prefix: str,
    limit: Optional[int] = None
) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        keys = sorted(blocks.keys())
        if limit is not None:
            keys = keys[:limit]
        for i, k in enumerate(keys, start=1):
            info = infos[k]
            title = f"{k.isoformat()} ~ {info['end'].isoformat()} | 이상치:{'예' if info['is_anomaly'] else '아니오'}"
            fig = plot_block(blocks[k], info, title)
            png_bytes = fig_to_png_bytes(fig)
            plt.close(fig)
            fname = f"{base_prefix}_{i:03d}_{'ANOM' if info['is_anomaly'] else 'OK'}_{safe_name(k.isoformat())}.png"
            zf.writestr(fname, png_bytes)
    mem.seek(0)
    return mem.read()

# ==================== [추가: 슬라이딩 윈도우 분석 전용] ====================
def iter_windows(n_total: int, win: int, stride: int):
    """
    슬라이딩 윈도우 시작 인덱스를 생성합니다.
    - win: 윈도우 크기(데이터 개수)
    - stride: '나누어진 데이터셋(윈도우) 사이의 간격' = 시작 인덱스 간격
    예) n_total=1000, win=600, stride=60 → [0:600), [60:660), ...
    """
    if win <= 0 or stride <= 0:
        return
    i = 0
    while i + win <= n_total:
        yield i, i + win
        i += stride

def merge_time_intervals(intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not intervals:
        return []
    iv = sorted(intervals, key=lambda x: x[0])
    merged = [iv[0]]
    for s, e in iv[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged

def analyze_sliding_windows(df: pd.DataFrame, win: int, stride: int, threshold: float, streak_min_: int):
    ts = df["timestamp"].to_numpy()
    vals = df["value"].to_numpy()
    n = len(df)

    window_rows = []
    window_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    point_runs: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    for i, j in iter_windows(n, win, stride):
        seg_df = pd.DataFrame({"timestamp": ts[i:j], "value": vals[i:j]})
        runs = find_consecutive_runs_outside_band(seg_df, streak_min_, threshold)
        is_anom = len(runs) > 0

        window_rows.append({
            "윈도우_시작": pd.Timestamp(ts[i]).isoformat(),
            "윈도우_끝": pd.Timestamp(ts[j-1]).isoformat(),
            "데이터_개수": int(j - i),
            "평균": float(seg_df["value"].mean()),
            "최솟값": float(seg_df["value"].min()),
            "최댓값": float(seg_df["value"].max()),
            "연속_이상_구간_수": len(runs),
            "이상윈도우": "예" if is_anom else "아니오",
        })

        if is_anom:
            window_intervals.append((pd.Timestamp(ts[i]), pd.Timestamp(ts[j-1])))
            for (s, e) in runs:
                point_runs.append((s, e))

    merged = merge_time_intervals(window_intervals)
    return window_rows, merged, point_runs

def plot_series_with_anomalies(
    df: pd.DataFrame,
    win: int,
    threshold: float,
    anomalous_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]],
    point_runs: List[Tuple[pd.Timestamp, pd.Timestamp]],
    show_points: bool
):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df["timestamp"], df["value"], linewidth=1.0)

    if win > 1 and win <= len(df):
        roll = df["value"].rolling(window=win, min_periods=max(1, win // 5)).mean()
        ax.plot(df["timestamp"], roll, linestyle="--", linewidth=1.0)
        ax.plot(df["timestamp"], roll + threshold, linestyle=":", linewidth=1.0)
        ax.plot(df["timestamp"], roll - threshold, linestyle=":", linewidth=1.0)

        if show_points:
            out_mask = (df["value"] > (roll + threshold)) | (df["value"] < (roll - threshold))
            out_df = df[out_mask]
            if not out_df.empty:
                ax.scatter(out_df["timestamp"], out_df["value"], s=16, zorder=3)

    for (s, e) in anomalous_intervals:
        ax.axvspan(s, e, color="orange", alpha=0.15)

    ax.set_title("시계열 및 이상 구간(슬라이딩 분석)")
    ax.set_xlabel("시간")
    ax.set_ylabel("값")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)
    return fig

# ==================== 메인 (개수 기반 블록 뷰어) ====================
if not uploaded_files:
    st.info("좌측/상단에서 로그 파일을 업로드하세요.")
    st.stop()

# 파일 선택
names = [f.name for f in uploaded_files]
file_idx = st.selectbox("분석할 파일 선택", options=list(range(len(names))), format_func=lambda i: names[i])

# 로드
try:
    df = load_log(uploaded_files[file_idx])
    if df.empty:
        st.warning("선택한 파일에 유효한 데이터가 없습니다.")
        st.stop()
except Exception as e:
    st.error(f"파일 로딩 중 오류: {e}")
    st.stop()

# 블록화(개수 기반)
blocks = split_into_count_blocks(df, int(block_count))
block_keys = sorted(blocks.keys())
if not block_keys:
    st.warning("블록으로 나눌 데이터가 없습니다.")
    st.stop()

# 블록 요약
rows = []
infos: Dict[pd.Timestamp, Dict] = {}
for k in block_keys:
    info = analyze_block(blocks[k], int(streak_min), float(threshold_abs))
    infos[k] = info
    rows.append({
        "블록 시작": k.isoformat(),
        "블록 종료": info["end"].isoformat(),
        "데이터 개수": info["n"],
        "블록 평균": info["mean"],
        "연속 이상 구간 수": len(info["intervals"]),
        "이상치": "예" if info["is_anomaly"] else "아니오",
        "최솟값": info["min"],
        "최댓값": info["max"],
    })

st.subheader("📄 블록 요약 (개수 기반)")
summary_df = pd.DataFrame(rows)
st.dataframe(summary_df, use_container_width=True)

# 요약 CSV 다운로드
csv_bytes = summary_df.to_csv(index=False).encode("utf-8-sig")
st.download_button("블록 요약 CSV 다운로드", data=csv_bytes, file_name="block_summary_count_based.csv", mime="text/csv")

# ZIP 저장
st.markdown("### 💾 모든 블록 그래프 저장 (ZIP)")
max_zip = st.number_input(
    "ZIP으로 저장할 최대 블록 수(성능 보호)",
    min_value=1,
    max_value=len(block_keys),
    value=min(len(block_keys), 100)
)
if st.button("모든 블록 그래프 ZIP 생성"):
    base_prefix = safe_name(f"{names[file_idx]}_blocks_{int(block_count)}count_streak{int(streak_min)}")
    zip_bytes = render_all_blocks_to_zip(blocks, infos, base_prefix, limit=int(max_zip))
    st.download_button("ZIP 다운로드", data=zip_bytes, file_name=f"{base_prefix}.zip", mime="application/zip")

st.divider()

# 블록 그래프 표시 + 개별 저장
st.subheader("📈 블록별 그래프 (개수 기반)")
if show_all:
    if len(block_keys) > max_show:
        st.warning(f"블록이 {len(block_keys)}개라 모든 그래프를 그리면 느릴 수 있어요. 상한 {max_show}개까지만 표시합니다.")
    to_show = block_keys[:max_show]
    for i, k in enumerate(to_show, start=1):
        info = infos[k]
        title = (
            f"[{i}/{len(block_keys)}] {k.isoformat()} ~ {info['end'].isoformat()} | "
            f"이상치: {'예' if info['is_anomaly'] else '아니오'}"
        )
        fig = plot_block(blocks[k], info, title)
        png = fig_to_png_bytes(fig)
        plt.close(fig)
        st.download_button(
            label=f"이 블록 PNG 저장 ({i})",
            data=png,
            file_name=f"{safe_name(names[file_idx])}_{i:03d}_{'ANOM' if info['is_anomaly'] else 'OK'}_{safe_name(k.isoformat())}.png",
            mime="image/png"
        )
else:
    idx = st.slider("표시할 블록 인덱스", 1, len(block_keys), 1)
    k = block_keys[idx - 1]
    info = infos[k]
    title = (
        f"[{idx}/{len(block_keys)}] {k.isoformat()} ~ {info['end'].isoformat()} | "
        f"이상치: {'예' if info['is_anomaly'] else '아니오'}"
    )
    fig = plot_block(blocks[k], info, title)
    png = fig_to_png_bytes(fig)
    plt.close(fig)
    st.download_button(
        label="현재 블록 PNG 저장",
        data=png,
        file_name=f"{safe_name(names[file_idx])}_{idx:03d}_{'ANOM' if info['is_anomaly'] else 'OK'}_{safe_name(k.isoformat())}.png",
        mime="image/png"
    )

# ==================== [추가 섹션: 슬라이딩 윈도우 로그 분석] ====================
st.divider()
st.subheader("🧪 로그 분석 (슬라이딩: 카운트 + Stride)")

# 타임스탬프 범위 선택 — 분석용(블록 뷰어에는 영향 없음)
min_ts = df["timestamp"].min()
max_ts = df["timestamp"].max()
ts_range = st.slider(
    "분석할 타임스탬프 범위",
    min_value=min_ts.to_pydatetime(),
    max_value=max_ts.to_pydatetime(),
    value=(min_ts.to_pydatetime(), max_ts.to_pydatetime())
)

df_an = df[(df["timestamp"] >= pd.Timestamp(ts_range[0])) & (df["timestamp"] <= pd.Timestamp(ts_range[1]))].reset_index(drop=True)
st.write(f"선택 구간 데이터 개수: {len(df_an)}")

if len(df_an) < max(5, int(window_count)):
    st.info("선택 구간 데이터가 윈도우 크기보다 적습니다. 구간을 넓히거나 윈도우/Stride를 줄여주세요.")
else:
    if st.button("🚀 로그 분석 실행"):
        rows2, merged_intervals, point_runs = analyze_sliding_windows(
            df_an, int(window_count), int(stride_count), float(threshold_abs), int(streak_min)
        )

        # 윈도우 요약
        st.markdown("**윈도우 요약**")
        summary_df2 = pd.DataFrame(rows2)
        st.dataframe(summary_df2, use_container_width=True)
        st.download_button(
            "윈도우 요약 CSV 다운로드",
            data=summary_df2.to_csv(index=False).encode("utf-8-sig"),
            file_name="window_summary.csv",
            mime="text/csv"
        )

        # 이상 구간(병합)
        st.markdown("**⚠️ 이상 구간(병합 결과)**")
        if merged_intervals:
            res_rows = []
            for s, e in merged_intervals:
                res_rows.append({
                    "이상_시작": s.isoformat(),
                    "이상_끝": e.isoformat(),
                    "지속시간(초)": (e - s).total_seconds()
                })
            anom_df = pd.DataFrame(res_rows)
            st.dataframe(anom_df, use_container_width=True)
            st.download_button(
                "이상 구간 CSV 다운로드",
                data=anom_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="anomalous_intervals.csv",
                mime="text/csv"
            )
        else:
            st.info("이상 구간이 탐지되지 않았습니다.")

        # 시각화
        st.markdown("**시각화**")
        fig2 = plot_series_with_anomalies(
            df_an, int(window_count), float(threshold_abs),
            merged_intervals, point_runs, show_detail_points
        )
        png2 = fig_to_png_bytes(fig2)
        plt.close(fig2)
        st.download_button(
            "현재 분석 그래프 PNG 저장",
            data=png2,
            file_name=f"{safe_name(names[file_idx])}_sliding_plot.png",
            mime="image/png"
        )
