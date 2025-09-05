# main.py
# - 10분 고정 블록으로 분할하여 표시
# - 이상치: 같은 블록 안에서 "블록 평균보다 큰 값"이 연속으로 나타나는 구간
# - 이상 구간은 연한 주황색 음영으로 표시(color="orange", alpha=0.15)
# - 리샘플링 없음, 밀리초 제외(초 단위)
# - 그래프 저장: 개별 블록 PNG, 전체 블록 ZIP
# - 한글 폰트: font/NanumGothic.otf 우선 적용 (없으면 대체 폰트)

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
    """
    1) font/NanumGothic.otf 있으면 등록 후 'NanumGothicOTF' 또는 'NanumGothic' 사용
    2) 없으면 시스템 폰트 중 'NanumGothic' → 'Noto Sans CJK KR' → 'DejaVu Sans' 순으로 시도
    """
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
st.set_page_config(page_title="에너지 로그: 10분 블록 뷰어", layout="wide")
st.title("📊 에너지 로그 10분 블록 뷰어")
st.caption("리샘플링 없이 원본 사용 · 밀리초 제외(초 단위) · "
           "블록 평균보다 큰 값의 '연속 구간'을 이상치로 표시 · 그래프 저장 지원")

# -------------------- 사이드바 --------------------
st.sidebar.title("설정")
block_minutes = st.sidebar.number_input("블록 크기 (분)", min_value=1, max_value=180, value=10, step=1)
streak_min = st.sidebar.number_input("이상치 최소 연속 길이(개)", min_value=1, max_value=1000, value=2, step=1)
tz = st.sidebar.text_input("타임존(예: Asia/Seoul)", value="Asia/Seoul")
show_all = st.sidebar.checkbox("모든 블록 그래프 한꺼번에 보기", value=False)
max_show = st.sidebar.number_input("한꺼번에 그릴 최대 블록 수(성능 보호)", min_value=1, max_value=300, value=30)

uploaded_files = st.file_uploader("로그 파일 업로드(여러 개 가능, .csv/.txt/.log)",
                                  type=["csv", "txt", "log"], accept_multiple_files=True)

st.markdown("""
**파일 형식 예시**
2025/07/07 05:00:02.555 , 1781.77
2025/07/07 05:00:05.586 , 1784.21왼쪽은 타임스탬프(밀리초 포함 가능), 오른쪽은 수치입니다. 쉼표와 공백은 유연하게 처리합니다.
""")

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
    # 밀리초 제외(초 단위 내림)
    df["timestamp"] = df["timestamp"].dt.floor("S")
    df = df.dropna(subset=["timestamp", "value"]).copy()
    # 타임존 지정(가능하면)
    if tz:
        try:
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
        except Exception:
            pass
    return df.sort_values("timestamp")

# -------------------- 블록 분할 --------------------
def split_into_blocks(df: pd.DataFrame, minutes: int) -> Dict[pd.Timestamp, pd.DataFrame]:
    slot = df["timestamp"].dt.floor(f"{minutes}T")
    df2 = df.copy()
    df2["block_start"] = slot
    blocks = {k: v.drop(columns=["block_start"]) for k, v in df2.groupby("block_start", sort=True)}
    return blocks

# -------------------- 이상치(연속 구간) 탐지 --------------------
def find_consecutive_runs_above_mean(block_df: pd.DataFrame, min_len: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    블록 평균보다 큰 값이 연속으로 나타나는 구간(start_ts, end_ts) 리스트를 반환.
    min_len 이상 길이의 연속 구간만 유지.
    """
    if block_df.empty:
        return []

    mean_val = float(block_df["value"].mean())
    values = block_df["value"].to_numpy()
    times = block_df["timestamp"].to_numpy()

    is_above = values > mean_val
    runs: List[Tuple[int, int]] = []  # (start_idx, end_idx) inclusive

    start = None
    for i, flag in enumerate(is_above):
        if flag and start is None:
            start = i
        elif (not flag) and (start is not None):
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(is_above) - 1))

    # min_len 필터
    good_runs = [(s, e) for s, e in runs if (e - s + 1) >= int(min_len)]

    # 타임스탬프로 변환
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for s, e in good_runs:
        intervals.append((pd.Timestamp(times[s]), pd.Timestamp(times[e])))
    return intervals

def analyze_block(block_df: pd.DataFrame, min_streak: int) -> Dict:
    mean_val = float(block_df["value"].mean())
    intervals = find_consecutive_runs_above_mean(block_df, min_streak)
    start_ts = block_df["timestamp"].min()
    end_ts = block_df["timestamp"].max()
    return {
        "mean": mean_val,
        "intervals": intervals,            # 연속 이상 구간들
        "is_anomaly": len(intervals) > 0,  # 하나라도 있으면 이상 블록
        "start": start_ts,
        "end": end_ts,
        "n": int(block_df.shape[0]),
        "min": float(block_df["value"].min()),
        "max": float(block_df["value"].max()),
    }

# -------------------- 그리기 --------------------
def plot_block(block_df: pd.DataFrame, info: Dict, title: str):
    fig, ax = plt.subplots(figsize=(11, 4))
    # 원본 값
    ax.plot(block_df["timestamp"], block_df["value"], label="원본 값")
    # 블록 평균선
    ax.axhline(info["mean"], linestyle="--", linewidth=1, label="블록 평균")

    # 평균 초과 포인트 마커
    above = block_df[block_df["value"] > info["mean"]]
    if not above.empty:
        ax.scatter(above["timestamp"], above["value"], marker="o", s=30, label="평균 초과", zorder=3)

    # 연속 이상 구간을 연한 주황색으로 음영 표시
    for (s, e) in info["intervals"]:
        ax.axvspan(s, e, color="orange", alpha=0.15, label="연속 이상 구간")

    # 범례 중복 라벨 제거
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="best")

    ax.set_title(title)
    ax.set_xlabel("시간")
    ax.set_ylabel("값")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)
    return fig

# -------------------- ZIP 생성 --------------------
def render_all_blocks_to_zip(blocks: Dict[pd.Timestamp, pd.DataFrame], infos: Dict[pd.Timestamp, Dict],
                             base_prefix: str, limit: int | None = None) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        keys = list(blocks.keys())
        if limit is not None:
            keys = keys[:limit]
        for i, k in enumerate(keys, start=1):
            info = infos[k]
            title = f"{k.isoformat()} ~ {info['end'].isoformat()} | 이상치:{'예' if info['is_anomaly'] else '아니오'}"
            fig = plot_block(blocks[k], info, title)  # 화면에도 렌더됨
            png_bytes = fig_to_png_bytes(fig)
            plt.close(fig)
            fname = f"{base_prefix}_{i:03d}_{'ANOM' if info['is_anomaly'] else 'OK'}_{safe_name(k.isoformat())}.png"
            zf.writestr(fname, png_bytes)
    mem.seek(0)
    return mem.read()

# -------------------- 메인 --------------------
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

# 블록화
blocks = split_into_blocks(df, block_minutes)
block_keys = list(blocks.keys())
if not block_keys:
    st.warning("블록으로 나눌 데이터가 없습니다.")
    st.stop()

# 블록 요약
rows = []
infos: Dict[pd.Timestamp, Dict] = {}
for k in block_keys:
    info = analyze_block(blocks[k], streak_min)
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

st.subheader("📄 블록 요약")
summary_df = pd.DataFrame(rows)
st.dataframe(summary_df, use_container_width=True)

# 요약 CSV 다운로드
csv_bytes = summary_df.to_csv(index=False).encode("utf-8-sig")
st.download_button("블록 요약 CSV 다운로드", data=csv_bytes, file_name="block_summary.csv", mime="text/csv")

# ZIP 저장
st.markdown("### 💾 모든 블록 그래프 저장 (ZIP)")
max_zip = st.number_input("ZIP으로 저장할 최대 블록 수(성능 보호)", min_value=1, max_value=len(block_keys),
                          value=min(len(block_keys), 100))
if st.button("모든 블록 그래프 ZIP 생성"):
    base_prefix = safe_name(f"{names[file_idx]}_blocks_{block_minutes}min_streak{streak_min}")
    zip_bytes = render_all_blocks_to_zip(blocks, infos, base_prefix, limit=int(max_zip))
    st.download_button("ZIP 다운로드", data=zip_bytes, file_name=f"{base_prefix}.zip", mime="application/zip")

st.divider()

# 블록 그래프 표시 + 개별 저장
st.subheader("📈 블록별 그래프")
if show_all:
    if len(block_keys) > max_show:
        st.warning(f"블록이 {len(block_keys)}개라 모든 그래프를 그리면 느릴 수 있어요. 상한 {max_show}개까지만 표시합니다.")
    to_show = block_keys[:max_show]
    for i, k in enumerate(to_show, start=1):
        info = infos[k]
        title = f"[{i}/{len(block_keys)}] {k.isoformat()} ~ {info['end'].isoformat()} | 이상치: {'예' if info['is_anomaly'] else '아니오'}"
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
    title = f"[{idx}/{len(block_keys)}] {k.isoformat()} ~ {info['end'].isoformat()} | 이상치: {'예' if info['is_anomaly'] else '아니오'}"
    fig = plot_block(blocks[k], info, title)
    png = fig_to_png_bytes(fig)
    plt.close(fig)
    st.download_button(
        label="현재 블록 PNG 저장",
        data=png,
        file_name=f"{safe_name(names[file_idx])}_{idx:03d}_{'ANOM' if info['is_anomaly'] else 'OK'}_{safe_name(k.isoformat())}.png",
        mime="image/png"
    )
