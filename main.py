
# main.py
# - 10분 고정 블록으로 분할하여 표시
# - 블록별 이상치 표시
# - 한글 폰트 자동 설정
# - 리샘플링 없음, 밀리초 제외(초 단위)
# - 그래프 저장 기능: 개별 블록 PNG 저장, 전체 블록 ZIP 저장

from typing import List, Tuple, Dict
import io
import zipfile
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import streamlit as st
from pathlib import Path
from matplotlib import font_manager, rcParams

# 📂 현재 main.py 파일이 있는 디렉토리 기준으로 font 경로 설정
FONT_PATH = Path(__file__).parent / "font" / "NanumGothic.otf"

# NanumGothic.otf 등록
font_manager.fontManager.addfont(str(FONT_PATH))

# Matplotlib 전역 설정
rcParams["font.family"] = "NanumGothic"   # 폰트 패밀리명
rcParams["axes.unicode_minus"] = False    # 마이너스 깨짐 방지
# --------------- Streamlit 설정 ---------------
st.set_page_config(page_title="에너지 로그: 10분 블록 뷰어", layout="wide")

# --------------- 폰트 설정(한글 깨짐 방지) ---------------
def set_korean_font() -> str:
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        available = set()

    priorities = [
        "Malgun Gothic",         # Windows
        "NanumGothic",           # Linux/Google Fonts
        "NanumBarunGothic",
        "AppleGothic",           # macOS
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "Arial Unicode MS",
        "DejaVu Sans",           # fallback
    ]

    chosen = None
    for name in priorities:
        if name in available:
            chosen = name
            break

    if chosen:
        plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    return chosen or "기본 폰트"

font_name = set_korean_font()
st.caption(f"그래프 폰트: {font_name}")

# --------------- 사이드바 ---------------
st.sidebar.title("설정")
block_minutes = st.sidebar.number_input("블록 크기 (분)", min_value=1, max_value=180, value=10, step=1)
threshold_count = st.sidebar.number_input("이상치 기준: 평균보다 큰 데이터 개수", min_value=1, max_value=1000, value=5, step=1)
tz = st.sidebar.text_input("타임존(예: Asia/Seoul)", value="Asia/Seoul")
show_all = st.sidebar.checkbox("모든 블록 그래프 한꺼번에 보기", value=False)
max_show = st.sidebar.number_input("한꺼번에 그릴 최대 블록 수(성능 보호)", min_value=1, max_value=300, value=30)

st.title("📊 에너지 로그 10분 블록 뷰어")
st.caption("리샘플링 없이 원본 사용 · 밀리초 제외(초 단위) · 블록별 이상치 표시 · 그래프 저장 지원")

uploaded_files = st.file_uploader("로그 파일 업로드(여러 개 가능, .csv/.txt/.log)", type=["csv", "txt", "log"], accept_multiple_files=True)
st.markdown("""
**파일 형식 예시**
```
2025/07/07 05:00:02.555 , 1781.77
2025/07/07 05:00:05.586 , 1784.21
```
왼쪽은 타임스탬프(밀리초 포함 가능), 오른쪽은 수치입니다. 쉼표와 공백은 유연하게 처리합니다.
""")

# --------------- 유틸 ---------------
def safe_name(s: str) -> str:
    # 파일 이름에 안전한 형태로 변환
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", s)

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

# --------------- 파싱 ---------------
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

# --------------- 블록 통계 & 이상치 판정 ---------------
def split_into_blocks(df: pd.DataFrame, minutes: int) -> Dict[pd.Timestamp, pd.DataFrame]:
    slot = df["timestamp"].dt.floor(f"{minutes}T")
    df2 = df.copy()
    df2["block_start"] = slot
    blocks = {k: v.drop(columns=["block_start"]) for k, v in df2.groupby("block_start", sort=True)}
    return blocks

def analyze_block(block_df: pd.DataFrame) -> Dict:
    mean_val = float(block_df["value"].mean())
    count_above = int((block_df["value"] > mean_val).sum())
    is_anomaly = count_above >= int(threshold_count)
    start_ts = block_df["timestamp"].min()
    end_ts = block_df["timestamp"].max()
    return {
        "mean": mean_val,
        "count_above_mean": count_above,
        "is_anomaly": is_anomaly,
        "start": start_ts,
        "end": end_ts,
        "n": int(block_df.shape[0]),
        "min": float(block_df["value"].min()),
        "max": float(block_df["value"].max()),
    }

# --------------- 그리기 ---------------
def plot_block(block_df: pd.DataFrame, block_info: Dict, title: str):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(block_df["timestamp"], block_df["value"], label="원본 값")
    ax.axhline(block_info["mean"], linestyle="--", linewidth=1, label="블록 평균")
    above = block_df[block_df["value"] > block_info["mean"]]
    if not above.empty:
        ax.scatter(above["timestamp"], above["value"], marker="o", s=30, label="평균 초과", zorder=3)
    if block_info["is_anomaly"]:
        ax.axvspan(block_info["start"], block_info["end"], alpha=0.15, label="이상치 블록")
    ax.set_title(title)
    ax.set_xlabel("시간")
    ax.set_ylabel("값")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    st.pyplot(fig)
    return fig

# --------------- ZIP 생성 ---------------
def render_all_blocks_to_zip(blocks: Dict[pd.Timestamp, pd.DataFrame], infos: Dict[pd.Timestamp, Dict], base_prefix: str, limit: int | None = None) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        keys = list(blocks.keys())
        if limit is not None:
            keys = keys[:limit]
        for i, k in enumerate(keys, start=1):
            info = infos[k]
            title = f"{k.isoformat()} ~ {info['end'].isoformat()} | 이상치:{'예' if info['is_anomaly'] else '아니오'}"
            fig = plot_block(blocks[k], info, title)  # 렌더(화면에도 표시되지만 ZIP 생성 목적)
            png_bytes = fig_to_png_bytes(fig)
            plt.close(fig)
            fname = f"{base_prefix}_{i:03d}_{'ANOM' if info['is_anomaly'] else 'OK'}_{safe_name(k.isoformat())}.png"
            zf.writestr(fname, png_bytes)
    mem.seek(0)
    return mem.read()

# --------------- 메인 로직 ---------------
if not uploaded_files:
    st.info("좌측/상단에서 로그 파일을 업로드하세요.")
    st.stop()

# 파일 선택
names = [f.name for f in uploaded_files]
file_idx = st.selectbox("분석할 파일 선택", options=list(range(len(names))), format_func=lambda i: names[i])

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

# 블록 요약표(이상치 유무 포함)
rows = []
infos = {}
for k in block_keys:
    info = analyze_block(blocks[k])
    infos[k] = info
    rows.append({
        "블록 시작": k.isoformat(),
        "블록 종료": info["end"].isoformat(),
        "데이터 개수": info["n"],
        "블록 평균": info["mean"],
        "평균 초과 개수": info["count_above_mean"],
        "이상치": "예" if info["is_anomaly"] else "아니오",
        "최솟값": info["min"],
        "최댓값": info["max"],
    })

st.subheader("📄 블록 요약")
summary_df = pd.DataFrame(rows)
st.dataframe(summary_df, use_container_width=True)

# 다운로드: 요약 CSV
csv_bytes = summary_df.to_csv(index=False).encode("utf-8-sig")
st.download_button("블록 요약 CSV 다운로드", data=csv_bytes, file_name="block_summary.csv", mime="text/csv")

# 모든 블록 그래프 ZIP 저장
st.markdown("### 💾 모든 블록 그래프 저장 (ZIP)")
max_zip = st.number_input("ZIP으로 저장할 최대 블록 수(성능 보호)", min_value=1, max_value=len(block_keys), value=min(len(block_keys), 100))
if st.button("모든 블록 그래프 ZIP 생성"):
    base_prefix = safe_name(f"{names[file_idx]}_blocks_{block_minutes}min")
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
        # 개별 PNG 저장 버튼
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
    # 현재 블록 PNG 저장
    png = fig_to_png_bytes(fig)
    plt.close(fig)
    st.download_button(
        label="현재 블록 PNG 저장",
        data=png,
        file_name=f"{safe_name(names[file_idx])}_{idx:03d}_{'ANOM' if info['is_anomaly'] else 'OK'}_{safe_name(k.isoformat())}.png",
        mime="image/png"
    )
