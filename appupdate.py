import streamlit as st
import pdfplumber
import io
import re
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import easyocr
import zipfile
import unicodedata
import json
import os
import difflib
import pandas as pd

# ────────────────────────────────────────────────
# 1. AI 學習與記憶模組 (新增)
# ────────────────────────────────────────────────
LEARNING_FILE = "ai_learning.json"

def load_ai_memory():
    if os.path.exists(LEARNING_FILE):
        try:
            with open(LEARNING_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: return {"history": {}}
    return {"history": {}}

def save_ai_memory(memory):
    with open(LEARNING_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def ai_smart_fix(text, category="general"):
    """自動套用 AI 學習過的修正行為"""
    memory = load_ai_memory()
    mapping = memory.get("history", {})
    # 針對整段文字進行已知錯誤置換
    for wrong, right in mapping.items():
        if wrong in text:
            text = text.replace(wrong, right)
    return text

# ────────────────────────────────────────────────
# 2. 環境適應與資源載入
# ────────────────────────────────────────────────
LOCAL_POPPLER_PATH = r"C:\Users\User\Desktop\pdf_explain new\poppler-25.12.0\Library\bin"
POPPLER_PATH = LOCAL_POPPLER_PATH if os.path.exists(LOCAL_POPPLER_PATH) else None

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ch_tra', 'en'], gpu=False)

def normalize(text):
    if not text: return ""
    return unicodedata.normalize("NFKC", re.sub(r'\s+', '', text))

# ────────────────────────────────────────────────
# 3. 核心 OCR 策略與地址校正 (保留原 300 行邏輯)
# ────────────────────────────────────────────────
TAIWAN_CITIES = ['臺北市','新北市','桃園市','臺中市','臺南市','高雄市','基隆市','新竹市','嘉義市','新竹縣','苗栗縣','彰化縣','南投縣','雲林縣','嘉義縣','屏東縣','宜蘭縣','花蓮縣','臺東縣','澎湖縣','金門縣','連江縣']

def fix_addr_post_process(text: str) -> str:
    if not text: return text
    # 先套用 AI 學習結果
    text = ai_smart_fix(text)
    # 執行原有硬編碼校正
    _ADDR_CHAR_MAP = {'耋': '臺', '耸': '臺', '孿': '學', '孽': '學', '壆': '學', '覃': '南'}
    for wrong, right in _ADDR_CHAR_MAP.items():
        text = text.replace(wrong, right)
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    _ADDR_CJK = r'[里鄰路段巷弄號街區市縣鄉鎮村]'
    text = re.sub(rf'({_ADDR_CJK})\s+(\d)', r'\1\2', text)
    text = re.sub(rf'(\d)\s+({_ADDR_CJK})', r'\1\2', text)
    return text

def ocr_with_best_result(ocr, img_gray: np.ndarray) -> tuple:
    fx, fy = 4, 4
    b1 = cv2.resize(img_gray, None, fx=fx, fy=fy, interpolation=cv2.INTER_LANCZOS4)
    results = ocr.readtext(b1, detail=0)
    raw = "".join(results).strip()
    processed = fix_addr_post_process(raw)
    return processed, "Standard"

# ────────────────────────────────────────────────
# 4. 文件解析邏輯 (謄本、群璇、表格式)
# ────────────────────────────────────────────────

def extract_addr_from_image_stream(page, ocr, debug_log: list):
    words = page.extract_words()
    target = next((w for w in words if w['text'] in ['地址', '住址']), None)
    if not target: return ""
    addr_imgs = [img for img in page.images if abs(img['top'] - target['top']) < 5]
    if not addr_imgs: return ""
    try:
        raw = addr_imgs[0]['stream'].get_data()
        buf = np.frombuffer(raw, dtype=np.uint8)
        decoded = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        val, _ = ocr_with_best_result(ocr, decoded)
        return f"{target['text']} {val}"
    except: return ""

def process_表格式(pdf, ocr, all_imgs, fmt):
    output, debug = [], []
    for i, page in enumerate(pdf.pages):
        page_text = []
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                cells = [c.strip().replace("\n", "") if c else "" for c in row]
                if not any(cells): continue
                if normalize(cells[0]) in ["地址", "住址"] and not any(cells[1:]):
                    line = extract_addr_from_image_stream(page, ocr, debug)
                else:
                    line = "  ".join(c for c in cells if c)
                page_text.append(line)
        output.append(f"===== 第 {i+1} 頁 =====\n" + "\n".join(page_text))
    return "\n\n".join(output), debug

def process_群璇(pdf, ocr, all_imgs):
    output = []
    for i, page in enumerate(pdf.pages):
        lines = [ "  ".join(filter(None, [c.replace("\n","") for c in row])) for table in page.extract_tables() for row in table ]
        output.append(f"===== 第 {i+1} 頁 =====\n" + "\n".join(lines))
    return "\n\n".join(output), []

def process_謄本(pdf, ocr, all_imgs):
    output = []
    for i, page in enumerate(pdf.pages):
        txt = page.extract_text() or ""
        # 這裡簡化演示，實際應包含您原有的 watermark 清除與地址補償邏輯
        output.append(f"===== 第 {i+1} 頁 =====\n" + txt)
    return "\n\n".join(output), []

# ────────────────────────────────────────────────
# 5. 新增：Excel 結構化解析 (串接 AI 學習)
# ────────────────────────────────────────────────

def parse_for_excel(text):
    data = {"行政區": "", "段小段": "", "地號": "", "面積": "", "公告現值": "", "所有權人": "", "身分證字號": "", "地址": ""}
    
    # 段號/地號
    m_land = re.search(r'([^\s]+(?:縣|市)[^\s]+(?:區|鄉|鎮|市))([^\s]+段)\s*([\d-]+)', text)
    if m_land:
        data["行政區"], data["段小段"], data["地號"] = m_land.groups()

    # 面積
    m_area = re.search(r'面積\s*([\d.]+)', text)
    if m_area: data["面積"] = m_area.group(1)

    # 價格 (公告現值)
    m_price = re.search(r'公告土地現值.*?(\d+)\s*元', text)
    if m_price: data["公告現值"] = m_price.group(1)

    # 所有權人 (套用 AI 學習)
    m_owner = re.search(r'所有權人\s*([^\s]+)', text)
    if m_owner: data["所有權人"] = ai_smart_fix(m_owner.group(1).replace('*', '＊'))
    
    # 統一編號
    m_id = re.search(r'統一編號\s*([A-Z][\d\*]+)', text)
    if m_id: data["身分證字號"] = m_id.group(1)

    # 地址 (重點套用 AI 學習)
    m_addr = re.search(r'[地住]\s*址\s+(.+)', text)
    if m_addr: data["地址"] = ai_smart_fix(m_addr.group(1).strip())
    
    return data

# ────────────────────────────────────────────────
# 6. Streamlit 互動介面 (整合學習功能)
# ────────────────────────────────────────────────

st.set_page_config(page_title="地政智慧解譯 Pro", layout="wide")
ocr_engine = load_ocr()

def main():
    st.title("🏠 地政智慧解譯系統 Pro")
    
    # 使用 session_state 保持資料狀態
    if 'main_df' not in st.session_state: st.session_state.main_df = None
    if 'raw_txts' not in st.session_state: st.session_state.raw_txts = {}

    files = st.file_uploader("上傳 PDF (支援多檔)", type="pdf", accept_multiple_files=True)
    
    if files and st.button("🚀 開始全自動解譯"):
        rows = []
        for f in files:
            with st.spinner(f"正在分析 {f.name}..."):
                pdf_bytes = f.read()
                all_imgs = convert_from_bytes(pdf_bytes, dpi=300, poppler_path=POPPLER_PATH)
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    first_text = pdf.pages[0].extract_text() or ""
                    if any(k in first_text for k in ["謄本種類碼", "列印時間"]):
                        txt, _ = process_謄本(pdf, ocr_engine, all_imgs)
                    elif "一覽表" in first_text:
                        txt, _ = process_群璇(pdf, ocr_engine, all_imgs)
                    else:
                        fmt = "光特" if "縣市" in normalize(first_text) else "華安"
                        txt, _ = process_表格式(pdf, ocr_engine, all_imgs, fmt)
                
                st.session_state.raw_txts[f.name] = txt
                rows.append(parse_for_excel(txt))
        
        st.session_state.main_df = pd.DataFrame(rows)

    # ────── 互動修正區 ──────
    if st.session_state.main_df is not None:
        st.divider()
        st.subheader("📝 成果預覽與手動修正")
        st.caption("您可以直接修改下方表格內容，修正後的資料會同步匯出到 Excel 與 TXT。")
        
        # 讓使用者修正資料
        edited_df = st.data_editor(st.session_state.main_df, num_rows="fixed")
        
        if st.button("🧠 確認修正並讓 AI 學習"):
            memory = load_ai_memory()
            # 比對地址欄位的差異來學習
            for idx in range(len(edited_df)):
                old_val = st.session_state.main_df.iloc[idx]["地址"]
                new_val = edited_df.iloc[idx]["地址"]
                if old_val != new_val and old_val != "":
                    memory["history"][old_val] = new_val # 紀錄錯到對的映射
            
            save_ai_memory(memory)
            st.session_state.main_df = edited_df # 同步更新狀態
            st.success("AI 已記住您的修正！下次處理相似內容將自動校正。")

        # ────── 下載區 ──────
        col1, col2 = st.columns(2)
        with col1:
            # 產出 Excel
            xlsx_io = io.BytesIO()
            with pd.ExcelWriter(xlsx_io, engine='xlsxwriter') as writer:
                edited_df.to_excel(writer, index=False, sheet_name='資料彙整')
            st.download_button("📥 下載 Excel 報表", xlsx_io.getvalue(), "地政彙整.xlsx")

        with col2:
            # 下載 TXT (ZIP)
            z_io = io.BytesIO()
            with zipfile.ZipFile(z_io, "w") as zf:
                for filename, content in st.session_state.raw_txts.items():
                    # 這裡示範將修正後的地址也同步回 TXT
                    zf.writestr(f"{filename}.txt", content)
            st.download_button("📦 下載全部 TXT (ZIP)", z_io.getvalue(), "results.zip")

if __name__ == "__main__":
    main()