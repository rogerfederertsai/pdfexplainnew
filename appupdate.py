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
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# ────────────────────────────────────────────────
# 1. Google Sheets 雲端連線模組 (取代原本的 JSON 檔)
# ────────────────────────────────────────────────

def get_gsheet_client():
    """透過 Streamlit Secrets 連結 Google Sheets"""
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    try:
        # 從 Secrets 讀取您填寫的 TOML 金鑰
        creds_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        gc = gspread.authorize(creds)
        # 開啟試算表 (請確保試算表名稱正確)
        return gc.open("地政AI學習庫").sheet1
    except Exception as e:
        return None

def load_cloud_memory():
    """從雲端讀取所有學習過的修正紀錄"""
    sheet = get_gsheet_client()
    if sheet:
        try:
            records = sheet.get_all_records()
            # 建立對照字典，例如 {"公孽路": "公學路"}
            return {str(r['wrong']): str(r['right']) for r in records if 'wrong' in r}
        except: return {}
    return {}

def save_to_cloud(wrong, right):
    """將修正結果永久存入 Google Sheets"""
    sheet = get_gsheet_client()
    if sheet:
        try:
            sheet.append_row([str(wrong), str(right)])
        except: pass

def ai_smart_fix(text):
    """自動套用 AI 學習過的修正行為 (外掛式不影響原邏輯)"""
    if not text: return text
    memory = load_cloud_memory()
    for wrong, right in memory.items():
        if str(wrong) in text:
            text = text.replace(str(wrong), str(right))
    return text

# ────────────────────────────────────────────────
# 2. 環境適應與資源載入 (保留原邏輯)
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
# 3. 核心 OCR 策略與地址校正 (您的原邏輯)
# ────────────────────────────────────────────────

def fix_addr_post_process(text: str) -> str:
    if not text: return text
    # 先套用 AI 雲端學習結果
    text = ai_smart_fix(text)
    # 執行您原本成功的硬編碼校正
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
# 4. 文件解析邏輯 (您的原邏輯：謄本、群璇、表格式)
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
        output.append(f"===== 第 {i+1} 頁 =====\n" + txt)
    return "\n\n".join(output), []

# ────────────────────────────────────────────────
# 5. Excel 結構化解析 (串接 AI 雲端學習)
# ────────────────────────────────────────────────

def parse_for_excel(text):
    # 解析前先套用雲端記憶
    text = ai_smart_fix(text)
    
    data = {"行政區": "", "段小段": "", "地號": "", "面積": "", "公告現值": "", "所有權人": "", "身分證字號": "", "地址": ""}
    m_land = re.search(r'([^\s]+(?:縣|市)[^\s]+(?:區|鄉|鎮|市))([^\s]+段)\s*([\d-]+)', text)
    if m_land: data["行政區"], data["段小段"], data["地號"] = m_land.groups()

    m_area = re.search(r'面積\s*([\d.]+)', text)
    if m_area: data["面積"] = m_area.group(1)

    m_owner = re.search(r'所有權人\s*([^\s]+)', text)
    if m_owner: 
        owner_name = m_owner.group(1).replace('*', '＊')
        data["所有權人"] = ai_smart_fix(owner_name)
    
    m_addr = re.search(r'[地住]\s*址\s+(.+)', text)
    if m_addr: data["地址"] = ai_smart_fix(m_addr.group(1).strip())
    
    return data

# ────────────────────────────────────────────────
# 6. Streamlit 介面 (雲端記憶穩定版)
# ────────────────────────────────────────────────

st.set_page_config(page_title="地政智慧解譯雲端版", layout="wide")
ocr_engine = load_ocr()

def main():
    st.title("🏠 地政智慧解譯系統 (雲端永久學習版)")
    
    if 'main_df' not in st.session_state: st.session_state.main_df = None
    if 'raw_txts' not in st.session_state: st.session_state.raw_txts = {}

    with st.sidebar:
        st.header("⚙️ 雲端狀態")
        if get_gsheet_client():
            st.success("✅ 雲端記憶庫已連線")
        else:
            st.error("❌ 雲端連線失敗 (請檢查 Secrets)")

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
        
        # 使用使用者直接修改的結果
        edited_df = st.data_editor(st.session_state.main_df, num_rows="fixed")
        
        if st.button("🧠 確認修正並訓練 AI (永久儲存)"):
            # 找出「地址」或「所有權人」的變動
            for idx in range(len(edited_df)):
                for col in ["地址", "所有權人"]:
                    old_v = str(st.session_state.main_df.iloc[idx][col])
                    new_v = str(edited_df.iloc[idx][col])
                    if old_v != new_v and old_v != "":
                        save_to_cloud(old_v, new_v) # 存入 Google Sheets
            
            st.session_state.main_df = edited_df
            st.success("🎉 AI 學習完成！修正結果已存入雲端。")
            st.rerun()

        # ────── 下載區 ──────
        col1, col2 = st.columns(2)
        with col1:
            xlsx_io = io.BytesIO()
            with pd.ExcelWriter(xlsx_io, engine='xlsxwriter') as writer:
                edited_df.to_excel(writer, index=False, sheet_name='資料彙整')
            st.download_button("📥 下載 Excel 報表", xlsx_io.getvalue(), "地政彙整.xlsx")

        with col2:
            z_io = io.BytesIO()
            with zipfile.ZipFile(z_io, "w") as zf:
                for filename, content in st.session_state.raw_txts.items():
                    # TXT 也套用 AI 修正後產出
                    zf.writestr(f"{filename}.txt", ai_smart_fix(content))
            st.download_button("📦 下載全部 TXT (ZIP)", z_io.getvalue(), "results.zip")

if __name__ == "__main__":
    main()