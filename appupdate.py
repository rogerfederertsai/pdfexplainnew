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
import pandas as pd
import gspread
import os
import difflib
from google.oauth2.service_account import Credentials

# ────────────────────────────────────────────────
# 1. 雲端記憶模組 (確保不快取，每次都抓最新)
# ────────────────────────────────────────────────
def get_gsheet_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    try:
        creds_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        gc = gspread.authorize(creds)
        return gc.open("地政AI學習庫").sheet1
    except Exception as e:
        return None

def load_cloud_memory():
    """強制抓取雲端最新資料，不使用 cache"""
    sheet = get_gsheet_client()
    if sheet:
        try:
            records = sheet.get_all_records()
            return {str(r['wrong']): str(r['right']) for r in records if 'wrong' in r}
        except:
            return {}
    return {}

def save_to_cloud(wrong, right):
    sheet = get_gsheet_client()
    if sheet:
        try:
            sheet.append_row([str(wrong), str(right)])
        except:
            pass

def ai_smart_fix(text, memory):
    """應用傳入的最新 memory 進行替換"""
    if not text or not memory: return text
    # 優先替換長的關鍵字（環境關鍵字），確保精準度
    sorted_keys = sorted(memory.keys(), key=len, reverse=True)
    for wrong_key in sorted_keys:
        if wrong_key in text:
            text = text.replace(wrong_key, str(memory[wrong_key]))
    return text

# ────────────────────────────────────────────────
# 2. 原有成功辨識邏輯 (完全保留不動)
# ────────────────────────────────────────────────
LOCAL_POPPLER_PATH = r"C:\Users\User\Desktop\pdf_explain new\poppler-25.12.0\Library\bin"
POPPLER_PATH = LOCAL_POPPLER_PATH if os.path.exists(LOCAL_POPPLER_PATH) else None

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ch_tra', 'en'], gpu=False)

def normalize(text):
    if not text: return ""
    return unicodedata.normalize("NFKC", re.sub(r'\s+', '', text))

# [保留您原有的 fix_addr_post_process, ocr_with_best_result, extract_addr_from_image_stream]
def fix_addr_post_process(text: str) -> str:
    if not text: return text
    _ADDR_CHAR_MAP = {'耋': '臺', '耸': '臺', '孿': '學', '孽': '學', '壆': '學', '覃': '南'}
    for wrong, right in _ADDR_CHAR_MAP.items():
        text = text.replace(wrong, right)
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    _ADDR_CJK = r'[里鄰路段巷弄號街區市縣鄉鎮村]'
    text = re.sub(rf'({_ADDR_CJK})\s+(\d)', r'\1\2', text)
    text = re.sub(rf'(\d)\s+({_ADDR_CJK})', r'\1\2', text)
    return text

# [保留原有的三大解析處理器 process_表格式, process_群璇, process_謄本]
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

def ocr_with_best_result(ocr, img_gray: np.ndarray) -> tuple:
    fx, fy = 4, 4
    b1 = cv2.resize(img_gray, None, fx=fx, fy=fy, interpolation=cv2.INTER_LANCZOS4)
    results = ocr.readtext(b1, detail=0)
    raw = "".join(results).strip()
    processed = fix_addr_post_process(raw)
    return processed, "Standard"

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

# ────────────────────────────────────────────────
# 3. Excel 解析 (在此處立即套用最新記憶)
# ────────────────────────────────────────────────
def parse_for_excel(text, memory):
    # 重點：在解析欄位前，先拿最新記憶刷洗一次文字
    text = ai_smart_fix(text, memory)
    
    data = {
        "行政區/段": "", "地號": "", "面積(m2)": "", 
        "公告土地現值": "", "所有權人": "", "統一編號": "", "地址": ""
    }
    m_loc = re.search(r'([^\s]+(?:縣|市)[^\s]+(?:區|鄉|鎮|市)[^\s]+段)', text)
    if m_loc: data["行政區/段"] = m_loc.group(1)
    
    m_no = re.search(r'(\d{4}-\d{4})', text)
    if m_no: data["地號"] = m_no.group(1)
    
    m_area = re.search(r'面積\s*[,，]?\s*([\d.]+)', text)
    if m_area: data["面積(m2)"] = m_area.group(1)
    
    m_price = re.search(r'公告土地現值.*?(\d+)\s*元', text)
    if m_price: data["公告土地現值"] = m_price.group(1)
    
    m_owner = re.search(r'所有權人\s*[,，]?\s*([^\s,，]+)', text)
    if m_owner: data["所有權人"] = m_owner.group(1).replace('*', '＊')
    
    m_id = re.search(r'統一編號\s*[,，]?\s*([A-Z\d\*]+)', text)
    if m_id: data["統一編號"] = m_id.group(1)

    m_addr = re.search(r'地\s*址\s*[,，]?\s*(.+)', text)
    if m_addr: data["地址"] = m_addr.group(1).strip()
    
    return data

# ────────────────────────────────────────────────
# 4. Streamlit 主流程
# ────────────────────────────────────────────────
st.set_page_config(page_title="地政解譯穩定版", layout="wide")
ocr_engine = load_ocr()

def main():
    st.title("🏠 地政 AI 智慧解譯 (同步刷新版)")
    
    if 'main_df' not in st.session_state: st.session_state.main_df = None
    if 'raw_txts' not in st.session_state: st.session_state.raw_txts = {}

    files = st.file_uploader("上傳 PDF", type="pdf", accept_multiple_files=True)
    
    if files and st.button("🚀 開始解譯"):
        # 強制在此刻重新讀取雲端，保證「秒記」
        latest_mem = load_cloud_memory()
        rows = []
        for f in files:
            with st.spinner(f"分析中: {f.name}"):
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
                
                # 存入時也同步修正 raw_txts
                fixed_txt = ai_smart_fix(txt, latest_mem)
                st.session_state.raw_txts[f.name] = fixed_txt
                rows.append(parse_for_excel(txt, latest_mem))
        st.session_state.main_df = pd.DataFrame(rows)

    if st.session_state.main_df is not None:
        st.subheader("📝 成果修正與 AI 學習")
        edited_df = st.data_editor(st.session_state.main_df, num_rows="fixed")
        
        if st.button("🧠 儲存修正 (動態窗口學習)"):
            for idx in range(len(edited_df)):
                for col in ["地址", "所有權人"]:
                    old_v = str(st.session_state.main_df.iloc[idx][col])
                    new_v = str(edited_df.iloc[idx][col])
                    if old_v != new_v and old_v != "":
                        # 動態窗口找出連續錯誤區塊
                        diff = list(difflib.ndiff(old_v, new_v))
                        i = 0
                        while i < len(diff):
                            if diff[i].startswith('- '):
                                wrong_block, right_block = "", ""
                                start_idx = i
                                while i < len(diff) and (diff[i].startswith('- ') or diff[i].startswith('+ ')):
                                    if diff[i].startswith('- '): wrong_block += diff[i][2:]
                                    if diff[i].startswith('+ '): right_block += diff[i][2:]
                                    i += 1
                                prefix = diff[start_idx-1][2:] if start_idx > 0 and diff[start_idx-1].startswith('  ') else ""
                                suffix = diff[i][2:] if i < len(diff) and diff[i].startswith('  ') else ""
                                save_to_cloud(f"{prefix}{wrong_block}{suffix}", f"{prefix}{right_block}{suffix}")
                            else: i += 1
            st.session_state.main_df = edited_df
            st.success("AI 學習完成！下次解譯將自動修正。")

        # ────── 下載區 (下載時再次強制確認) ──────
        c1, c2 = st.columns(2)
        with c1:
            xlsx_io = io.BytesIO()
            edited_df.to_excel(xlsx_io, index=False)
            st.download_button("📥 下載 Excel", xlsx_io.getvalue(), "地政彙整.xlsx")
        
        with c2:
            z_io = io.BytesIO()
            # 下載前再刷一次最新的雲端
            final_mem = load_cloud_memory() 
            with zipfile.ZipFile(z_io, "w") as zf:
                for fname, content in st.session_state.raw_txts.items():
                    zf.writestr(f"{fname}.txt", ai_smart_fix(content, final_mem))
            st.download_button("📦 下載修正後 TXT (ZIP)", z_io.getvalue(), "results.zip")

if __name__ == "__main__":
    main()