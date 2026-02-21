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
# 1. Google Sheets 雲端連線核心 (AI 的記憶體)
# ────────────────────────────────────────────────

def get_gsheet_client():
    """透過 Streamlit Secrets 連結 Google Sheets"""
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    try:
        # 從 Secrets 抓取 TOML 格式的金鑰資料
        creds_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        gc = gspread.authorize(creds)
        # 開啟試算表
        return gc.open("地政AI學習庫").sheet1
    except Exception as e:
        st.error(f"⚠️ 雲端連線失敗，請檢查 Secrets 設定: {e}")
        return None

def load_cloud_memory():
    """讀取雲端已學習的修正規則"""
    sheet = get_gsheet_client()
    if sheet:
        try:
            records = sheet.get_all_records()
            return {str(r['wrong']): str(r['right']) for r in records if 'wrong' in r}
        except: return {}
    return {}

def save_to_cloud(wrong, right):
    """將新的學習紀錄寫入雲端"""
    sheet = get_gsheet_client()
    if sheet:
        try:
            sheet.append_row([str(wrong), str(right)])
        except Exception as e:
            st.error(f"寫入雲端失敗: {e}")

def ai_smart_fix(text):
    """應用 AI 學習結果：將 OCR 錯誤字串自動替換為正確字串"""
    if not text: return text
    memory = load_cloud_memory()
    for wrong, right in memory.items():
        if wrong in text:
            text = text.replace(wrong, right)
    return text

# ────────────────────────────────────────────────
# 2. 環境與 OCR 初始化
# ────────────────────────────────────────────────

LOCAL_POPPLER_PATH = r"C:\Users\User\Desktop\pdf_explain new\poppler-25.12.0\Library\bin"
POPPLER_PATH = LOCAL_POPPLER_PATH if os.path.exists(LOCAL_POPPLER_PATH) else None

@st.cache_resource
def load_ocr():
    # 使用 CPU 模式
    return easyocr.Reader(['ch_tra', 'en'], gpu=False)

def normalize(text):
    if not text: return ""
    return unicodedata.normalize("NFKC", re.sub(r'\s+', '', text))

# ────────────────────────────────────────────────
# 3. 核心辨識與解析邏輯 (保留您原有的 300 行大腦)
# ────────────────────────────────────────────────

def fix_addr_post_process(text: str) -> str:
    """基礎地址校正 + AI 智慧校正"""
    if not text: return text
    # 先過一遍 AI 學習過的記憶
    text = ai_smart_fix(text)
    # 基礎常見錯誤置換
    _MAP = {'耋': '臺', '耸': '臺', '孿': '學', '孽': '學', '壆': '學', '覃': '南'}
    for w, r in _MAP.items():
        text = text.replace(w, r)
    return text

def parse_for_excel(text):
    """將 OCR 全文轉為 Excel 結構化欄位"""
    # 確保全文先經過 AI 校正
    text = ai_smart_fix(text)
    
    data = {"行政區": "", "段小段": "", "地號": "", "面積": "", "公告現值": "", "所有權人": "", "身分證字號": "", "地址": ""}
    
    # 1. 抓取地段與地號
    m_land = re.search(r'([^\s]+(?:縣|市)[^\s]+(?:區|鄉|鎮|市))([^\s]+段)\s*([\d-]+)', text)
    if m_land:
        data["行政區"], data["段小段"], data["地號"] = m_land.groups()

    # 2. 面積
    m_area = re.search(r'面積\s*([\d.]+)', text)
    if m_area: data["面積"] = m_area.group(1)

    # 3. 所有權人 (也要 AI 校正)
    m_owner = re.search(r'所有權人\s*([^\s]+)', text)
    if m_owner: 
        owner_name = m_owner.group(1).replace('*', '＊')
        data["所有權人"] = ai_smart_fix(owner_name)
    
    # 4. 身分證
    m_id = re.search(r'統一編號\s*([A-Z][\d\*]+)', text)
    if m_id: data["身分證字號"] = m_id.group(1)

    # 5. 地址 (最需要 AI 學習的地方)
    m_addr = re.search(r'[地住]\s*址\s+(.+)', text)
    if m_addr: 
        data["地址"] = ai_smart_fix(m_addr.group(1).strip())
    
    return data

# ... 此處請放入您原本的 process_謄本, process_群璇, process_表格式 等函式 ...
# 務必確保這些函式內的地址提取部分有調用 ai_smart_fix()

# ────────────────────────────────────────────────
# 4. Streamlit 介面與互動
# ────────────────────────────────────────────────

st.set_page_config(page_title="地政 AI 智慧解譯雲端版", layout="wide")
ocr_reader = load_ocr()

def main():
    st.title("🏠 地政 AI 智慧解譯 (Google 雲端同步學習版)")
    
    # 初始化 session state
    if 'df_results' not in st.session_state: st.session_state.df_results = None
    if 'file_texts' not in st.session_state: st.session_state.file_texts = {}

    with st.sidebar:
        st.header("⚙️ 系統狀態")
        client = get_gsheet_client()
        if client:
            st.success("✅ 雲端記憶體已連線 (Google Sheets)")
        else:
            st.error("❌ 雲端未連線，請檢查 Secrets")

    uploaded_files = st.file_uploader("上傳 PDF (可多選)", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and st.button("🚀 開始智慧解譯"):
        rows = []
        for f in uploaded_files:
            with st.spinner(f"正在深度分析 {f.name}..."):
                pdf_bytes = f.read()
                # 這裡調用您原有的 PDF 處理大腦
                # txt = process_pdf_logic(pdf_bytes, ocr_reader)
                txt = "測試解譯內容：地址 臺南市公孽路一段 1 號" # 模擬產出
                
                st.session_state.file_texts[f.name] = txt
                rows.append(parse_for_excel(txt))
        
        st.session_state.df_results = pd.DataFrame(rows)

    # ────── 5. 互動修正與 AI 訓練區 ──────
    if st.session_state.df_results is not None:
        st.divider()
        st.subheader("📝 成果預覽與手動校正")
        st.caption("直接修改下方表格，修正後的資料會同步反映在下載檔中。點擊「訓練 AI」可讓程式永久記住修正。")
        
        # 讓使用者直接編輯結果
        edited_df = st.data_editor(st.session_state.df_results, num_rows="fixed", key="main_editor")
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("🧠 訓練 AI 記憶"):
                diff_count = 0
                # 比對「地址」欄位，如果使用者有改，就存進 Google Sheets
                for idx in range(len(edited_df)):
                    old_val = str(st.session_state.df_results.iloc[idx]["地址"])
                    new_val = str(edited_df.iloc[idx]["地址"])
                    if old_val != new_val and old_val != "":
                        save_to_cloud(old_val, new_val)
                        diff_count += 1
                
                if diff_count > 0:
                    st.session_state.df_results = edited_df
                    st.success(f"已成功紀錄 {diff_count} 筆修正到雲端！")
                    st.rerun() # 重新整理以載入最新記憶
                else:
                    st.info("沒有偵測到任何欄位變更。")

        # ────── 6. 下載區 ──────
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            # 匯出修正後的 Excel
            output_xlsx = io.BytesIO()
            with pd.ExcelWriter(output_xlsx, engine='xlsxwriter') as writer:
                edited_df.to_excel(writer, index=False, sheet_name='解譯成果')
            st.download_button("📥 下載修正後的 Excel", output_xlsx.getvalue(), "地政智慧報表.xlsx")
        
        with c2:
            # 匯出 TXT (打包成 ZIP)
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for fname, content in st.session_state.file_texts.items():
                    # TXT 也同步套用 AI 校正
                    final_txt = ai_smart_fix(content)
                    zf.writestr(f"{fname}.txt", final_txt)
            st.download_button("📦 下載修正後的 TXT (ZIP)", zip_buffer.getvalue(), "解譯純文字檔.zip")

if __name__ == "__main__":
    main()