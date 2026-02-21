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

# ────────────────────────────────────────────────
#  環境設定：自動偵測環境 (Local vs Cloud)
# ────────────────────────────────────────────────
# 如果在本機測試，請將 Poppler 放在專案資料夾內或設定環境變數
LOCAL_POPPLER_PATH = r"C:\Users\User\Desktop\pdf_explain new\poppler-25.12.0\Library\bin"
POPPLER_PATH = LOCAL_POPPLER_PATH if os.path.exists(LOCAL_POPPLER_PATH) else None
CORRECTIONS_FILE = "addr_corrections.json"

@st.cache_resource
def load_ocr():
    # 在雲端環境 gpu=False 是必須的，除非你有付費升級 GPU 資源
    return easyocr.Reader(['ch_tra', 'en'], gpu=False)

def normalize(text):
    if not text: return ""
    return unicodedata.normalize("NFKC", re.sub(r'\s+', '', text))

# ────────────────────────────────────────────────
#  台灣縣市清單與校正邏輯
# ────────────────────────────────────────────────
TAIWAN_CITIES = [
    '臺北市', '新北市', '桃園市', '臺中市', '臺南市', '高雄市',
    '基隆市', '新竹市', '嘉義市', '新竹縣', '苗栗縣', '彰化縣', 
    '南投縣', '雲林縣', '嘉義縣', '屏東縣', '宜蘭縣', '花蓮縣', 
    '臺東縣', '澎湖縣', '金門縣', '連江縣', '台北市', '台中市', 
    '台南市', '台東縣', '台北縣', '桃園縣', '台中縣', '台南縣', '高雄縣',
]

_CITY_LEVEL = {c: ('市' if c.endswith('市') else '縣') for c in TAIWAN_CITIES}
_DISTRICT_FOR_CITY = ['區']
_DISTRICT_FOR_COUNTY = ['區', '鄉', '鎮', '市']

def load_corrections() -> dict:
    if os.path.exists(CORRECTIONS_FILE):
        try:
            with open(CORRECTIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_correction(wrong: str, right: str):
    corrections = load_corrections()
    corrections[wrong.strip()] = right.strip()
    with open(CORRECTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2)

def apply_corrections(text: str) -> str:
    corrections = load_corrections()
    return corrections.get(text.strip(), text)

def validate_addr_prefix(text: str) -> bool:
    return any(text.startswith(city) for city in TAIWAN_CITIES)

def check_addr_city_district(text: str) -> tuple:
    if not text or len(text) < 6:
        return True, ""
    matched_city = next((city for city in TAIWAN_CITIES if text.startswith(city)), None)
    if not matched_city:
        return False, f"無法識別縣市名稱（開頭：{text[:3]}）"
    
    rest = text[len(matched_city):]
    district_char = next((ch for ch in rest if ch in ['區', '鄉', '鎮']), None)
    if district_char is None:
        return True, ""

    level = _CITY_LEVEL.get(matched_city, '')
    if level == '市' and district_char not in _DISTRICT_FOR_CITY:
        return False, f"層級錯誤：「{matched_city}」應配「區」"
    if level == '縣' and district_char not in _DISTRICT_FOR_COUNTY:
        return False, f"層級錯誤：「{matched_city}」行政區不應為「{district_char}」"
    return True, ""

def fix_addr_prefix(text: str) -> tuple:
    if not text or len(text) < 3:
        return text, False
    if validate_addr_prefix(text):
        return text, False
    prefix = text[:3]
    best_match, best_score = None, 0.0
    for city in TAIWAN_CITIES:
        score = difflib.SequenceMatcher(None, prefix, city[:3]).ratio()
        if score > best_score:
            best_score, best_match = score, city
    if best_match and best_score >= 0.6:
        return best_match[:3] + text[3:], True
    return text, False

_ADDR_CHAR_MAP = {'耋': '臺', '耸': '臺', '孿': '學', '孽': '學', '壆': '學', '覃': '南'}

def fix_addr_post_process(text: str) -> str:
    if not text: return text
    text = apply_corrections(text.strip())
    for wrong, right in _ADDR_CHAR_MAP.items():
        text = text.replace(wrong, right)
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    _ADDR_CJK = r'[里鄰路段巷弄號街區市縣鄉鎮村]'
    text = re.sub(rf'({_ADDR_CJK})\s+(\d)', r'\1\2', text)
    text = re.sub(rf'(\d)\s+({_ADDR_CJK})', r'\1\2', text)
    text, _ = fix_addr_prefix(text)
    return text

# ────────────────────────────────────────────────
#  OCR 處理策略
# ────────────────────────────────────────────────
def preprocess_for_ocr(img_gray: np.ndarray) -> list:
    imgs = []
    # 策略 1 & 2 & 3
    for mode in ['normal', 'clahe', 'sharp']:
        processed = img_gray.copy()
        if mode == 'clahe':
            processed = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(processed)
        elif mode == 'sharp':
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            processed = cv2.filter2D(processed, -1, kernel)
        
        big = cv2.resize(processed, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
        imgs.append(cv2.copyMakeBorder(big, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255))
    return imgs

def ocr_with_best_result(ocr_model, img_gray: np.ndarray) -> tuple:
    strategy_names = ['原始放大', 'CLAHE增強', '銳化']
    candidates = []
    for i, img in enumerate(preprocess_for_ocr(img_gray)):
        results = ocr_model.readtext(img, detail=1, paragraph=False)
        raw = "".join([res[1] for res in results if normalize(res[1]) not in ['地址', '住址']]).strip()
        processed = fix_addr_post_process(raw)
        candidates.append((processed, strategy_names[i]))

    def score_result(item):
        txt, _ = item
        s = 0
        if validate_addr_prefix(txt): s += 2
        ok, _ = check_addr_city_district(txt)
        if ok: s += 2
        if len(txt) > 5: s += 1
        return s
    return max(candidates, key=score_result) if candidates else ("", "無結果")

# ────────────────────────────────────────────────
#  PDF 解析核心
# ────────────────────────────────────────────────
def extract_addr_from_image_stream(page, ocr_model, debug_log: list):
    words = page.extract_words()
    addr_word = next((w for w in words if w['text'] in ['地址', '住址']), None)
    if not addr_word: return ""
    
    label = "住" if addr_word['text'] == '住址' else "地"
    addr_imgs = [img for img in page.images if abs(img['top'] - addr_word['top']) < 5]
    if not addr_imgs: return ""

    try:
        raw_data = addr_imgs[0]['stream'].get_data()
        buf = np.frombuffer(raw_data, dtype=np.uint8)
        decoded = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if decoded is None: return ""
        addr_val, strat = ocr_with_best_result(ocr_model, decoded)
        debug_log.append(f"✅ Stream 成功({strat}): {addr_val}")
        return f"{label} 址 {addr_val}"
    except Exception as e:
        debug_log.append(f"❌ Stream 失敗: {e}")
        return ""

def ocr_addr_fallback(img_np, page, ocr_model, debug_log: list):
    h, w = img_np.shape[:2]
    sy, sx = h/page.height, w/page.width
    words = page.extract_words()
    addr_word = next((w for w in words if w['text'] in ['地址', '住址']), None)
    if not addr_word: return "[定位失敗]"

    next_words = [w for w in words if w['top'] > addr_word['bottom'] + 1]
    row_bottom = next_words[0]['top'] if next_words else addr_word['bottom'] + 20
    crop = img_np[max(0, int((addr_word['top']-2)*sy)):min(h, int((row_bottom+2)*sy)), int(175*sx):]
    
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    addr_val, strat = ocr_with_best_result(ocr_model, gray)
    debug_log.append(f"⚠️ 備援成功({strat}): {addr_val}")
    return f"{('住' if addr_word['text']=='住址' else '地')} 址 {addr_val or '[無法辨識]'}"

def detect_format(pdf):
    text = pdf.pages[0].extract_text() or ""
    if any(k in text for k in ["謄本種類碼", "列印時間"]): return "謄本"
    if "一覽表" in text: return "群璇"
    if "縣市" in normalize(text): return "光特"
    return "華安"

# ... (其餘 process_群璇, process_謄本, process_表格式 邏輯保持不變，但調用優化後的 OCR 函數) ...
# [此處為了簡潔，略過重複的表格解析邏輯，請確保調用時使用 ocr_with_best_result]

# ────────────────────────────────────────────────
#  Streamlit 主介面
# ────────────────────────────────────────────────
st.set_page_config(page_title="地政文件透視器", layout="wide")
ocr_model = load_ocr()

st.title("🏠 地政文件透視器")

uploaded_files = st.file_uploader("請上傳 PDF (最多 5 個)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.error("最多 5 個檔案")
    elif st.button("🚀 開始處理"):
        results = {}
        for uploaded_file in uploaded_files:
            with st.spinner(f"正在處理 {uploaded_file.name}..."):
                pdf_bytes = uploaded_file.read()
                # 這裡調用你原本的 process_pdf，但內部確保使用優化過的函數
                # 為節省空間，請將你原有的 process_ 函數群組放回此處
                # 記得 convert_from_bytes(pdf_bytes, dpi=300, poppler_path=POPPLER_PATH)
                st.success(f"{uploaded_file.name} 處理完成")