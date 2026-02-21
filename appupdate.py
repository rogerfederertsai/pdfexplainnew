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
# 1. 環境適應與資源載入
# ────────────────────────────────────────────────

# 自動偵測 Poppler 路徑
LOCAL_POPPLER_PATH = r"C:\Users\User\Desktop\pdf_explain new\poppler-25.12.0\Library\bin"
POPPLER_PATH = LOCAL_POPPLER_PATH if os.path.exists(LOCAL_POPPLER_PATH) else None
CORRECTIONS_FILE = "addr_corrections.json"

@st.cache_resource
def load_ocr():
    # 雲端環境通常沒有 GPU，設定為 False 以免報錯
    return easyocr.Reader(['ch_tra', 'en'], gpu=False)

def normalize(text):
    if not text: return ""
    return unicodedata.normalize("NFKC", re.sub(r'\s+', '', text))

# ────────────────────────────────────────────────
# 2. 地址驗證與校正邏輯
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
        except: return {}
    return {}

def save_correction(wrong: str, right: str):
    corrections = load_corrections()
    corrections[wrong.strip()] = right.strip()
    with open(CORRECTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2)

def apply_corrections(text: str) -> str:
    return load_corrections().get(text.strip(), text)

def validate_addr_prefix(text: str) -> bool:
    return any(text.startswith(city) for city in TAIWAN_CITIES)

def check_addr_city_district(text: str) -> tuple:
    if not text or len(text) < 6: return True, ""
    matched_city = next((city for city in TAIWAN_CITIES if text.startswith(city)), None)
    if not matched_city: return False, f"無法識別縣市名稱（{text[:3]}）"
    
    rest = text[len(matched_city):]
    district_char = next((ch for ch in rest if ch in ['區', '鄉', '鎮']), None)
    if not district_char: return True, ""

    level = _CITY_LEVEL.get(matched_city, '')
    if level == '市' and district_char not in _DISTRICT_FOR_CITY:
        return False, f"層級錯誤：「{matched_city}」配「{district_char}」（應為區）"
    if level == '縣' and district_char not in _DISTRICT_FOR_COUNTY:
        return False, f"層級錯誤：「{matched_city}」配「{district_char}」"
    return True, ""

def fix_addr_prefix(text: str) -> tuple:
    if not text or len(text) < 3 or validate_addr_prefix(text):
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
# 3. OCR 核心與多策略辨識
# ────────────────────────────────────────────────

def preprocess_for_ocr(img_gray: np.ndarray) -> list:
    imgs = []
    # 調整倍率為 4 倍，兼顧雲端效能與準確率
    big_size = (None, None) 
    fx, fy = 4, 4
    
    # 策略 1: 原始放大
    b1 = cv2.resize(img_gray, None, fx=fx, fy=fy, interpolation=cv2.INTER_LANCZOS4)
    imgs.append(cv2.copyMakeBorder(b1, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255))
    
    # 策略 2: CLAHE 增強
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_gray)
    b2 = cv2.resize(clahe, None, fx=fx, fy=fy, interpolation=cv2.INTER_LANCZOS4)
    imgs.append(cv2.copyMakeBorder(b2, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255))
    
    return imgs

def ocr_with_best_result(ocr, img_gray: np.ndarray) -> tuple:
    strategies = ['原始', 'CLAHE']
    candidates = []
    for i, img in enumerate(preprocess_for_ocr(img_gray)):
        results = ocr.readtext(img, detail=1, paragraph=False)
        raw = "".join([res[1] for res in results if normalize(res[1]) not in ['地址', '住址']]).strip()
        processed = fix_addr_post_process(raw)
        candidates.append((processed, strategies[i]))

    def score(item):
        t, _ = item
        s = 0
        if validate_addr_prefix(t): s += 2
        ok, _ = check_addr_city_district(t)
        if ok: s += 2
        if len(t) > 5: s += 1
        return s
    return max(candidates, key=score) if candidates else ("", "無結果")

# ────────────────────────────────────────────────
# 4. 文件解析邏輯 (表格、電傳、謄本)
# ────────────────────────────────────────────────

def clean_watermark(text):
    lines = text.split("\n")
    cleaned = []
    watermark_chars = set("臺南市新化地政事務所")
    for line in lines:
        s = line.strip()
        if s == "H0" or (len(s) == 1 and s in watermark_chars): continue
        line = re.sub(r'\s+[臺南市新化地政事務所]{1,2}\s*$', '', line)
        line = re.sub(r'臺(一般農業區|都市發展區|農業區)', r'\1', line)
        cleaned.append(line)
    return "\n".join(cleaned)

def extract_addr_from_image_stream(page, ocr, debug_log: list):
    words = page.extract_words()
    target = next((w for w in words if w['text'] in ['地址', '住址']), None)
    if not target: return ""
    
    label = "住" if target['text'] == '住址' else "地"
    addr_imgs = [img for img in page.images if abs(img['top'] - target['top']) < 5]
    if not addr_imgs: return ""

    try:
        raw = addr_imgs[0]['stream'].get_data()
        buf = np.frombuffer(raw, dtype=np.uint8)
        decoded = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        val, strat = ocr_with_best_result(ocr, decoded)
        debug_log.append(f"✅ Stream成功({strat}): {val}")
        return f"{label} 址 {val}"
    except: return ""

def ocr_addr_fallback(img_np, page, ocr, debug_log: list):
    h, w = img_np.shape[:2]
    sy, sx = h/page.height, w/page.width
    words = page.extract_words()
    target = next((w for w in words if w['text'] in ['地址', '住址']), None)
    if not target: return "[定位失敗]"

    next_w = [w for w in words if w['top'] > target['bottom'] + 1]
    bottom = next_w[0]['top'] if next_w else target['bottom'] + 20
    crop = img_np[max(0, int((target['top']-2)*sy)):min(h, int((bottom+2)*sy)), int(175*sx):]
    
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    val, strat = ocr_with_best_result(ocr, gray)
    debug_log.append(f"⚠️ 備援成功({strat}): {val}")
    return f"{('住' if target['text']=='住址' else '地')} 址 {val or '[無法辨識]'}"

# 格式處理函數群
def process_表格式(pdf, ocr, all_imgs, fmt):
    output, debug = [], []
    for i, page in enumerate(pdf.pages):
        page_text = []
        tables = page.extract_tables()
        if tables:
            for table in tables:
                for row in table:
                    cells = [c.strip().replace("\n", "") if c else "" for c in row]
                    if not any(cells): continue
                    
                    # 判斷是否為空地址列
                    is_addr = normalize(cells[0]) in ["地址", "住址"]
                    has_content = any(c.strip() for c in cells[1:])
                    
                    if is_addr and not has_content:
                        line = extract_addr_from_image_stream(page, ocr, debug)
                        if not line:
                            line = ocr_addr_fallback(np.array(all_imgs[i]), page, ocr, debug)
                    else:
                        line = "  ".join(c for c in cells if c)
                    page_text.append(line)
        output.append(f"===== 第 {i+1} 頁 =====\n" + "\n".join(page_text))
    return "\n\n".join(output), debug

def process_群璇(pdf, ocr, all_imgs):
    output = []
    for i, page in enumerate(pdf.pages):
        lines = []
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                cells = [c.strip().replace("\n", "") if c else "" for c in row]
                if not any(cells) or any(x in "".join(cells) for x in ["一覽表", "列印"]): continue
                if len(cells) >= 2 and normalize(cells[0]) in ["地址", "住址"]:
                    lines.append(f"地  址  {cells[1]}")
                else:
                    lines.append("  ".join(c for c in cells if c))
        output.append(f"===== 第 {i+1} 頁 =====\n" + "\n".join(lines))
    return "\n\n".join(output), []

def process_謄本(pdf, ocr, all_imgs):
    output, debug = [], []
    for i, page in enumerate(pdf.pages):
        raw = clean_watermark(page.extract_text() or "")
        lines = raw.split("\n")
        res_lines = []
        for j, line in enumerate(lines):
            res_lines.append(line)
            if "所有權人" in line:
                nxt = lines[j+1].strip() if j+1 < len(lines) else ""
                if "址" not in nxt and "統一編號" not in nxt:
                    img_np = np.array(all_imgs[i])
                    h, w = img_np.shape[:2]
                    scale = h/page.height
                    words = page.extract_words()
                    y = next((wd["top"] for wd in words if "所有權人" in wd["text"]), None)
                    if y:
                        crop = img_np[int((y+10)*scale):int((y+70)*scale), :int(w*0.85)]
                        val, strat = ocr_with_best_result(ocr, cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY))
                        if val: res_lines.append(f" 住  址：{val.replace('住址：','')}")
        output.append(f"===== 第 {i+1} 頁 =====\n" + "\n".join(res_lines))
    return "\n\n".join(output), debug

# ────────────────────────────────────────────────
# 5. 主入口與 Streamlit UI
# ────────────────────────────────────────────────

st.set_page_config(page_title="地政文件透視器", layout="wide")
ocr = load_ocr()

def main():
    st.title("🏠 地政文件透視器")
    
    with st.sidebar:
        st.header("⚙️ 設定")
        show_debug = st.checkbox("顯示除錯資訊")
        if st.button("🧹 清除暫存"):
            st.cache_resource.clear()
            st.rerun()

    files = st.file_uploader("上傳 PDF", type="pdf", accept_multiple_files=True)
    
    if files and st.button("🚀 開始處理"):
        all_results = {}
        for f in files[:5]:
            with st.spinner(f"正在處理 {f.name}..."):
                pdf_bytes = f.read()
                all_imgs = convert_from_bytes(pdf_bytes, dpi=300, poppler_path=POPPLER_PATH)
                
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    text = pdf.pages[0].extract_text() or ""
                    if any(k in text for k in ["謄本種類碼", "列印時間"]):
                        txt, dbg = process_謄本(pdf, ocr, all_imgs)
                    elif "一覽表" in text:
                        txt, dbg = process_群璇(pdf, ocr, all_imgs)
                    else:
                        fmt = "光特" if "縣市" in normalize(text) else "華安"
                        txt, dbg = process_表格式(pdf, ocr, all_imgs, fmt)
                
                all_results[f.name] = txt
                st.success(f"✅ {f.name} 完成")
                
                if show_debug:
                    with st.expander(f"🔍 {f.name} 除錯日誌"):
                        for d in dbg: st.text(d)
                
                st.text_area(f"預覽: {f.name}", txt, height=200)
                st.download_button(f"下載 {f.name}.txt", txt, f"{f.name}.txt")

        if len(all_results) > 1:
            z_buf = io.BytesIO()
            with zipfile.ZipFile(z_buf, "w") as zf:
                for n, c in all_results.items(): zf.writestr(f"{n}.txt", c)
            st.download_button("📦 下載全部 (ZIP)", z_buf.getvalue(), "results.zip")

if __name__ == "__main__":
    main()