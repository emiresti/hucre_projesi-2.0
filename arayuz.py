import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import sqlite3
import datetime
import os
from fpdf import FPDF

# --- YENİ ÖZELLİK 1: VERİTABANI FONKSİYONLARI ---
def init_db():
    conn = sqlite3.connect('laboratuvar_gecmisi.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS analiz_kayitlari (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tarih TEXT,
            boya_tipi TEXT,
            toplam_hucre INTEGER,
            saglikli INTEGER,
            atipik INTEGER
        )
    ''')
    conn.commit()
    conn.close()

# Uygulama açıldığında veritabanını kontrol et/oluştur
init_db()

# --- YENİ ÖZELLİK 2: PDF OLUŞTURMA FONKSİYONU ---
def pdf_rapor_olustur(hedef_sayisi, saglikli, atipik, df_rapor, boya_tipi):
    pdf = FPDF()
    pdf.add_page()
    
    # PDF HATA ÇÖZÜMÜ: Emojileri ve Türkçe karakterleri PDF'in anlayacağı şekle çeviriyoruz
    temiz_boya = boya_tipi.replace("🟢", "").replace("🔵", "").replace("🔴", "")
    temiz_boya = temiz_boya.replace("Yeşil", "Yesil").replace("Kırmızı", "Kirmizi").replace("ı", "i").replace("ş", "s").strip()
    
    # PDF'de karakter hatası almamak için Türkçe harfleri İngilizce karşılıklarıyla kullanıyoruz
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Biyoinformatik Hucre Analiz Raporu", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Analiz Tarihi: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, txt=f"Kullanilan Florasan Boya: {temiz_boya}", ln=True)
    pdf.cell(200, 10, txt=f"Toplam Tespit Edilen Hedef: {hedef_sayisi}", ln=True)
    pdf.cell(200, 10, txt=f"Duzenli (Saglikli) Hucre Sayisi: {saglikli}", ln=True)
    pdf.cell(200, 10, txt=f"Deforme (Atipik) Hucre Sayisi: {atipik}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Hucre Morfolojisi Ozeti (Ilk 20 Hedef):", ln=True)
    pdf.set_font("Arial", size=10)
    
    for i, row in df_rapor.head(20).iterrows():
        durum_text = "Duzenli" if "Düzenli" in row['Durum'] else "Deforme"
        pdf.cell(200, 8, txt=f"ID: {row['Hedef ID']} | Alan: {row['Alan (px)']} px | Dairesellik: {row['Dairesellik']} | Durum: {durum_text}", ln=True)
        
    if len(df_rapor) > 20:
        pdf.cell(200, 8, txt="... (Raporun devami CSV dosyasindadir) ...", ln=True)
        
    temp_pdf = "rapor_temp.pdf"
    pdf.output(temp_pdf)
    with open(temp_pdf, "rb") as f:
        pdf_bytes = f.read()
    os.remove(temp_pdf)
    return pdf_bytes

# --- SAYFA TASARIMI VE VERİTABANI MENÜSÜ ---
st.set_page_config(page_title="Hücre Analiz AI", page_icon="🔬", layout="wide")

# Sol Taraftaki Sabit Yan Menü (Sidebar)
with st.sidebar:
    st.header("🗄️ Laboratuvar Geçmişi")
    st.write("Önceki analiz kayıtlarınız SQLite ile tutulmaktadır:")
    conn = sqlite3.connect('laboratuvar_gecmisi.db')
    df_gecmis = pd.read_sql_query("SELECT * FROM analiz_kayitlari ORDER BY id DESC", conn)
    conn.close()
    
    if not df_gecmis.empty:
        st.dataframe(df_gecmis, use_container_width=True)
        st.caption(f"Veritabanında toplam {len(df_gecmis)} analiz kaydı bulunuyor.")
    else:
        st.info("Henüz kaydedilmiş bir analiz yok.")

st.title("🔬 Biyoinformatik Hücre Analiz Dashboard'u")
st.markdown("**U87 Hücre Hattı - Florasan ve Morfoloji Analiz Arayüzü**")
st.markdown("---")

# Gelişmiş Ayarlar (Slider'lar)
with st.expander("⚙️ Gelişmiş Kalibrasyon Ayarları (İsteğe Bağlı)"):
    min_alan = st.slider("Minimum Hücre Alanı (px) - Parazitleri Eler:", min_value=0, max_value=200, value=15, step=5)
    hassasiyet = st.slider("Hücre Ayrıştırma Hassasiyeti (Watershed):", min_value=0.1, max_value=0.9, value=0.4, step=0.1)
st.markdown("---")

st.markdown("### 📥 Görüntü Kaynağı Seçimi")
kaynak = st.radio("Lütfen analiz edilecek görüntünün kaynağını seçin:", ("📁 Bilgisayardan Fotoğraf Yükle", "📸 Canlı Kamera / Mikroskop"))

islem_dosyasi = None
if kaynak == "📁 Bilgisayardan Fotoğraf Yükle":
    islem_dosyasi = st.file_uploader("Lütfen mikroskop görüntüsünü yükleyin (JPG/PNG)", type=["jpg", "jpeg", "png"])
else:
    islem_dosyasi = st.camera_input("Mikroskop veya Web Kameranızdan anlık görüntü alın:")

st.markdown("### 🧪 Analiz Edilecek Florasan Rengi")
secilen_renk = st.selectbox("Hedef boya tipini seçin:", ("🟢 Yeşil (GFP / FITC)", "🔵 Mavi (DAPI / Hoechst)", "🔴 Kırmızı (RFP / Texas Red)"))

if islem_dosyasi is not None:
    image = Image.open(islem_dosyasi)
    img_array = np.array(image)
    resim = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    if st.button("🚀 Yapay Zeka Analizini Başlat"):
        with st.spinner('Görüntü işleniyor, veritabanına kaydediliyor ve rapor hazırlanıyor...'):
            hsv_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2HSV)
            
            if "Yeşil" in secilen_renk:
                maske = cv2.inRange(hsv_resim, np.array([35, 50, 50]), np.array([85, 255, 255]))
                cizim_rengi = (0, 255, 0)
            elif "Mavi" in secilen_renk:
                maske = cv2.inRange(hsv_resim, np.array([110, 50, 50]), np.array([160, 255, 255]))
                cizim_rengi = (255, 0, 0)
            elif "Kırmızı" in secilen_renk:
                maske = cv2.inRange(hsv_resim, np.array([0, 50, 50]), np.array([10, 255, 255])) + cv2.inRange(hsv_resim, np.array([160, 50, 50]), np.array([180, 255, 255]))
                cizim_rengi = (0, 0, 255)
            
            cekirdek = np.ones((3,3), np.uint8)
            temiz_maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, cekirdek, iterations=1)
            mesafe_donusumu = cv2.distanceTransform(temiz_maske, cv2.DIST_L2, 5)
            _, ayrilmis_hucreler = cv2.threshold(mesafe_donusumu, hassasiyet * mesafe_donusumu.max(), 255, 0)
            ayrilmis_hucreler = np.uint8(ayrilmis_hucreler)
            
            konturlar, _ = cv2.findContours(ayrilmis_hucreler, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            kopya_resim = resim.copy()
            hedef_sayisi = 0
            saglikli_sayisi = 0
            atipik_sayisi = 0
            rapor_verileri = []

            for kontur in konturlar:
                alan = cv2.contourArea(kontur)
                if alan > min_alan:
                    hedef_sayisi += 1
                    cevre = cv2.arcLength(kontur, True)
                    dairesellik = 0 if cevre == 0 else 4 * np.pi * (alan / (cevre * cevre))
                    
                    if dairesellik > 0.7:
                        durum = "Düzenli (Sağlıklı)"
                        yazi_rengi = (0, 255, 0)
                        saglikli_sayisi += 1
                    else:
                        durum = "Deforme (Atipik)"
                        yazi_rengi = (0, 0, 255)
                        atipik_sayisi += 1
                    
                    cv2.drawContours(kopya_resim, [kontur], -1, cizim_rengi, 2)
                    M = cv2.moments(kontur)
                    if M["m00"] != 0:
                        cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    else:
                        cX, cY = 0, 0
                    cv2.putText(kopya_resim, str(hedef_sayisi), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, yazi_rengi, 2)
                    
                    rapor_verileri.append({"Hedef ID": hedef_sayisi, "Alan (px)": round(alan, 2), "Dairesellik": round(dairesellik, 3), "Durum": durum})

            if hedef_sayisi > 0:
                conn = sqlite3.connect('laboratuvar_gecmisi.db')
                c = conn.cursor()
                tarih_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                c.execute('INSERT INTO analiz_kayitlari (tarih, boya_tipi, toplam_hucre, saglikli, atipik) VALUES (?, ?, ?, ?, ?)', 
                          (tarih_str, secilen_renk.split(" ")[1], hedef_sayisi, saglikli_sayisi, atipik_sayisi))
                conn.commit()
                conn.close()

        st.success(f"✅ Analiz Tamamlandı ve Veritabanına Kaydedildi! Toplam {hedef_sayisi} Hedef Bulundu. ({saglikli_sayisi} Düzenli, {atipik_sayisi} Deforme)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Orijinal Görüntü", use_container_width=True)
        with col2:
            kopya_resim_rgb = cv2.cvtColor(kopya_resim, cv2.COLOR_BGR2RGB)
            st.image(kopya_resim_rgb, caption="Akıllı Teşhis Görüntüsü (Yeşil No = Düzenli, Kırmızı No = Deforme)", use_container_width=True)

        if hedef_sayisi > 0:
            st.markdown("---")
            st.markdown("### 📊 Hücre Analiz Raporu ve İndirme Seçenekleri")
            df = pd.DataFrame(rapor_verileri)
            
            col_tablo, col_grafik = st.columns([1, 2])
            
            with col_tablo:
                st.dataframe(df, use_container_width=True)
                
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    csv_verisi = df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Excel İndir", data=csv_verisi, file_name='hucre_raporu.csv', mime='text/csv', use_container_width=True)
                
                with btn_col2:
                    pdf_verisi = pdf_rapor_olustur(hedef_sayisi, saglikli_sayisi, atipik_sayisi, df, secilen_renk)
                    st.download_button(label="📄 PDF Raporu İndir", data=pdf_verisi, file_name='biyoinformatik_rapor.pdf', mime='application/pdf', use_container_width=True)
                
            with col_grafik:
                st.bar_chart(df.set_index("Hedef ID")["Alan (px)"])