import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# 1. Sayfa Tasarımı ve Düzeni
st.set_page_config(page_title="Hücre Analiz AI", page_icon="🔬", layout="wide")
st.title("🔬 Biyoinformatik Hücre Analiz Dashboard'u")
st.markdown("**U87 Hücre Hattı - Florasan ve Morfoloji Analiz Arayüzü**")
st.markdown("---")

# Gelişmiş Ayarlar (Slider'lar)
with st.expander("⚙️ Gelişmiş Kalibrasyon Ayarları (İsteğe Bağlı)"):
    st.markdown("Buradan yapay zekanın keskinliğini ve eleyeceği çöp boyutunu anlık ayarlayabilirsiniz.")
    min_alan = st.slider("Minimum Hücre Alanı (px) - Parazitleri Eler:", min_value=0, max_value=200, value=15, step=5)
    hassasiyet = st.slider("Hücre Ayrıştırma Hassasiyeti (Watershed):", min_value=0.1, max_value=0.9, value=0.4, step=0.1)
st.markdown("---")

# --- ÖZELLİK 1: GÖRÜNTÜ KAYNAĞI SEÇİMİ (KAMERA VS FOTOĞRAF) ---
st.markdown("### 📥 Görüntü Kaynağı Seçimi")
kaynak = st.radio("Lütfen analiz edilecek görüntünün kaynağını seçin:", ("📁 Bilgisayardan Fotoğraf Yükle", "📸 Canlı Kamera / Mikroskop"))

islem_dosyasi = None
if kaynak == "📁 Bilgisayardan Fotoğraf Yükle":
    islem_dosyasi = st.file_uploader("Lütfen mikroskop görüntüsünü yükleyin (JPG/PNG)", type=["jpg", "jpeg", "png"])
else:
    islem_dosyasi = st.camera_input("Mikroskop veya Web Kameranızdan anlık görüntü alın:")

# Renk Seçim Menüsü
st.markdown("### 🧪 Analiz Edilecek Florasan Rengi")
secilen_renk = st.selectbox("Hedef boya tipini seçin:", ("🟢 Yeşil (GFP / FITC)", "🔵 Mavi (DAPI / Hoechst)", "🔴 Kırmızı (RFP / Texas Red)"))

if islem_dosyasi is not None:
    image = Image.open(islem_dosyasi)
    img_array = np.array(image)
    resim = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    if st.button("🚀 Yapay Zeka Analizini Başlat"):
        with st.spinner('Görüntü işleniyor, teşhis konuluyor ve morfolojik veriler hesaplanıyor...'):
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
                    
                    dairesellik = 0
                    if cevre > 0:
                        dairesellik = 4 * np.pi * (alan / (cevre * cevre))
                    
                    # --- ÖZELLİK 2: AKILLI SINIFLANDIRMA VE TEŞHİS ---
                    if dairesellik > 0.7:
                        durum = "Düzenli (Sağlıklı)"
                        yazi_rengi = (0, 255, 0) # Yeşil yazı
                        saglikli_sayisi += 1
                    else:
                        durum = "Deforme (Atipik)"
                        yazi_rengi = (0, 0, 255) # Kırmızı yazı
                        atipik_sayisi += 1
                    
                    cv2.drawContours(kopya_resim, [kontur], -1, cizim_rengi, 2)
                    
                    M = cv2.moments(kontur)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = 0, 0
                    
                    # Hücre numarasını durumuna göre yeşil veya kırmızı yazar
                    cv2.putText(kopya_resim, str(hedef_sayisi), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, yazi_rengi, 2)
                    
                    rapor_verileri.append({
                        "Hedef ID": hedef_sayisi,
                        "Alan (px)": round(alan, 2),
                        "Dairesellik": round(dairesellik, 3),
                        "Durum": durum
                    })

        st.success(f"✅ Analiz Tamamlandı! Toplam {hedef_sayisi} Hedef Bulundu. ({saglikli_sayisi} Düzenli, {atipik_sayisi} Deforme)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Orijinal Görüntü", use_container_width=True)
        with col2:
            kopya_resim_rgb = cv2.cvtColor(kopya_resim, cv2.COLOR_BGR2RGB)
            st.image(kopya_resim_rgb, caption="Akıllı Teşhis Görüntüsü (Yeşil No = Düzenli, Kırmızı No = Deforme)", use_container_width=True)

        if hedef_sayisi > 0:
            st.markdown("---")
            st.markdown("### 📊 Hücre Analiz Raporu ve Grafikler")
            df = pd.DataFrame(rapor_verileri)
            
            col_tablo, col_grafik = st.columns([1, 2])
            
            with col_tablo:
                st.dataframe(df, use_container_width=True)
                csv_verisi = df.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Verileri Excel (CSV) İndir", data=csv_verisi, file_name='hucre_raporu.csv', mime='text/csv')
                
            with col_grafik:
                st.bar_chart(df.set_index("Hedef ID")["Alan (px)"])
                st.caption("Grafik 1: Her bir hedefin piksel bazında kapladığı alan büyüklüğü.")