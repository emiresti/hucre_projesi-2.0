import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math
import io
import csv

# 1. Sayfa Tasarımı
st.set_page_config(page_title="Hücre Analiz AI", page_icon="🔬")
st.title("🔬 Biyoinformatik Hücre Analiz Dashboard'u")
st.markdown("**U87 Hücre Hattı - Florasan ve Morfoloji Analiz Arayüzü**")
st.markdown("---")

# 2. Dosya Yükleme Alanı
yuklenen_dosya = st.file_uploader("Lütfen mikroskop görüntüsünü yükleyin (JPG/PNG)", type=["jpg", "jpeg", "png"])

if yuklenen_dosya is not None:
    image = Image.open(yuklenen_dosya)
    img_array = np.array(image)
    resim = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Orijinal Görüntü", use_container_width=True)

    # 3. Analiz Butonu
    if st.button("🚀 Yapay Zeka Analizini Başlat"):
        with st.spinner('Görüntü işleniyor, hedefler aranıyor ve morfolojik veriler hesaplanıyor...'):
            hsv_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2HSV)
            alt_yesil = np.array([35, 50, 50])
            ust_yesil = np.array([85, 255, 255])
            maske = cv2.inRange(hsv_resim, alt_yesil, ust_yesil)
            konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            kopya_resim = resim.copy()
            hedef_sayisi = 0
            
            # --- YENİ: Excel (CSV) Verilerini Tutacağımız Liste ---
            rapor_verileri = []
            rapor_verileri.append(['Kimlik_ID', 'Alan(px)', 'Cevre(px)', 'Dairesellik', 'Durum'])

            for kontur in konturlar:
                alan = cv2.contourArea(kontur)
                if alan > 15: # U87 Kalibrasyonu
                    hedef_sayisi += 1
                    kimlik = f"HEDEF_{hedef_sayisi}"
                    
                    # Morfoloji Hesaplamaları
                    cevre = cv2.arcLength(kontur, True)
                    dairesellik = 0
                    if cevre > 0:
                        dairesellik = (4 * math.pi * alan) / (cevre * cevre)
                        
                    durum = "Pozitif" if dairesellik > 0.4 else "Atipik/Bozuk"
                    
                    # Veriyi listeye ekle
                    rapor_verileri.append([kimlik, round(alan, 1), round(cevre, 1), round(dairesellik, 3), durum])

                    # Çizim İşlemleri (Sınırları çiz ve numara ver)
                    cv2.drawContours(kopya_resim, [kontur], -1, (255, 255, 0), 3)
                    M = cv2.moments(kontur)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(kopya_resim, str(hedef_sayisi), (cX - 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            sonuc_rgb = cv2.cvtColor(kopya_resim, cv2.COLOR_BGR2RGB)
            
            st.success(f"✅ Analiz Tamamlandı! Tespit edilen florasan hedef yapısı: {hedef_sayisi}")
            st.image(sonuc_rgb, caption="Analiz Edilmiş Görüntü", use_container_width=True)

            # --- YENİ: CSV İndirme Butonu Oluşturma ---
            st.markdown("### 📊 Laboratuvar Raporu")
            
            csv_buffer = io.StringIO()
            csv_yazici = csv.writer(csv_buffer)
            csv_yazici.writerows(rapor_verileri)
            csv_metni = csv_buffer.getvalue()

            st.download_button(
                label="📥 İstatistik Raporunu İndir (.CSV)",
                data=csv_metni,
                file_name="u87_florasan_analiz_raporu.csv",
                mime="text/csv"
            )