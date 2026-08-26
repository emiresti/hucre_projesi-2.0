import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd # Veri tabloları ve CSV indirme için eklendi

# 1. Sayfa Tasarımı ve Düzeni (Geniş Ekran Modu Eklendi)
st.set_page_config(page_title="Hücre Analiz AI", page_icon="🔬", layout="wide")
st.title("🔬 Biyoinformatik Hücre Analiz Dashboard'u")
st.markdown("**U87 Hücre Hattı - Florasan ve Morfoloji Analiz Arayüzü**")
st.markdown("---")

# --- ÖZELLİK 1: İNTERAKTİF KALİBRASYON (SLIDER'LAR) ---
with st.expander("⚙️ Gelişmiş Kalibrasyon Ayarları (İsteğe Bağlı)"):
    st.markdown("Buradan yapay zekanın keskinliğini ve eleyeceği çöp boyutunu anlık ayarlayabilirsiniz.")
    min_alan = st.slider("Minimum Hücre Alanı (px) - Parazitleri Eler:", min_value=0, max_value=200, value=15, step=5)
    hassasiyet = st.slider("Hücre Ayrıştırma Hassasiyeti (Watershed):", min_value=0.1, max_value=0.9, value=0.4, step=0.1)
st.markdown("---")

# 2. Dosya Yükleme Alanı
yuklenen_dosya = st.file_uploader("Lütfen mikroskop görüntüsünü yükleyin (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Renk Seçim Menüsü
st.markdown("### 🧪 Analiz Edilecek Florasan Rengi")
secilen_renk = st.selectbox("Hedef boya tipini seçin:", ("🟢 Yeşil (GFP / FITC)", "🔵 Mavi (DAPI / Hoechst)", "🔴 Kırmızı (RFP / Texas Red)"))

if yuklenen_dosya is not None:
    image = Image.open(yuklenen_dosya)
    img_array = np.array(image)
    resim = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 3. Analiz Butonu
    if st.button("🚀 Yapay Zeka Analizini Başlat"):
        with st.spinner('Görüntü işleniyor, hedefler aranıyor ve morfolojik veriler hesaplanıyor...'):
            hsv_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2HSV)
            
            # Seçilen Renge Göre Maske Oluşturma
            if "Yeşil" in secilen_renk:
                alt_renk = np.array([35, 50, 50])
                ust_renk = np.array([85, 255, 255])
                maske = cv2.inRange(hsv_resim, alt_renk, ust_renk)
                cizim_rengi = (0, 255, 0)
            elif "Mavi" in secilen_renk:
                alt_renk = np.array([110, 50, 50])
                ust_renk = np.array([160, 255, 255])
                maske = cv2.inRange(hsv_resim, alt_renk, ust_renk)
                cizim_rengi = (255, 0, 0)
            elif "Kırmızı" in secilen_renk:
                maske1 = cv2.inRange(hsv_resim, np.array([0, 50, 50]), np.array([10, 255, 255]))
                maske2 = cv2.inRange(hsv_resim, np.array([160, 50, 50]), np.array([180, 255, 255]))
                maske = maske1 + maske2
                cizim_rengi = (0, 0, 255)
            
            # Watershed (Mesafe Dönüşümü) Adımları
            cekirdek = np.ones((3,3), np.uint8)
            temiz_maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, cekirdek, iterations=1)
            
            mesafe_donusumu = cv2.distanceTransform(temiz_maske, cv2.DIST_L2, 5)
            # Slider'dan gelen 'hassasiyet' değişkenini kullanıyoruz
            _, ayrilmis_hucreler = cv2.threshold(mesafe_donusumu, hassasiyet * mesafe_donusumu.max(), 255, 0)
            ayrilmis_hucreler = np.uint8(ayrilmis_hucreler)
            
            konturlar, _ = cv2.findContours(ayrilmis_hucreler, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            kopya_resim = resim.copy()
            hedef_sayisi = 0
            rapor_verileri = [] # Verileri Excel için burada toplayacağız

            for kontur in konturlar:
                alan = cv2.contourArea(kontur)
                # Slider'dan gelen 'min_alan' değişkenini kullanıyoruz
                if alan > min_alan:
                    hedef_sayisi += 1
                    cevre = cv2.arcLength(kontur, True)
                    
                    # Dairesellik Hesaplama (Matematiksel Morfoloji)
                    dairesellik = 0
                    if cevre > 0:
                        dairesellik = 4 * np.pi * (alan / (cevre * cevre))
                    
                    cv2.drawContours(kopya_resim, [kontur], -1, cizim_rengi, 2)
                    
                    # Hedefin tam merkezine numara yazdırma
                    M = cv2.moments(kontur)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = 0, 0
                    cv2.putText(kopya_resim, str(hedef_sayisi), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Bulunan veriyi listeye kaydetme
                    rapor_verileri.append({
                        "Hedef ID": hedef_sayisi,
                        "Alan (px)": round(alan, 2),
                        "Çevre (px)": round(cevre, 2),
                        "Dairesellik": round(dairesellik, 3)
                    })

        st.success(f"✅ Analiz Tamamlandı! Tespit edilen florasan hedef yapısı: {hedef_sayisi}")
        
        # --- ÖZELLİK 3: YAN YANA KARŞILAŞTIRMA GÖRÜNÜMÜ ---
        col1, col2 = st.columns(2) # Ekranı ikiye böler
        with col1:
            st.image(image, caption="Orijinal Görüntü", use_container_width=True)
        with col2:
            kopya_resim_rgb = cv2.cvtColor(kopya_resim, cv2.COLOR_BGR2RGB)
            st.image(kopya_resim_rgb, caption="Analiz Edilmiş Görüntü", use_container_width=True)

        if hedef_sayisi > 0:
            st.markdown("---")
            st.markdown("### 📊 Hücre Analiz Raporu ve Grafikler")
            
            # Verileri Pandas formatına çeviriyoruz
            df = pd.DataFrame(rapor_verileri)
            
            col_tablo, col_grafik = st.columns([1, 2]) # Tabloya 1 birim, Grafiğe 2 birim yer ayırır
            
            # Tablo ve İndirme Butonu Alanı
            with col_tablo:
                st.dataframe(df, use_container_width=True)
                
                # --- ÖZELLİK 4: EXCEL (CSV) İNDİRME BUTONU ---
                csv_verisi = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Verileri Excel (CSV) İndir",
                    data=csv_verisi,
                    file_name='hucre_analiz_raporu.csv',
                    mime='text/csv',
                )
                
            # --- ÖZELLİK 2: ANLIK GRAFİK GÖRSELLEŞTİRME ---
            with col_grafik:
                # X ekseninde Hedef ID, Y ekseninde Alan olacak şekilde Bar Grafiği çizdirir
                st.bar_chart(df.set_index("Hedef ID")["Alan (px)"])
                st.caption("Grafik 1: Her bir hedefin piksel bazında kapladığı alan büyüklüğü.")