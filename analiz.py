import cv2
import numpy as np

# 1. Görüntüyü Yükle
resim = cv2.imread("hucre.jpg")
kopya_resim = resim.copy()

# 2. Biyolojik Kalibrasyon Parametreleri (Örn: U87 Hücre Hattı İçin)
# U87 gibi glial hücrelerin veya iç organellerinin analizi için özel eşik değerleri
U87_MIN_ALAN = 15 # Organeller küçük olduğu için alanı düşürdük

# 3. İleri Seviye Görüntü İşleme: Renk Uzayını Değiştirme (BGR'den HSV'ye)
# İnsan gözünün rengi algılama şekline en yakın format HSV'dir.
hsv_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2HSV)

# 4. Florasan Yeşil Rengi Hedefleme (Dalga Boyu Filtresi)
# Sadece spesifik olarak parlayan yeşil boyayı/proteini yakalamak için sınırları belirliyoruz
alt_yesil = np.array([35, 50, 50])
ust_yesil = np.array([85, 255, 255])

# 5. Sadece yeşil olan yerleri beyaz, geri kalan her şeyi siyah yapan bir maske oluştur
maske = cv2.inRange(hsv_resim, alt_yesil, ust_yesil)

# 6. Sadece bu maskelenmiş yeşil alanların sınırlarını bul
konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

hedef_sayisi = 0

# 7. Hedefleri Analiz Et ve İşaretle
for kontur in konturlar:
    alan = cv2.contourArea(kontur)
    
    # Sadece kalibrasyon parametremizden büyük olanları say
    if alan > U87_MIN_ALAN:
        # Bulunan yapıların etrafını bu sefer Turkuaz (Cyan) rengiyle çiziyoruz
        cv2.drawContours(kopya_resim, [kontur], -1, (255, 255, 0), 2)
        hedef_sayisi += 1

# 8. Ekrana Profesyonel Rapor Baskısı
cv2.putText(kopya_resim, "U87 HUCRE HATTI - FLORASAN ANALIZI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(kopya_resim, f"Tespit Edilen Hedef Yapi: {hedef_sayisi}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

# 9. Sonucu Kaydet
cv2.imwrite("florasan_analiz.jpg", kopya_resim)
print(f"Florasan analizi tamamlandı! {hedef_sayisi} adet hedef yapı bulundu.")
print("Lütfen florasan_analiz.jpg dosyasını kontrol et.")