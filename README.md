# SRT → Türkçe MMS TTS (facebook/mms-tts-tur)

Bu proje, bir **SRT altyazı dosyasını**, Meta'nın **facebook/mms-tts-tur** Türkçe TTS modeli ile okuyup,
altyazıdaki **zaman damgalarına sadık kalarak** tek bir `.wav` ses dosyasına dönüştürür.

Amaç:  
Örneğin İngilizce bir videonun SRT formatındaki **Türkçe çevirisini**,  
Türkçe sesli betimleme/dublaj gibi, videoya ikinci ses kanalı olarak eklemek.

---

## ✨ Özellikler

- Sadece **görsel dosya seçimi** (Windows dosya aç penceresi) – terminalde yol yazma yok.
- SRT içindeki her blok için:
  - `başlangıç zamanı → bitiş zamanı` aralığında TTS ile konuşma üretir.
  - Sesler zaman çizelgesi üzerinde yerine **tam oturtulur**.
- Sonuçta:
  - `orijinal_srt_adı_mms_timed.wav` isminde **tek bir Türkçe ses dosyası** oluşur.
  - Bu dosya, SRT ile **senkron** olacak şekilde düzenlenir.
- Sayılar (0–99) basitçe Türkçe okunur (örn: `25` → `yirmi beş`).
- `%` işareti `yüzde` olarak çevrilir.
- İngilizce kelimelere özellikle zorlamalı dönüşüm yapılmaz (model biraz aksanlı okuyabilir ama genellikle idare eder).

---

## 🧩 Gereksinimler

- **Python 3.9+** (kullanıcıda 3.13 ile çalışıyor)
- İnternet (model ilk kullanımda Hugging Face’ten indiriliyor)
- Aşağıdaki Python paketleri:

```bash
pip install -r requirements.txt

