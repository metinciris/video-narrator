import os
import re
import numpy as np
import torch
import soundfile as sf

from transformers import AutoTokenizer, VitsModel

# GUI dosya seçim için
import tkinter as tk
from tkinter import filedialog

# -----------------------------
# AYARLAR
# -----------------------------
MODEL_NAME = "facebook/mms-tts-tur"   # Türkçe MMS TTS modeli
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAKS_KARAKTER = 350   # Çok uzun altyazı satırı olursa kesmek için
# -----------------------------


def srt_time_to_seconds(t: str) -> float:
    """
    'HH:MM:SS,mmm' -> saniye (float)
    Örn: '00:01:23,456' -> 83.456
    """
    h, m, s_ms = t.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt_entries(path: str):
    """
    SRT dosyasını okur, her blok için {start, end, text} listesi döndürür.
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    entries = []

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2:
            continue

        # Zaman satırını bul (içinde "-->" geçen satır)
        time_line = None
        time_idx = None
        for i, ln in enumerate(lines):
            if "-->" in ln:
                time_line = ln.strip()
                time_idx = i
                break

        if time_line is None:
            continue

        # Örn: 00:00:01,000 --> 00:00:03,000
        m = re.match(r"(\d\d:\d\d:\d\d,\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d,\d\d\d)", time_line)
        if not m:
            continue

        start_str, end_str = m.group(1), m.group(2)
        start = srt_time_to_seconds(start_str)
        end = srt_time_to_seconds(end_str)
        if end <= start:
            continue

        # Zaman satırından sonraki satırlar: metin
        text_lines = []
        for ln in lines[time_idx + 1:]:
            lt = ln.strip()
            if lt:
                text_lines.append(lt)

        if not text_lines:
            continue

        text = " ".join(text_lines)
        entries.append({"start": start, "end": end, "text": text})

    # Baştan sona sırala
    entries.sort(key=lambda x: x["start"])
    return entries


# ----- Basit sayı -> Türkçe (0-99) -----
ONES = {
    0: "sıfır",
    1: "bir",
    2: "iki",
    3: "üç",
    4: "dört",
    5: "beş",
    6: "altı",
    7: "yedi",
    8: "sekiz",
    9: "dokuz",
}

TENS = {
    10: "on",
    20: "yirmi",
    30: "otuz",
    40: "kırk",
    50: "elli",
    60: "altmış",
    70: "yetmiş",
    80: "seksen",
    90: "doksan",
}


def number_to_turkish(n: int) -> str:
    if n < 10:
        return ONES.get(n, str(n))
    if n < 100:
        on = (n // 10) * 10
        birler = n % 10
        if birler == 0:
            return TENS.get(on, str(n))
        return TENS.get(on, str(on)) + " " + ONES.get(birler, str(birler))
    return str(n)  # 100+ için dokunmuyoruz


def normalize_text(text: str) -> str:
    """
    Küçük temizlik:
    - % işaretini 'yüzde' yap
    - 0-99 arası sayıları yazıyla oku
    (İngilizce kelimelere özellikle dokunmuyoruz.)
    """
    text = text.replace("%", " yüzde ")

    def repl(m):
        s = m.group(0)
        try:
            n = int(s)
            return number_to_turkish(n)
        except ValueError:
            return s

    text = re.sub(r"\b\d{1,2}\b", repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_mms_model():
    print(f"Model yükleniyor ({MODEL_NAME})... (İlk sefer uzun sürebilir)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = VitsModel.from_pretrained(MODEL_NAME).to(DEVICE)
    sample_rate = model.config.sampling_rate
    print(f"Model hazır. Örnekleme frekansı: {sample_rate} Hz")
    return tokenizer, model, sample_rate


def tts_mms(tokenizer, model, text: str, sample_rate: int) -> np.ndarray:
    """
    Tek bir altyazı satırını MMS ile seslendirir, float32 numpy (mono) döndürür.
    """
    text = normalize_text(text)

    if len(text) > MAKS_KARAKTER:
        text = text[:MAKS_KARAKTER] + "..."

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)

    audio = out.waveform.squeeze().cpu().numpy().astype(np.float32)
    return audio


def select_srt_gui():
    """
    Sadece görsel seçim: Windows dosya penceresi ile SRT seçtirir.
    """
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="SRT dosyasını seç",
        filetypes=[("SRT dosyaları", "*.srt"), ("Tüm dosyalar", "*.*")]
    )
    root.destroy()
    return path


def main():
    print("SRT -> MMS TTS Türkçe WAV (ZAMAN DAMGALI, videoya uygun)")
    print("Dosya seçme penceresi açılıyor...")

    srt_path = select_srt_gui()
    if not srt_path:
        print("Hiç dosya seçmedin, çıkıyorum.")
        return

    if not os.path.isfile(srt_path):
        print(f"SRT dosyası bulunamadı: {srt_path}")
        return

    base, _ = os.path.splitext(srt_path)
    out_wav = base + "_mms_timed.wav"

    print(f"Seçilen SRT: {srt_path}")
    entries = parse_srt_entries(srt_path)

    if not entries:
        print("SRT içinden zaman & metin çıkarılamadı.")
        return

    print(f"Toplam altyazı bloğu: {len(entries)}")

    tokenizer, model, sample_rate = load_mms_model()

    # Son altyazı bitişinden biraz sonrası kadar ses dizisi
    last_end = max(e["end"] for e in entries)
    total_duration_sec = last_end + 0.5  # yarım saniye pay
    total_samples = int(total_duration_sec * sample_rate)
    timeline = np.zeros(total_samples, dtype=np.float32)

    for i, e in enumerate(entries, start=1):
        start_s = e["start"]
        end_s = e["end"]
        text = e["text"]

        print(f"[{i}/{len(entries)}] {start_s:.3f} - {end_s:.3f}  {text[:80]}...")

        audio = tts_mms(tokenizer, model, text, sample_rate)

        start_idx = int(start_s * sample_rate)
        end_idx = int(end_s * sample_rate)
        slot_len = end_idx - start_idx

        if slot_len <= 0:
            continue

        # Eğer üretilen ses slot'tan uzunsa, slot uzunluğuna kırp
        if len(audio) > slot_len:
            print(f"  UYARI: Ses {len(audio)/sample_rate:.2f} sn ama slot {slot_len/sample_rate:.2f} sn, kırpılıyor.")
            audio = audio[:slot_len]
        # Kısa ise, kalan kısmı sessizlikle doldur
        elif len(audio) < slot_len:
            pad = np.zeros(slot_len - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, pad])

        # Slotu timeline içine yerleştir (varsa üstüne ekler = mix)
        timeline[start_idx:start_idx + len(audio)] += audio

    # Normalizasyon (taşma olmasın)
    max_abs = np.max(np.abs(timeline))
    if max_abs > 0:
        if max_abs > 0.99:
            print(f"Normalizasyon uygulanıyor (max amplitude = {max_abs:.3f})")
            timeline = timeline / max_abs * 0.95

    print(f"WAV dosyası kaydediliyor: {out_wav}")
    sf.write(out_wav, timeline, sample_rate)
    print("Bitti. Bu dosyayı videoya ikinci ses parçası olarak ekleyebilirsin.")


if __name__ == "__main__":
    main()
