# 📈 BORSAPRO: Gelişmiş Pair Trading Analiz Paneli

BORSAPRO, Borsa İstanbul (BIST) hisse senetleri için istatistiksel arbitraj (Pair Trading) stratejileri geliştiren ve bu stratejileri derin öğrenme (Bidirectional LSTM) ile teyit eden interaktif bir Streamlit analiz panelidir.

## 🚀 Temel Özellikler

- **Aday Belirleme (PCA):** Fiyat hareketlerinin benzerliğini analiz ederek potansiyel çift adaylarını görselleştirme.
- **İstatistiksel Analiz (Eşbütünleşme & Z-Skoru):** Engle-Granger eşbütünleşme (cointegration) testi, dinamik hedge oranı ve hareketli Z-skoru takibi.
- **Derin Öğrenme (Bidirectional LSTM):** Spread serisi üzerinde zaman serisi tahmini yaparak gelecek fiyat farkı yönünü kestirme.
- **🎯 Hibrit Karar Hunisi:** Klasik Z-skoru sinyalleri ile LSTM derin öğrenme tahminini birleştirerek onaylı al/sat sinyalleri ve teorik kar alma hedefleri üretme.

## 🛠️ Kurulum ve Çalıştırma

```bash
# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# Uygulamayı başlatın
streamlit run main.py
```

## 📦 Kullanılan Teknolojiler
- **Streamlit** - Web arayüzü
- **TensorFlow & Keras** - Bidirectional LSTM modeli
- **Statsmodels** - Eşbütünleşme testi ve OLS regresyonu
- **Scikit-learn** - PCA analizi ve veri ölçekleme
- **yfinance** - Tarihsel piyasa verileri
- **Plotly** - İnteraktif grafikler
