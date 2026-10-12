# ==============================================================================
# BORSAPRO: GELİŞMİŞ PAIR TRADING ANALİZ PANELİ
# Projenin Nihai Kodu (19. Gün İtibarıyla Tamamlanmış Hali)
# ==============================================================================

# ------------------------------------------------------------------------------
# GEREKLİ KÜTÜPHANELERİN İÇE AKTARILMASI
# ------------------------------------------------------------------------------
import os
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta, date

# Analitik Kütüphaneler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

# Derin Öğrenme Kütüphaneleri (TensorFlow & Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Görselleştirme Kütüphaneleri
import plotly.express as px

# ------------------------------------------------------------------------------
# VERİ İŞLEME VE ANALİZ FONKSİYONLARI (Refactoring Sonrası)
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner="Hisse senedi verileri çekiliyor...")
def veri_cek_ve_hazirla(hisse_kodlari_tuple, baslangic_tarihi, bitis_tarihi):
    """
    Belirtilen hisse senetleri için tarihsel fiyat verilerini çeker ve hazırlar.
    Streamlit'in cache mekanizması için hisse listesi tuple olarak alınır.
    """
    hisse_kodlari = list(hisse_kodlari_tuple)
    try:
        data = yf.download(hisse_kodlari, start=baslangic_tarihi, end=bitis_tarihi, progress=False, auto_adjust=True)
        if data.empty: return None
        kapanis_fiyatlari = data['Close']
        if isinstance(kapanis_fiyatlari, pd.Series):
            kapanis_fiyatlari = kapanis_fiyatlari.to_frame(name=hisse_kodlari[0])
        kapanis_fiyatlari = kapanis_fiyatlari.ffill().bfill()
        return kapanis_fiyatlari.dropna(axis=1)
    except Exception as e:
        st.error(f"Veri çekme sırasında bir hata oluştu: {e}")
        return None

@st.cache_data(show_spinner="PCA Analizi yapılıyor...")
def pca_analizi_yap(fiyat_df):
    """
    Fiyat verileri üzerinde PCA uygular ve potansiyel çiftleri görselleştirir.
    """
    if fiyat_df is None or fiyat_df.shape[1] < 2: return None, None
    getiriler = fiyat_df.pct_change().dropna()
    if getiriler.empty: return None, None
    standart_getiriler = StandardScaler().fit_transform(getiriler)
    pca = PCA(n_components=2)
    pca.fit(standart_getiriler)
    yukler = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=getiriler.columns)
    
    fig = px.scatter(
        yukler, x='PC1', y='PC2', text=yukler.index,
        title='PCA Yükleri Dağılım Grafiği ile Potansiyel Çift Adayları'
    )
    fig.update_traces(textposition='top center', marker=dict(size=12))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.add_vline(x=0, line_dash="dash", line_color="grey")
    fig.update_layout(xaxis_title="Birinci Temel Bileşen (PC1)", yaxis_title="İkinci Temel Bileşen (PC2)")
    return yukler, fig

@st.cache_data
def sinyal_uret(hisse_A_fiyat, hisse_B_fiyat, pencere=21):
    """
    İki hisse senedi için spread, hedge oranı ve Z-Skoru hesaplar.
    """
    X = sm.add_constant(hisse_B_fiyat)
    model = sm.OLS(hisse_A_fiyat, X).fit()
    beta = model.params.iloc[1]
    spread = hisse_A_fiyat - beta * hisse_B_fiyat
    hareketli_ortalama = spread.rolling(window=pencere).mean()
    hareketli_std = spread.rolling(window=pencere).std()
    z_skoru = (spread - hareketli_ortalama) / hareketli_std
    
    return pd.DataFrame({
        'Spread': spread, 
        'Hedge_Orani': beta, 
        'Spread_Ortalamasi': hareketli_ortalama, 
        'Z_Skoru': z_skoru
    })

def create_lstm_dataset(dataset, look_back=60):
    """LSTM modeli için girdi-çıktı çiftleri oluşturur."""
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

@st.cache_data(show_spinner="🧠 LSTM Modeli eğitiliyor ve tahmin yapılıyor...")
def advanced_lstm_ile_tahmin_yap(_spread_serisi, look_back, epochs, future_steps):
    """
    Gelişmiş Bidirectional LSTM modeli eğitir ve geleceğe yönelik tahmin yapar.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = _spread_serisi.values.reshape(-1, 1)
    train_size = int(len(dataset) * 0.8)
    train, validation = dataset[:train_size], dataset[train_size:]
    train_scaled = scaler.fit_transform(train)
    validation_scaled = scaler.transform(validation)
    
    X_train, y_train = create_lstm_dataset(train_scaled, look_back)
    X_val, y_val = create_lstm_dataset(validation_scaled, look_back)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

    model = Sequential([
        Bidirectional(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1))),
        Dropout(0.2),
        Bidirectional(LSTM(units=50)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, 
                        validation_data=(X_val, y_val), 
                        callbacks=[early_stopping, reduce_lr], verbose=0)

    predictions = []
    current_input_scaled = scaler.transform(dataset[-look_back:])
    for _ in range(future_steps):
        input_reshape = np.reshape(current_input_scaled, (1, look_back, 1))
        predicted_value_scaled = model.predict(input_reshape, verbose=0)[0][0]
        predictions.append(predicted_value_scaled)
        current_input_scaled = np.append(current_input_scaled[1:], predicted_value_scaled).reshape(-1, 1)
        
    predicted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predicted_values, _spread_serisi.index[-1], history

def hibrit_sinyal_uret(sinyal_df, tahmin_df):
    """Z-Skoru ve LSTM tahminini birleştirerek hibrit sinyal üretir."""
    son_z_skoru = sinyal_df['Z_Skoru'].iloc[-1]
    son_spread = sinyal_df['Spread'].iloc[-1]
    if pd.isna(son_z_skoru): return "Veri Yetersiz"
    
    tahmin_yonu = "Yükseliş" if tahmin_df['Tahmin'].iloc[-1] > son_spread else "Düşüş"
    
    if son_z_skoru < -2.0 and tahmin_yonu == "Yükseliş":
        return "Al Sinyali (LSTM Onaylı 👍)"
    elif son_z_skoru > 2.0 and tahmin_yonu == "Düşüş":
        return "Sat Sinyali (LSTM Onaylı 👍)"
    elif son_z_skoru < -2.0:
        return "Sinyal Filtrelendi (LSTM Düşüş Bekliyor 👎)"
    elif son_z_skoru > 2.0:
        return "Sinyal Filtrelendi (LSTM Yükseliş Bekliyor 👎)"
    else:
        return "Bekle (Anormal Durum Yok)"

def hedef_bolge_hesapla(sinyal_df, hisse_a_fiyat, hisse_b_fiyat):
    """Onaylı bir sinyal durumunda teorik kar alma hedeflerini hesaplar."""
    hedef_spread = sinyal_df['Spread_Ortalamasi'].iloc[-1]
    if pd.isna(hedef_spread): return None
    
    beta = sinyal_df['Hedge_Orani'].iloc[-1]
    son_fiyat_a = hisse_a_fiyat.iloc[-1]
    son_fiyat_b = hisse_b_fiyat.iloc[-1]
    
    hedef_fiyat_a = hedef_spread + (beta * son_fiyat_b)
    hedef_fiyat_b = (son_fiyat_a - hedef_spread) / beta
    return {'son_a': son_fiyat_a, 'hedef_a': hedef_fiyat_a, 'son_b': son_fiyat_b, 'hedef_b': hedef_fiyat_b}


# ------------------------------------------------------------------------------
# STREAMLIT KULLANICI ARAYÜZÜ (UI)
# ------------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="BORSAPRO", page_icon="📈")

with st.sidebar:
    logo_path = os.path.join(os.path.dirname(__file__), "ChatGPT Image 5 Eki 2025 16_21_12.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=100)
    elif os.path.exists("/Users/azrabahsi/Downloads/ChatGPT Image 5 Eki 2025 16_21_12.png"):
        st.image("/Users/azrabahsi/Downloads/ChatGPT Image 5 Eki 2025 16_21_12.png", width=100)
    st.title("BORSAPRO")
    st.caption("Gelişmiş Pair Trading Paneli")

    with st.expander("📈 Parametreler", expanded=True):
        hisse_kodlari_str = st.text_area("Hisse Kodları", "EREGL.IS, KRDMD.IS, THYAO.IS, PGSUS.IS, GARAN.IS, AKBNK.IS")
        bas_tarih = st.date_input("Başlangıç Tarihi", date(2022, 1, 1))
        bit_tarih = st.date_input("Bitiş Tarihi", date.today())
    
    with st.expander("🧠 LSTM Ayarları"):
        tahmin_gunu = st.slider("Tahmin Günü Sayısı", 5, 30, 10)
        look_back = st.slider("Geriye Bakış Penceresi", 30, 120, 60)
        epochs = st.slider("Maksimum Eğitim Sayısı (Epochs)", 50, 200, 100)
    
    if st.button("🚀 Analizi Başlat", type="primary", use_container_width=True):
        hisse_listesi = [h.strip().upper() for h in hisse_kodlari_str.split(',') if h.strip()]
        if len(hisse_listesi) < 2:
            st.warning("Lütfen en az iki hisse senedi kodu giriniz.")
            st.stop()
        else:
            st.session_state.clear() 
            st.session_state.run_analysis = True
            st.session_state.hisse_listesi_tuple = tuple(hisse_listesi)
            st.session_state.bas_tarih, st.session_state.bit_tarih = bas_tarih, bit_tarih
            st.session_state.tahmin_gunu, st.session_state.look_back, st.session_state.epochs = tahmin_gunu, look_back, epochs
            st.rerun()

st.title("📈 Pair Trading Strateji Geliştirme Paneli")

if not st.session_state.get('run_analysis'):
    st.info("Analize başlamak için lütfen sol menüden parametreleri girip 'Analizi Başlat' butonuna tıklayınız.")
else:
    try:
        if 'fiyat_df' not in st.session_state:
            st.session_state.fiyat_df = veri_cek_ve_hazirla(
                st.session_state.hisse_listesi_tuple, 
                st.session_state.bas_tarih, 
                st.session_state.bit_tarih
            )
        fiyat_df = st.session_state.fiyat_df

        if fiyat_df is None or fiyat_df.empty:
            st.error("Seçilen hisseler veya tarih aralığı için veri bulunamadı.")
            st.stop()

        if 'hisse_a' not in st.session_state: 
            st.session_state.hisse_a = fiyat_df.columns[0]
        if 'hisse_b' not in st.session_state:
            st.session_state.hisse_b = fiyat_df.columns[1]

        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Aday Belirleme", "📊 Derinlemesine Analiz", "🎯 Hibrit Sinyal", "ℹ️ Notlar"])

        with tab1:
            yukler, fig_pca = pca_analizi_yap(fiyat_df)
            st.markdown("Bu grafik, hisselerin fiyat hareketlerindeki benzerlikleri gösterir. Birbirine yakın konumdaki hisseler, genellikle iyi Pair Trading adaylarıdır.")
            if fig_pca is not None:
                st.plotly_chart(fig_pca, use_container_width=True)

        with tab2:
            st.header("Seçilen Çiftin İstatistiksel Analizi")
            col_a, col_b = st.columns(2)
            
            secilen_hisse_a = col_a.selectbox("Hisse A", fiyat_df.columns, index=list(fiyat_df.columns).index(st.session_state.hisse_a))
            secilen_hisse_b = col_b.selectbox("Hisse B", fiyat_df.columns, index=list(fiyat_df.columns).index(st.session_state.hisse_b))

            if (secilen_hisse_a != st.session_state.hisse_a) or (secilen_hisse_b != st.session_state.hisse_b):
                st.session_state.hisse_a, st.session_state.hisse_b = secilen_hisse_a, secilen_hisse_b
                st.session_state.recalculate = True
                st.rerun()

            if st.session_state.hisse_a == st.session_state.hisse_b:
                st.warning("Lütfen birbirinden farklı iki hisse senedi seçin.")
            else:
                hisse_a_s, hisse_b_s = fiyat_df[st.session_state.hisse_a], fiyat_df[st.session_state.hisse_b]
                
                coint_score, p_value, _ = coint(hisse_a_s, hisse_b_s)
                st.metric("Eşbütünleşme Testi (p-değeri)", f"{p_value:.3f}", "İlişki Anlamlı ✅" if p_value < 0.05 else "İlişki Anlamsız ❌")
                
                sinyal_df_tab2 = sinyal_uret(hisse_a_s, hisse_b_s)
                fig_z = px.line(sinyal_df_tab2, y='Z_Skoru', title=f"Z-Skoru Grafiği")
                fig_z.add_hline(y=2, line_dash="dash", line_color="red"); fig_z.add_hline(y=-2, line_dash="dash", line_color="green")
                st.plotly_chart(fig_z, use_container_width=True)

        with tab3:
            st.header("Hibrit Sinyal Karar Hunisi")
            
            if 'hibrit_sinyal' not in st.session_state or st.session_state.get('recalculate', False):
                with st.spinner("Hibrit Sinyal hesaplanıyor... Bu işlem birkaç dakika sürebilir."):
                    hisse_a_s = fiyat_df[st.session_state.hisse_a]
                    hisse_b_s = fiyat_df[st.session_state.hisse_b]
                    st.session_state.sinyal_df = sinyal_uret(hisse_a_s, hisse_b_s)
                    spread_serisi = st.session_state.sinyal_df['Spread'].dropna()
                    
                    if len(spread_serisi) > st.session_state.look_back + 50:
                        predicted_values, son_tarih, history = advanced_lstm_ile_tahmin_yap(spread_serisi, st.session_state.look_back, st.session_state.epochs, st.session_state.tahmin_gunu)
                        st.session_state.model_history = history
                        tahmin_tarihleri = pd.date_range(start=son_tarih, periods=st.session_state.tahmin_gunu + 1, freq='B')[1:]
                        st.session_state.tahmin_df = pd.DataFrame(index=tahmin_tarihleri, data={'Tahmin': predicted_values.flatten()})
                        st.session_state.hibrit_sinyal = hibrit_sinyal_uret(st.session_state.sinyal_df, st.session_state.tahmin_df)
                    else:
                        st.session_state.hibrit_sinyal = "Veri Yetersiz"
                st.session_state.recalculate = False
            
            if st.session_state.hibrit_sinyal != "Veri Yetersiz":
                col1, col2, col3 = st.columns(3)
                sinyal_df, tahmin_df, hibrit_sinyal = st.session_state.sinyal_df, st.session_state.tahmin_df, st.session_state.hibrit_sinyal

                with col1, st.container(border=True):
                    st.markdown("##### 1. Z-Skor Kontrolü"); st.metric("Mevcut Z-Skoru", f"{sinyal_df['Z_Skoru'].iloc[-1]:.2f}")
                with col2, st.container(border=True):
                    st.markdown("##### 2. LSTM Onay Kontrolü"); st.metric("LSTM Tahmin Yönü", "Yükseliş 📈" if tahmin_df['Tahmin'].iloc[-1] > sinyal_df['Spread'].iloc[-1] else "Düşüş 📉")
                with col3, st.container(border=True):
                    st.markdown("##### 3. Nihai Karar"); st.metric("Hibrit Sinyal", hibrit_sinyal.split('(')[0])

                with st.expander("📈 Detaylı Grafikler ve Hedef Fiyatlar"):
                    fig_tahmin = px.line(sinyal_df['Spread'].tail(120), title="Geçmiş Spread ve Gelecek Tahmini")
                    fig_tahmin.add_scatter(x=tahmin_df.index, y=tahmin_df['Tahmin'], mode='lines+markers', name='LSTM Tahmini')
                    st.plotly_chart(fig_tahmin, use_container_width=True)
                    
                    history_df = pd.DataFrame(st.session_state.model_history.history)
                    fig_history = px.line(history_df, y=['loss', 'val_loss'], title="Eğitim ve Doğrulama Kayıp Grafiği")
                    st.plotly_chart(fig_history, use_container_width=True)

                    if "Onaylı" in hibrit_sinyal:
                        hedef_verisi = hedef_bolge_hesapla(sinyal_df, fiyat_df[st.session_state.hisse_a], fiyat_df[st.session_state.hisse_b])
                        if hedef_verisi:
                            st.subheader("Pozisyon Kapatma Hedefleri (Teorik)")
                            col_h1, col_h2 = st.columns(2)
                            col_h1.metric(f"{st.session_state.hisse_a} Hedefi", f"{hedef_verisi['hedef_a']:.2f} ₺")
                            col_h2.metric(f"{st.session_state.hisse_b} Hedefi", f"{hedef_verisi['hedef_b']:.2f} ₺")
            else:
                st.warning(f"LSTM modelini eğitmek için yeterli veri yok. Gerekli: {st.session_state.look_back + 50}, Mevcut: {len(fiyat_df)}")

        with tab4:
            st.header("Strateji Notları ve Kullanım Kılavuzu")
            st.markdown("""
            ### Pair Trading Stratejisi Nedir?
            Pair Trading, piyasanın genel yönünden bağımsız olarak, geçmişte birlikte hareket ettiği gözlemlenen
            iki finansal varlık arasındaki fiyat farkından (spread) kar elde etmeyi amaçlayan bir piyasa-nötr stratejisidir. Temel varsayım, bu spread'in zamanla tarihsel ortalamasına geri döneceğidir (mean reversion).
            
            #### Hibrit Sinyal Ne İşe Yarar?
            Bu panel, klasik Z-Skoru sinyallerini, spread'in gelecek yönünü tahmin eden bir LSTM yapay zeka modeli ile teyit eder. Eğer Z-Skoru ve LSTM tahmini aynı yönü işaret ediyorsa, bu **"Onaylı Sinyal"** olarak kabul edilir ve daha yüksek bir güvenilirlik sunar.
            
            **Yasal Uyarı:** Bu araç yalnızca eğitim ve araştırma amaçlıdır. Burada sunulan hiçbir bilgi yatırım tavsiyesi niteliği taşımaz.
            """)
            
    except Exception as e:
        st.error(f"Uygulama çalışırken bir hata oluştu: {e}")
        st.info("Lütfen sol menüden parametreleri kontrol edip analizi yeniden başlatın.")