# 📊 Kredi Skorlama ve Karar Destek Motoru Test Raporu

Bu rapor, Explainable AI ve yerel LLM tabanlı çok aşamalı ajantik kredi skorlama motorunun test çıktılarını içermektedir.

## 🔍 Finansal Faktörler ve Rasyo Açıklamaları

Aşağıdaki tabloda modelin kredi kararını verirken kullandığı finansal faktörlerin açıklamaları yer almaktadır:

| Faktör Adı | Açıklama | Finansal Anlamı |
| :--- | :--- | :--- |
| **income** | Aylık Net Gelir | Müşterinin borç geri ödeme kapasitesinin temel kaynağıdır. |
| **savings** | Toplam Birikim | Olası kriz anlarında güvence sağlayan finansal varlıklardır. |
| **debt** | Mevcut Toplam Borç | Müşterinin bankacılık sistemindeki mevcut yükümlülük büyüklüğüdür. |
| **monthly_debt_payment** | Aylık Borç Ödemesi | Aylık gelirden diğer borçlar için ayrılan nakit çıktısıdır. |
| **inquiry_count** | Son 6 Aydaki Sorgu Sayısı | Son dönemdeki acil nakit/kredi arayış sıklığını (riskini) gösterir. |
| **sector_risk_index** | Sektörel Risk Endeksi | Faaliyet gösterilen sektörün ekonomik kırılganlık derecesidir. |
| **delinquency_days** | Gecikme Gün Sayısı | Geçmiş ödeme ahlakını gösteren en kritik risk faktörüdür. |
| **liquidity_ratio** | Likidite Oranı | Birikimlerin toplam borca oranıdır; acil nakit gücünü ölçer. |
| **dscr** | Borç Servis Karşılama Oranı | Gelirin aylık borç ödemesine oranıdır; nakit akış kapasitesidir. |
| **risk_behavior_score** | Risk Davranış Skoru | Gecikme günü ve sorguların birleşik risk puanını ifade eder. |

---

## 📋 Müşteri Değerlendirme Sonuçları

### 👤 Müşteri 1: Ali Koç
*   **Sektör**: İnşaat (Sektör Risk Endeksi: 0.85)
*   **Yaş**: 58 | **Gecikme Gün Sayısı**: 60 gün
*   **Kredi Risk Olasılığı (XGBoost)**: %98.18
*   **LLM Risk Değerlendirme Sınıfı**: **HIGH**
*   **Kök Neden Kategorisi**: Gecikmeli Ödemeler
*   **KREDİ DURUMU**: **🔴 ONAYLANMADI**

👉 **Müşteriye Gösterilecek Mesaj**:
> *"Sayın Ali Koç, kredi başvurunuza şu anlık olumlu yanıt veremiyoruz. Finansal sağlığınızı düzeltmek için önümüzdeki 30 gün boyunca 'Gecikmeli borçları temizleme' adımlarını uygulayıp ödemelerinizi düzenli yaptıktan sonra tekrar başvurabilir veya alternatif olarak bankamızın sunduğu 'Sektöre Özel Teminatlı Kart' ürününe hemen başvuru yapabilirsiniz."*

**Denetim Özeti (Audit Summary)**: Ali Koç, 58 yaşındaki İnşaat sektöründe faaliyet gösteren müşteri, gecikmeli ödemeleri ve risk davranış puanı yüksek olduğu için finansal denetimde yüksek rütbelere sahip. Müşterinin aktiflik oranı düşük olmasına rağmen gelir ve varlık seviyeleri etkilidir.

**SHAP Önemli Faktörler**:
- Riski Artıran İlk 3 Faktör: delinquency_days (Katkı: 4.629), risk_behavior_score (Katkı: 0.882), inquiry_count (Katkı: 0.629)
- Riski Azaltan İlk 2 Faktör: liquidity_ratio (Katkı: -0.249), dscr (Katkı: -0.132)

---

### 👤 Müşteri 2: Murat Demir
*   **Sektör**: Gıda (Sektör Risk Endeksi: 0.35)
*   **Yaş**: 60 | **Gecikme Gün Sayısı**: 45 gün
*   **Kredi Risk Olasılığı (XGBoost)**: %0.19
*   **LLM Risk Değerlendirme Sınıfı**: **HIGH**
*   **Kök Neden Kategorisi**: Gecikmeli Ödemeler
*   **KREDİ DURUMU**: **🔴 ONAYLANMADI**

👉 **Müşteriye Gösterilecek Mesaj**:
> *"Sayın Murat Demir, kredi başvurunuza şu anlık olumlu yanıt veremiyoruz. Finansal sağlığınızı düzeltmek için önümüzdeki 30 gün boyunca 'Geçmiş gecikmeli ödemeleri tamamlayarak borç yükünü azaltın. Bu, Murat Demir için hemen etkili olabilir ve finansal kurtuluşa yakın olma sürecini hızlandırır.' adımlarını uygulayıp ödemelerinizi düzenli yaptıktan sonra tekrar başvurabilir veya alternatif olarak bankamızın sunduğu 'Yaş Grubuna Özel Otomatik Ödemeli Mikro-Limit' ürününe hemen başvuru yapabilirsiniz."*

**Denetim Özeti (Audit Summary)**: Murat Demir, 60 yaşındaki gişe sektöründeki bir müşteri, finansal denetimde yüksek gecikmeli ödeme riski ve borç seviyesi ile karşılaştı. Delinquency_days ve loan_duration faktörleri riski artırmaktadır. Bununla birlikte,liquidity_ratio ve delinquency_days faktörlerinde pozitif etkiler olduğu gözlenmektedir.

**SHAP Önemli Faktörler**:
- Riski Artıran İlk 3 Faktör: loan_duration (Katkı: 0.029), debt (Katkı: 0.002)
- Riski Azaltan İlk 2 Faktör: delinquency_days (Katkı: -1.274), liquidity_ratio (Katkı: -0.747)

---

### 👤 Müşteri 3: Zeynep Koç
*   **Sektör**: Tarım (Sektör Risk Endeksi: 0.55)
*   **Yaş**: 37 | **Gecikme Gün Sayısı**: 0 gün
*   **Kredi Risk Olasılığı (XGBoost)**: %0.24
*   **LLM Risk Değerlendirme Sınıfı**: **LOW**
*   **Kök Neden Kategorisi**: Gecikmeli Ödemeler
*   **KREDİ DURUMU**: **🟢 ONAYLANDI**

👉 **Müşteriye Gösterilecek Mesaj**:
> *"Sayın Zeynep Koç, yaptığımız değerlendirmeler sonucunda kredi başvurunuz olumlu yanıtlanmıştır. Kredinizi dilediğiniz zaman şubelerimizden veya mobil şubemizden hemen kullanabilirsiniz."*

**Denetim Özeti (Audit Summary)**: Zeynep Koç, 37 yaşında tarım sektöründeki bir müşteri, düşük gecikmeli ödeme faktörü ve yüksek gelir-oranlı varlıklı olarak değerlendirmenin riski düşük. Ancak, aylık borc ödemelerinin yüksek olması bu durumu etkiliyor.

**SHAP Önemli Faktörler**:
- Riski Artıran İlk 3 Faktör: monthly_debt_payment (Katkı: 0.167), savings (Katkı: 0.136), loan_amount (Katkı: 0.043)
- Riski Azaltan İlk 2 Faktör: risk_behavior_score (Katkı: -1.477), delinquency_days (Katkı: -1.25)

---

