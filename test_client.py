# Test işlemlerinde HTTP request'leri mock'lamak için FastAPI TestClient,
# JSON verileriyle çalışmak için ise standart json kütüphanesini import ediyoruz.
from fastapi.testclient import TestClient
from main import app
import json

# Uygulamamızın doğruluğunu test eden ana fonksiyonumuz.
# ornekmusteridata.json dosyasından verileri okuyup ilk 3 müşteri için asenkron api testlerini gerçekleştirir.
# Başarılı sonuçları ham JSON (test_sonuclari.json) ve rapor formatında (test_sonuclari.md) olarak kaydeder.
def run_test():
    with TestClient(app) as client:
        try:
            with open("ornekmusteridata.json", "r") as f:
                musteri_listesi = json.load(f)
        except Exception as file_err:
            print(f"Musteri veri dosyasi okunamadi: {str(file_err)}")
            return
            
        print(f"Toplam {len(musteri_listesi)} adet musteri verisi basariyla yuklendi.")
        max_test_count = 3
        print(f"İlk {max_test_count} musteri verisi uzerinde asenkron test baslatiliyor...")
        
        basarili_sonuclar = []
        
        for index, payload in enumerate(musteri_listesi[:max_test_count]):
            print(f"\n--- MUSTERI TEST {index + 1} ---")
            print(f"Gonderilen Veri: {json.dumps(payload, indent=2)}")
            response = client.post("/score", json=payload)
            print(f"Yanit Durum Kodu: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                basarili_sonuclar.append(data)
                print("Kredi Skorlama Basarili! Dönen Yanit:")
                print(json.dumps(data, indent=2))
            else:
                print(f"Hata Olustu! Yanit Detayi: {response.text}")
                
        if basarili_sonuclar:
            # Sonuçları JSON formatında kaydetme bloğumuz.
            try:
                with open("test_sonuclari.json", "w", encoding="utf-8") as out_file:
                    json.dump(basarili_sonuclar, out_file, indent=2, ensure_ascii=False)
                print("\n[OK] Tum basarili test sonuclari 'test_sonuclari.json' dosyasina kaydedildi.")
            except Exception as write_err:
                print(f"\n[ERROR] JSON sonuclar dosyaya yazilamadi: {str(write_err)}")
                
            # Sonuçları Markdown formatında kurumsal bir rapor halinde kaydetme bloğumuz.
            # ,------.,--.   ,--.,--.,------. ,--.  ,--.,------.,--.   ,--.  
            # |  .---'|   `.'   ||  ||  .--. '|  '--'  ||  .---' \  `.'  /   
            # |  `--, |  |'.'|  ||  ||  '--'.'|  .--.  ||  `--,   .'    \   
            # |  `---.|  |   |  ||  ||  |\  \ |  |  |  ||  `---. /  .'.  \   
            # `------'`--'   `--'`--'`--' '--'`--'  `--'`------''--'   '--'  
            # Linkedin : https://www.linkedin.com/in/emir-capar
            # Github : https://github.com/EmirHEX
            try:
                md_content = (
                    "# 📊 Kredi Skorlama ve Karar Destek Motoru Test Raporu\n\n"
                    "Bu rapor, Explainable AI ve yerel LLM tabanlı çok aşamalı ajantik kredi skorlama motorunun test çıktılarını içermektedir.\n\n"
                    "## 🔍 Finansal Faktörler ve Rasyo Açıklamaları\n\n"
                    "Aşağıdaki tabloda modelin kredi kararını verirken kullandığı finansal faktörlerin açıklamaları yer almaktadır:\n\n"
                    "| Faktör Adı | Açıklama | Finansal Anlamı |\n"
                    "| :--- | :--- | :--- |\n"
                    "| **income** | Aylık Net Gelir | Müşterinin borç geri ödeme kapasitesinin temel kaynağıdır. |\n"
                    "| **savings** | Toplam Birikim | Olası kriz anlarında güvence sağlayan finansal varlıklardır. |\n"
                    "| **debt** | Mevcut Toplam Borç | Müşterinin bankacılık sistemindeki mevcut yükümlülük büyüklüğüdür. |\n"
                    "| **monthly_debt_payment** | Aylık Borç Ödemesi | Aylık gelirden diğer borçlar için ayrılan nakit çıktısıdır. |\n"
                    "| **inquiry_count** | Son 6 Aydaki Sorgu Sayısı | Son dönemdeki acil nakit/kredi arayış sıklığını (riskini) gösterir. |\n"
                    "| **sector_risk_index** | Sektörel Risk Endeksi | Faaliyet gösterilen sektörün ekonomik kırılganlık derecesidir. |\n"
                    "| **delinquency_days** | Gecikme Gün Sayısı | Geçmiş ödeme ahlakını gösteren en kritik risk faktörüdür. |\n"
                    "| **liquidity_ratio** | Likidite Oranı | Birikimlerin toplam borca oranıdır; acil nakit gücünü ölçer. |\n"
                    "| **dscr** | Borç Servis Karşılama Oranı | Gelirin aylık borç ödemesine oranıdır; nakit akış kapasitesidir. |\n"
                    "| **risk_behavior_score** | Risk Davranış Skoru | Gecikme günü ve sorguların birleşik risk puanını ifade eder. |\n\n"
                    "---\n\n"
                    "## 📋 Müşteri Değerlendirme Sonuçları\n\n"
                )
                
                for index, result in enumerate(basarili_sonuclar):
                    c_data = result["client_data"]
                    agents = result["agentic_workflow"]
                    audit = agents["audit"]
                    
                    if audit["risk_class"] in ["LOW", "MEDIUM"]:
                        kredi_durumu = "🟢 ONAYLANDI"
                        musteri_mesaji = (
                            f"Sayın {c_data['customer_name']}, yaptığımız değerlendirmeler sonucunda kredi başvurunuz olumlu yanıtlanmıştır. "
                            "Kredinizi dilediğiniz zaman şubelerimizden veya mobil şubemizden hemen kullanabilirsiniz."
                        )
                    else:
                        kredi_durumu = "🔴 ONAYLANMADI"
                        alt_prod = agents["retention"]["alternative_products"]
                        önerilen_urun = alt_prod[0]["product_name"] if alt_prod else "Nakit Bloke Teminatlı Kart"
                        roadmap = agents["strategist"]
                        if isinstance(roadmap.get("roadmap_30_days"), list):
                            ilk_adim = roadmap["roadmap_30_days"][0].get("description", "borç yapılandırma")
                        elif isinstance(roadmap.get("roadmap_30_days"), dict):
                            actions = roadmap["roadmap_30_days"].get("actions", [])
                            ilk_adim = actions[0].get("description", "borç yapılandırma") if actions else "borç yapılandırma"
                        else:
                            ilk_adim = "borç yapılandırma"
                            
                        musteri_mesaji = (
                            f"Sayın {c_data['customer_name']}, kredi başvurunuza şu anlık olumlu yanıt veremiyoruz. "
                            f"Finansal sağlığınızı düzeltmek için önümüzdeki 30 gün boyunca '{ilk_adim}' adımlarını uygulayıp "
                            f"ödemelerinizi düzenli yaptıktan sonra tekrar başvurabilir veya alternatif olarak bankamızın sunduğu "
                            f"'{önerilen_urun}' ürününe hemen başvuru yapabilirsiniz."
                        )
                        
                    md_content += (
                        f"### 👤 Müşteri {index + 1}: {c_data['customer_name']}\n"
                        f"*   **Sektör**: {c_data['sector_name']} (Sektör Risk Endeksi: {c_data['sector_risk_index']})\n"
                        f"*   **Yaş**: {c_data['age']} | **Gecikme Gün Sayısı**: {c_data['delinquency_days']} gün\n"
                        f"*   **Kredi Risk Olasılığı (XGBoost)**: %{round(result['shap_explanation']['probability'] * 100, 2)}\n"
                        f"*   **LLM Risk Değerlendirme Sınıfı**: **{audit['risk_class']}**\n"
                        f"*   **Kök Neden Kategorisi**: {audit['root_cause_category']}\n"
                        f"*   **KREDİ DURUMU**: **{kredi_durumu}**\n\n"
                        f"👉 **Müşteriye Gösterilecek Mesaj**:\n"
                        f"> *\"{musteri_mesaji}\"*\n\n"
                        f"**Denetim Özeti (Audit Summary)**: {audit['audit_summary']}\n\n"
                        "**SHAP Önemli Faktörler**:\n"
                        f"- Riski Artıran İlk 3 Faktör: {', '.join([f'{f['feature']} (Katkı: {round(f['shap_value'], 3)})' for f in result['shap_explanation']['negative_factors']])}\n"
                        f"- Riski Azaltan İlk 2 Faktör: {', '.join([f'{f['feature']} (Katkı: {round(f['shap_value'], 3)})' for f in result['shap_explanation']['positive_factors']])}\n\n"
                        "---\n\n"
                    )
                    
                with open("test_sonuclari.md", "w", encoding="utf-8") as out_md_file:
                    out_md_file.write(md_content)
                print("[OK] Tum basarili test sonuclari 'test_sonuclari.md' dosyasina kaydedildi.")
            except Exception as write_md_err:
                print(f"[ERROR] Markdown sonuclar dosyaya yazilamadi: {str(write_md_err)}")

if __name__ == "__main__":
    run_test()
