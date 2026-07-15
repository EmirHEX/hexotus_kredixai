# Matematiksel ve matris işlemleri için numpy, veri analizi ve dataframe işlemleri için pandas kütüphanelerini kullanıyoruz.
# Model kurma aşamasında XGBoost, model kararlarını açıklamak ve analiz etmek için shap kütüphanesini tercih ettik.
# Web servis yapısı için FastAPI, veri doğrulama ve şema kontrolü için ise Pydantic v2 kütüphanesini kullanıyoruz.
# Ollama yerel LLM entegrasyonunda asenkron HTTP istekleri atmak amacıyla httpx kütüphanesini projeye dahil ediyoruz.
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import httpx
import json
from contextlib import asynccontextmanager

# ,------.,--.   ,--.,--.,------. ,--.  ,--.,------.,--.   ,--.  
# |  .---'|   `.'   ||  ||  .--. '|  '--'  ||  .---' \  `.'  /   
# |  `--, |  |'.'|  ||  ||  '--'.'|  .--.  ||  `--,   .'    \   
# |  `---.|  |   |  ||  ||  |\  \ |  |  |  ||  `---. /  .'.  \   
# `------'`--'   `--'`--'`--' '--'`--'  `--'`------''--'   '--'  
# Linkedin : https://www.linkedin.com/in/emir-capar
# Github : https://github.com/EmirHEX


# Kredi başvurusunda kullanıcılardan alacağımız verileri doğrulamak için Pydantic modelimizi tanımlıyoruz.
# Müşteri adı, yaşı, sektörü ve gecikme gün sayısı gibi kritik değişkenlerin yanı sıra
# gelir, birikim, mevcut borç miktarı gibi finansal bilgileri de burada zorunlu tutuyoruz.
class CreditApplicationRequest(BaseModel):
    customer_name: str = Field(..., description="Musterinin ad ve soyadi")
    sector_name: str = Field(..., description="Musterinin faaliyet gosterdigi sektor adi")
    age: int = Field(..., ge=18, le=120, description="Musterinin yasi")
    delinquency_days: int = Field(..., ge=0, description="Son 12 aydaki toplam gecikme gun sayisi")
    income: float = Field(..., gt=0, description="Musterinin aylik net geliri")
    savings: float = Field(..., ge=0, description="Musterinin bankadaki toplam birikimi")
    debt: float = Field(..., ge=0, description="Musterinin mevcut toplam borc miktari")
    monthly_debt_payment: float = Field(..., ge=0, description="Musterinin aylik borc odeme taksiti")
    inquiry_count: int = Field(..., ge=0, description="Son 6 aydaki kredi basvuru sorgu sayisi")
    sector_risk_index: float = Field(..., ge=0, le=1, description="Sektorel kirilganlik endeksi")
    loan_amount: float = Field(..., gt=0, description="Talep edilen kredi miktari")
    loan_duration: int = Field(..., gt=0, description="Talep edilen kredinin vadesi (ay)")


# Kullanıcının temel finansal verilerinden bankacılık rasyolarını üreten fonksiyonumuz.
# Buradaki amacımız varlık likidite oranını (liquidity_ratio) ve borç ödeme gücünü gösteren DSCR oranını hesaplamak.
# Ayrıca gecikme gün sayısı ile başvuru sorgularını birleştiren bir risk davranış skoru da üretiyoruz.
def calculate_advanced_features(req: CreditApplicationRequest) -> Dict[str, float]:
    liquidity_ratio = req.savings / (req.debt + 1.0)
    dscr = req.income / (req.monthly_debt_payment + 1.0)
    risk_behavior_score = (req.delinquency_days * 0.7) + (req.inquiry_count * 3.0)
    return {
        "liquidity_ratio": liquidity_ratio,
        "dscr": dscr,
        "risk_behavior_score": risk_behavior_score
    }


# Kredi skorlamasını yapan ve kararların SHAP ile açıklanmasını sağlayan sınıfımız.
# Uygulama ayağa kalkarken sentetik veriyle bir XGBoost modeli eğitiyoruz.
# Tahmin aşamasında ise bu modeli ve SHAP TreeExplainer'ı kullanarak riski etkileyen ilk 3 negatif ve ilk 2 pozitif faktörü buluyoruz.
class CreditScoringEngine:
    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names = [
            "age", "delinquency_days", "income", "savings", "debt", 
            "monthly_debt_payment", "inquiry_count", "sector_risk_index", 
            "loan_amount", "loan_duration", "liquidity_ratio", "dscr", 
            "risk_behavior_score"
        ]

    # Modelimizi eğitmek için 1000 adet gerçekçi sentetik veri üretiyoruz.
    # Bu veriler üzerinde dscr, delinquency_days gibi bankacılık kurallarına göre etiketleme yapıyoruz.
    # Son olarak XGBClassifier nesnemizi fit edip SHAP TreeExplainer açıklayıcımızı bağlıyoruz.
    def train_synthetic_model(self):
        np.random.seed(42)
        n_samples = 1000
        age = np.random.randint(18, 71, n_samples)
        delinquency_days = np.random.choice([0, 5, 15, 30, 60, 90], size=n_samples, p=[0.6, 0.15, 0.1, 0.05, 0.05, 0.05])
        income = np.random.uniform(10000, 150000, n_samples)
        savings = np.random.uniform(0, 200000, n_samples)
        debt = np.random.uniform(0, 100000, n_samples)
        monthly_debt_payment = np.random.uniform(0, 15000, n_samples)
        inquiry_count = np.random.randint(0, 11, n_samples)
        sector_risk_index = np.random.uniform(0.0, 1.0, n_samples)
        loan_amount = np.random.uniform(10000, 500000, n_samples)
        loan_duration = np.random.randint(12, 61, n_samples)
        liquidity_ratio = savings / (debt + 1.0)
        dscr = income / (monthly_debt_payment + 1.0)
        risk_behavior_score = (delinquency_days * 0.7) + (inquiry_count * 3.0)
        
        df = pd.DataFrame({
            "age": age,
            "delinquency_days": delinquency_days,
            "income": income,
            "savings": savings,
            "debt": debt,
            "monthly_debt_payment": monthly_debt_payment,
            "inquiry_count": inquiry_count,
            "sector_risk_index": sector_risk_index,
            "loan_amount": loan_amount,
            "loan_duration": loan_duration,
            "liquidity_ratio": liquidity_ratio,
            "dscr": dscr,
            "risk_behavior_score": risk_behavior_score
        })
        
        labels = np.zeros(n_samples)
        for i in range(n_samples):
            risk_score = 0.0
            if delinquency_days[i] > 30:
                risk_score += 0.45
            if dscr[i] < 1.2:
                risk_score += 0.30
            if liquidity_ratio[i] < 0.2:
                risk_score += 0.25
            if inquiry_count[i] > 4:
                risk_score += 0.20
            if sector_risk_index[i] > 0.6:
                risk_score += 0.15
            if age[i] < 22:
                risk_score += 0.10
            if risk_score > 0.45:
                labels[i] = 1
            else:
                labels[i] = 0
                
        self.model = xgb.XGBClassifier(
            max_depth=4,
            learning_rate=0.1,
            n_estimators=50,
            objective="binary:logistic",
            random_state=42
        )
        self.model.fit(df, labels)
        self.explainer = shap.TreeExplainer(self.model)

    # Gelen başvurunun risk olasılığını hesaplıyoruz ve SHAP değerlerini çıkartıyoruz.
    # SHAP değerlerini sırayla ayıklayarak riski en çok tetikleyen 3 negatif ve en çok azaltan 2 pozitif faktörü dönüyoruz.
    def analyze_credit(self, req: CreditApplicationRequest, advanced_features: Dict[str, float]) -> Dict[str, Any]:
        input_data = {
            "age": req.age,
            "delinquency_days": req.delinquency_days,
            "income": req.income,
            "savings": req.savings,
            "debt": req.debt,
            "monthly_debt_payment": req.monthly_debt_payment,
            "inquiry_count": req.inquiry_count,
            "sector_risk_index": req.sector_risk_index,
            "loan_amount": req.loan_amount,
            "loan_duration": req.loan_duration,
            "liquidity_ratio": advanced_features["liquidity_ratio"],
            "dscr": advanced_features["dscr"],
            "risk_behavior_score": advanced_features["risk_behavior_score"]
        }
        
        df_input = pd.DataFrame([input_data])
        proba = float(self.model.predict_proba(df_input)[0][1])
        
        shap_values = self.explainer.shap_values(df_input)
        
        if isinstance(shap_values, list):
            shap_array = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            shap_array = shap_values[0, :, 1]
        elif len(shap_values.shape) == 2 and shap_values.shape[0] == 1:
            shap_array = shap_values[0]
        else:
            shap_array = shap_values[0]
            
        shap_dict = []
        for name, val in zip(self.feature_names, shap_array):
            shap_dict.append({"feature": name, "shap_value": float(val)})
            
        negative_factors = sorted([x for x in shap_dict if x["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)[:3]
        positive_factors = sorted([x for x in shap_dict if x["shap_value"] < 0], key=lambda x: x["shap_value"])[:2]
        
        return {
            "probability": proba,
            "negative_factors": negative_factors,
            "positive_factors": positive_factors
        }


# Yerel Ollama üzerinde çalışan Qwen 2.5 modeli ile asenkron iletişimi yöneten ve
# 3 farklı ajanın (Audit, Strategist, Retention) prompt zincirini çalıştıran sınıfımız.
class OllamaAgentWorkflow:
    def __init__(self, ollama_url: str = "http://localhost:11434/api/chat"):
        self.ollama_url = ollama_url
        self.model_name = "qwen2.5:7b"

    # Ollama API'sine HTTP POST isteği gönderen ortak asenkron fonksiyonumuz.
    # LLM tarafına mutlaka JSON formatında yanıt dönmesini zorluyoruz.
    async def _call_ollama(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "format": "json",
                "stream": False
            }
            try:
                response = await client.post(self.ollama_url, json=payload)
                response.raise_for_status()
                response_json = response.json()
                content_str = response_json["message"]["content"]
                return json.loads(content_str)
            except httpx.HTTPError as http_err:
                raise HTTPException(status_code=500, detail=f"Ollama API baglanti hatasi: {str(http_err)}")
            except json.JSONDecodeError as json_err:
                raise HTTPException(status_code=500, detail=f"Ollama yaniti JSON formatina donusturulemedi: {str(json_err)}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Beklenmedik Ollama hatasi: {str(e)}")

    # 1. Aşama: Audit Agent (Kredi Denetim Ajanı) bulguları analiz edip risk sınıfını belirler.
    async def run_audit_agent(self, client_data: Dict[str, Any], shap_analysis: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = (
            "Sen uzman bir FinTek Kredi Denetçisisin (Audit Agent).\n"
            "Görevin, verilen müşteri finansal verilerini ve SHAP açıklanabilirlik çıktılarını analiz etmektir.\n"
            "Müşteriye mutlaka ismiyle hitap etmeli, faaliyet gösterdiği sektörü ve yaşını analizinde göz önünde bulundurmalısın.\n"
            "Çıktı olarak mutlaka geçerli ve parse edilebilir bir JSON objesi dönmelisin.\n"
            "JSON objesi şu alanları içermelidir:\n"
            "- risk_class: 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL' değerlerinden biri\n"
            "- root_cause_category: Riski tetikleyen ana faktörün kategorisi (Örn: 'Gecikmeli Ödemeler', 'Yüksek Borçluluk', 'Likidite Yetersizliği' vb.)\n"
            "- audit_summary: Müşteri adını ve yaşını içeren, kısa ve öz finansal denetim özeti\n"
            "Asla JSON haricinde açıklama veya metin yazma."
        )
        user_prompt = f"""
        Musteri Verileri: {json.dumps(client_data, indent=2)}
        XGBoost/SHAP Analiz Cikti Detaylari: {json.dumps(shap_analysis, indent=2)}
        Lütfen bu verileri analiz et ve istenen JSON formatında sonuç üret.
        """
        return await self._call_ollama(system_prompt, user_prompt)

    # 2. Aşama: Financial Health Strategist Agent (Finansal Sağlık Stratejisti) 30-60-90 günlük yol haritası hazırlar.
    async def run_strategist_agent(self, client_data: Dict[str, Any], audit_result: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = (
            "Sen kıdemli bir Finansal Sağlık Stratejistisin (Financial Health Strategist Agent).\n"
            "Denetim Ajanının risk raporunu, müşteri verilerini ve çalıştığı sektörü kullanarak, müşterinin finansal durumunu iyileştirmesi için aksiyon planı hazırlamalısın.\n"
            "Müşteriye ismiyle hitap etmeyi ve sektörünün dinamiklerine uygun tavsiyeler vermeyi unutma.\n"
            "Çıktı olarak mutlaka geçerli ve parse edilebilir bir JSON objesi dönmelisin.\n"
            "JSON objesi şu alanları içermelidir:\n"
            "- roadmap_30_days: 30 günlük süreçte atılması gereken adımlar (Örn: Gecikmiş borçları temizleme, harcama kısıntısı)\n"
            "- roadmap_60_days: 60 günlük süreçte atılması gereken adımlar (Örn: Sektörel dalgalanmalara karşı acil durum fonu oluşturma)\n"
            "- roadmap_90_days: 90 günlük süreçte atılması gereken adımlar (Örn: Yeni tasarruf alışkanlıkları, skor takibi)\n"
            "Asla JSON haricinde açıklama veya metin yazma."
        )
        user_prompt = f"""
        Musteri Verileri: {json.dumps(client_data, indent=2)}
        Denetim Ajani Bulgulari: {json.dumps(audit_result, indent=2)}
        Lütfen bu girdilere göre 30, 60 ve 90 günlük gelişim reçetesini içeren JSON çıktısını üret.
        """
        return await self._call_ollama(system_prompt, user_prompt)

    # 3. Aşama: Customer Retention Agent (Müşteri Elde Tutma Ajanı) risksiz alternatif ürün önerir.
    async def run_retention_agent(self, client_data: Dict[str, Any], audit_result: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = (
            "Sen müşteri sadakati ve elde tutma uzmanısın (Customer Retention Agent).\n"
            "Kredi başvurusu olumsuz veya riskli değerlendirilen müşterileri rakip bankalara kaptırmamak adına risksiz alternatif teklifler türetmelisin.\n"
            "Müşteriye ismiyle hitap etmeli, onun çalıştığı sektöre veya yaşına özel avantajlı teklifler sunmalısın.\n"
            "Çıktı olarak mutlaka geçerli ve parse edilebilir bir JSON objesi dönmelisin.\n"
            "JSON objesi şu alanları içermelidir:\n"
            "- alternative_products: En az iki adet alternatif risksiz ürün teklifi içeren liste. Her ürün şunları içermelidir:\n"
            "  * product_name: Ürünün adı (Örn: Sektöre Özel Teminatlı Kart, Yaş Grubuna Özel Otomatik Ödemeli Mikro-Limit vb.)\n"
            "  * reason_and_benefits: Bu ürünün müşteriye ve bankaya sağladığı fayda, neden önerildiği\n"
            "Asla JSON haricinde açıklama veya metin yazma."
        )
        user_prompt = f"""
        Musteri Verileri: {json.dumps(client_data, indent=2)}
        Denetim Ajani Bulgulari: {json.dumps(audit_result, indent=2)}
        Lütfen bu verilere göre rakip bankalara geçişi engelleyecek alternatif risksiz ürün önerilerini içeren JSON çıktısını üret.
        """
        return await self._call_ollama(system_prompt, user_prompt)


# Uygulama nesnelerimizi global seviyede tanımlayıp hazırlıyoruz.
scoring_engine = CreditScoringEngine()
agent_workflow = OllamaAgentWorkflow()

# FastAPI lifespan yöneticisi sayesinde uygulamamız başlarken XGBoost modelimizi sentetik veriyle eğitiyoruz.
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("XGBoost modeli sentetik veri kümesiyle eğitiliyor...")
    scoring_engine.train_synthetic_model()
    print("Model eğitimi ve SHAP TreeExplainer kurulumu başarıyla tamamlandı.")
    yield
    print("Uygulama kapatılıyor...")

app = FastAPI(title="Explainable AI & Agentic Credit Scoring Engine", lifespan=lifespan)

# Kredi skorlama endpoint'imiz. Gelen verileri işler, XGBoost/SHAP analizini yapar,
# Ollama asenkron ajantik akışını çalıştırır ve müşteriye yönelik dinamik bildirimlerle zenginleştirilmiş yanıt döner.
@app.post("/score")
async def score_credit(request: CreditApplicationRequest):
    try:
        client_data = request.model_dump()
        advanced_features = calculate_advanced_features(request)
        client_data.update(advanced_features)
        
        shap_analysis = scoring_engine.analyze_credit(request, advanced_features)
        
        audit_result = await agent_workflow.run_audit_agent(client_data, shap_analysis)
        strategist_result = await agent_workflow.run_strategist_agent(client_data, audit_result)
        retention_result = await agent_workflow.run_retention_agent(client_data, audit_result)
        
        if audit_result.get("risk_class") in ["LOW", "MEDIUM"]:
            kredi_durumu = "ONAYLANDI"
            musteri_mesaji = (
                f"Sayin {request.customer_name}, yaptigimiz degerlendirmeler sonucunda kredi basvurunuz olumlu yanitlanmistir. "
                "Kredinizi dilediginiz zaman mobil subemizden hemen kullanabilirsiniz."
            )
        else:
            kredi_durumu = "ONAYLANMADI"
            alt_products = retention_result.get("alternative_products", [])
            onerilen_urun = alt_products[0].get("product_name") if alt_products else "Nakit Bloke Teminatli Kart"
            roadmap_30 = strategist_result.get("roadmap_30_days", [])
            if isinstance(roadmap_30, list) and roadmap_30:
                ilk_adim = roadmap_30[0].get("description", "borc yapilandirma")
            elif isinstance(roadmap_30, dict):
                actions = roadmap_30.get("actions", [])
                ilk_adim = actions[0].get("description", "borc yapilandirma") if actions else "borc yapilandirma"
            else:
                ilk_adim = "borc yapilandirma"
                
            musteri_mesaji = (
                f"Sayin {request.customer_name}, kredi basvurunuza su anlik olumlu yanit veremiyoruz. "
                f"Finansal sagliginizi duzeltmek ve skorumuzu yukseltmek icin fatura odemelerinizi duzenli yapip, "
                f"30 gun boyunca '{ilk_adim}' adimini uygulayarak ilerleyen zamanda tekrar basvurabilirsiniz. "
                f"Alternatif olarak dilerseniz hemen '{onerilen_urun}' urunumuze basvuru yapabilirsiniz."
            )
            
        faktor_aciklamalari = {
            "income": "Aylik net gelir. Musterinin borc geri odeme kapasitesinin temel kaynagidir.",
            "savings": "Toplam birikim. Olasi kriz anlarinda guvence saglayan finansal varliklardir.",
            "debt": "Mevcut toplam borc. Musterinin bankacilik sistemindeki toplam yukumluluk buyuklugudur.",
            "monthly_debt_payment": "Aylik borc odemesi. Aylik gelirden diger borclar icin ayrilan nakit ciktisidir.",
            "inquiry_count": "Son 6 aydaki sorgu sayisi. Son donemdeki acil nakit/kredi arayis sikligini (riskini) gosterir.",
            "sector_risk_index": "Sektorel risk endeksi. Faaliyet gosterilen sektorun ekonomik kirilganlik derecesidir.",
            "delinquency_days": "Gecikme gun sayisi. Gecmis odeme ahlakini gosteren en kritik risk faktorudur.",
            "liquidity_ratio": "Likidite orani. Birikimlerin toplam borca oranidir; acil nakit gucunu olcer.",
            "dscr": "Borc servis karsilama orani. Gelirin aylik borc odemesine oranidir; nakit akis kapasitesidir.",
            "risk_behavior_score": "Risk davranis skoru. Gecikme gunu ve sorgularin birlesik risk puanini ifade eder."
        }
        
        return {
            "kredi_durumu": kredi_durumu,
            "musteri_mesaji": musteri_mesaji,
            "faktor_aciklamalari": faktor_aciklamalari,
            "client_data": client_data,
            "shap_explanation": shap_analysis,
            "agentic_workflow": {
                "audit": audit_result,
                "strategist": strategist_result,
                "retention": retention_result
            }
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Kredi skorlama islemi basarisiz oldu: {str(exc)}")
