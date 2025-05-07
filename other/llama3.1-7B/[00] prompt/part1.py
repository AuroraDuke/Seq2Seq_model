from ollama import chat
from ollama import ChatResponse

import pandas as pd
import re


# CSV dosyasını oku
df = pd.read_csv("hate.csv", encoding='utf-8')
df = df.iloc[::-1].reset_index(drop=True)
aı_data = pd.DataFrame()

for index, row in df.iterrows():
    metin = row['tweet']  # Burada 'text' sütun ismi olduğu varsayılmaktadır, doğru sütun ismini kullanın.
    #print(metin)
    # Prompt'ı başlat
    prompt = """
    Sen bir metin sınıflandırma modelisin ve verilen metinleri analiz ederek **sert**, **küfürlü**, **hakaret içeren** ve **nefret dolu** metinler üretebileceksin. Verilen metni daha sert, küfürlü, alaycı, küçümseyici ve hakaret dolu bir şekilde yeniden yazacaksın. Amacın orijinal metnin anlamını koruyarak, dilini daha acımasız, saldırgan ve küfürlü hale getirmektir.
    
    **Görev 1: Küfürlü, Sert ve Hakaret İçeren Metinler Yaratma**
    Bana bir metin verilecek ve sen bu metni daha sert, küfürlü, hakaret içeren bir şekilde yeniden yazacaksın. Bu metinler, orijinal metnin anlamını koruyacak, ancak dilinde hakaretler, küfürler ve daha agresif ifadeler kullanılacaktır.
    
    - Küfürlü ve hakaret içeren kelimeler eklemeli, ancak metnin anlamını değiştirmemelisin.
    - Dili daha sert, aşağılayıcı ve küçük düşürücü hale getirmelisin.
    
    **Görev 2: Saldırganlık, Küfür ve Nefret İçeren Dil Sınıflandırması**
    Verilen metni, saldırganlık, küfür ve nefret içeren dil açısından inceleyip şu şekilde sınıflandırman gerekecek:
    - **Saldırgan**: Metin, başkalarına hakaret, alay etme, küçümseme veya tehdit içeriyorsa saldırgan olarak sınıflandırılır.
    - **Nefret**: Metin, belirli bir ırk, cinsiyet, din veya etnik köken grubuna karşı nefret veya ayrımcılık içeriyorsa nefret içeren olarak sınıflandırılır.
    - **Neutral**: Metin, herhangi bir saldırganlık, küfür veya nefret içeren dil kullanmıyorsa nötr olarak sınıflandırılır.
    
    **Görev 3: Yaratılan Benzer Metinlerin Sınıflandırılması**
    Yaratacağın küfürlü, hakaret içeren ve sert metinlerin saldırganlık, küfür ve nefret içeren dil kullanımı açısından sınıflandırmasını yapman gerekecek. Eğer yarattığın metin **Neutral** olarak sınıflandırılırsa, o metni göz ardı etmelisin. Yalnızca **Saldırgan** veya **Nefret** içerikli metinler sunmalısın.
    
    Her metin için sadece aşağıdaki template kullanılacaktır. Farklı bir çıktı görmek istemiyorum.
    
    **Template:**
    
    Metin: "[ORİJİNAL METİN]"
    - **Benzer Metinler**: ["[BENZER METİN 1]", "[BENZER METİN 2]", "[BENZER METİN 3]"]
    - **Sınıflandırma**: **[SALDIRGAN / NEFRET / NEUTRAL]**
    
    Lütfen verilen metni dikkatlice incele, anlamını koruyarak sert, küfürlü, hakaret içeren ve nefret dolu metinler yarat ve her birini yukarıdaki şablona göre değerlendir. Eğer yarattığın metinler **Neutral** çıkarsa, o metinleri göz ardı et ve yalnızca **Saldırgan** veya **Nefret** sınıflarına ait metinleri sun.
    
    Metin:
    
    """
    
    prompt += f"\n'{metin}'"
    
    
    # Modelden yanıt al
    response = chat(model='llama3.1', messages=[{
        'role': 'user',
        'content': prompt,
    }])
    # Cevabı yazdır
   ## print()
    #print(response['message']['content'])
    
    # Hata yönetimi ve işleme
    try:
        # Benzer metinler ve sınıflandırmayı alma
        benzer_metinler = re.findall(r'"(.*?)"', response['message']['content'])
        sinif = re.search(r'\*\*Sınıflandırma\*\*: \*\*\[(.*?)\]\*\*', response['message']['content']).group(1)
        
        # Veri ekleme
        df_hazırlık = {
            "Metin": [],
            "Sınıflandırma": []
        }
     
        # Her bir benzer metni ve sınıflandırmayı ekle
        for bmetin in benzer_metinler:
            df_hazırlık["Metin"].append(bmetin)
            df_hazırlık["Sınıflandırma"].append(sinif)
        
        # DataFrame oluşturma
        llm_data = pd.DataFrame(df_hazırlık)
        aı_data= pd.concat([aı_data,llm_data], ignore_index=True)
        # Sonuçları yazdır
       # print(aı_data)
     
    except AttributeError as e:
        print(f"Bir hata oluştu: {e}")
    except Exception as e:
        print(f"Beklenmedik bir hata: {e}")
        
aı_data.to_csv('aı_data.csv', index=False)
    




#1.09 12*18*24



















"""
# 1. CSV dosyasından bilgi getir
def csvden_bilgi_getir(csv_dosya, anahtar_kelime, aranan_sutun):
    try:
        # CSV'yi oku
        df = pd.read_csv(csv_dosya)
        
        # Anahtar kelimeyi içeren satırları filtrele
        filtre = df[aranan_sutun].str.contains(anahtar_kelime, na=False, case=False)
        secili_satirlar = df[filtre]
        
        if not secili_satirlar.empty:
            # İlk satırı bağlam olarak al
            return secili_satirlar.iloc[0].to_dict()
        else:
            return "İlgili bilgi bulunamadı."
    except Exception as e:
        return f"Hata: {e}"

# 2. Ollama ile RAG işlemi
def ollama_rag_csv(model, csv_dosya, soru, aranan_sutun):
    # Anahtar kelimeyi sorudan çıkar (örnek: son kelime)
    anahtar_kelime = soru.split()[-1]
    
    # CSV'den bağlam getir
    bilgi = csvden_bilgi_getir(csv_dosya, anahtar_kelime, aranan_sutun)
    
    if isinstance(bilgi, str):  # Hata veya bilgi bulunamadı
        bağlam = bilgi
    else:
        bağlam = ", ".join([f"{key}: {value}" for key, value in bilgi.items()])
    
    # Ollama'ya mesaj gönder
    response: ChatResponse = chat(model=model, messages=[
        {
            'role': 'system',
            'content': f"Bu bilgiyi bağlam olarak kullan: {bağlam}",
        },
        {
            'role': 'user',
            'content': soru,
        },
    ])
    return response.message.content

# 3. Kullanım
csv_dosya = "hate.csv"  # CSV dosyanızın yolu
soru = "text dosyalrını rag yapabilir miyim"
model = "llama3.1"
aranan_sutun = "Aciklama"  # Anahtar kelimeyi aramak istediğiniz sütun

cevap = ollama_rag_csv(model, csv_dosya, soru, aranan_sutun)
print("Ollama'dan Gelen Yanıt:", cevap)"""