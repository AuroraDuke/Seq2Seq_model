# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 21:31:36 2024

@author: ilker
"""

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
    print(metin)
    # Prompt'ı başlat
    prompt = """
    Sen bir metin sınıflandırma modelisin ve verilen metinleri analiz ederek **sert**, **küfürlü**, **hakaret içeren** ve **saldırgan** metinler üretebileceksin. Verilen metni daha sert, küfürlü, alaycı, küçümseyici ve hakaret dolu bir şekilde yeniden yazacaksın. Amacın orijinal metnin anlamını koruyarak, dilini daha acımasız, saldırgan ve küfürlü hale getirmektir.
    
    **Görev 1: Küfürlü, Sert ve Hakaret İçeren Metinler Yaratma**  
    Bana bir metin verilecek ve sen bu metni daha sert, küfürlü, hakaret içeren bir şekilde yeniden yazacaksın. Bu metinler, orijinal metnin anlamını koruyacak, ancak dilinde hakaretler, küfürler ve daha agresif ifadeler kullanılacaktır.
    
    - Küfürlü ve hakaret içeren kelimeler eklemeli, ancak metnin anlamını değiştirmemelisin.
    - Dili daha sert, aşağılayıcı ve küçük düşürücü hale getirmelisin.
    
    **Görev 2: Saldırganlık ve Küfür İçeren Dil Sınıflandırması**  
    Verilen metni, saldırganlık ve küfür içeren dil açısından inceleyip şu şekilde sınıflandırman gerekecek:
    - **Saldırgan**: Metin, başkalarına hakaret, alay etme, küçümseme veya tehdit içeriyorsa saldırgan olarak sınıflandırılır.
    
    **Görev 3: Yaratılan Benzer Metinlerin Sınıflandırılması**  
    Yaratacağın küfürlü, hakaret içeren ve sert metinlerin saldırganlık ve küfür içeren dil kullanımı açısından sınıflandırmasını yapman gerekecek. Yalnızca **Saldırgan** içerikli metinler sunmalısın.
    
    Her metin için sadece aşağıdaki template kullanılacaktır. Farklı bir çıktı görmek istemiyorum.
    
    **Template:**
    
    Metin: "[ORİJİNAL METİN]"  
    - **Benzer Metinler**: ["[BENZER METİN 1]", "[BENZER METİN 2]", "[BENZER METİN 3]"]  
    - **Sınıflandırma**: **[SALDIRGAN]**
    
    Lütfen verilen metni dikkatlice incele, anlamını koruyarak sert, küfürlü, hakaret içeren ve saldırgan metinler yarat ve her birini yukarıdaki şablona göre değerlendir.

    Metin:
    
    """
    
    prompt += f"\n'{metin}'"
    
    
    # Modelden yanıt al
    response = chat(model='llama3.1', messages=[{
        'role': 'user',
        'content': prompt,
    }])
    # Cevabı yazdır
    print()
    print(response['message']['content'])
    
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
        print(aı_data)
     
    except AttributeError as e:
        print(f"Bir hata oluştu: {e}")
    except Exception as e:
        print(f"Beklenmedik bir hata: {e}")
        
aı_data.to_csv('aı_hate_data.csv', index=False)