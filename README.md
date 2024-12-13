# Kredi Kartı Dolandırıcılığı Tespiti: Dengesiz Verilerle Mücadele ve Makine Öğrenmesi Yaklaşımı

Bu projede, kredi kartı dolandırıcılığını tespit etmek için makine öğrenmesi modelleri kullanılmıştır. Kullanılan veri seti, sınıf dengesizliği sorunu barındırdığı için çeşitli dengeleme yöntemleri uygulanmıştır. Projede, model eğitimi ve değerlendirme süreçleri Python programlama dili kullanılarak gerçekleştirilmiştir.

## Veri Seti Bilgileri

**Açıklama:**

- Veri seti, Eylül 2013'te Avrupa'daki kredi kartı kullanıcılarının işlemlerini içermektedir.
- İki günlük süre boyunca yapılan işlemler kaydedilmiş olup, toplamda 284,807 işlem içermektedir.
- Bu işlemlerden 492 tanesi dolandırıcılık (fraud) olarak işaretlenmiştir. Bu da veri setinde %0.172 oranında dengesiz bir sınıf dağılımı olduğunu göstermektedir.

**Özellikler:**

- Veri seti yalnızca nümerik özellikler içermektedir ve bu özellikler, PCA (Principal Component Analysis) dönüşümüne tabi tutulmuştur.
- `V1`, `V2`, ..., `V28` özellikleri PCA ile elde edilmiştir.
- `Time`: İlk işlemden itibaren geçen süre (saniye cinsinden).
- `Amount`: İşlem tutarı (miktar).
- `Class`: Hedef değişken. 1, dolandırıcılık işlemi; 0 ise normal işlemi temsil eder.

**Değerlendirme Metriği:**

Veri setindeki sınıf dengesizliği nedeniyle, modelin başarısını ölçmek için **AUC-ROC Skoru** ve **Precision-Recall Eğrisi** kullanılmıştır.

---

## Kullanılan Kütüphaneler

```python
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import matplotlib

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
```

---

## Proje Süreci

### Veri Dengesizliği ve Çözüm

Veri setindeki ciddi sınıf dengesizliği, dolandırıcılığı tespit etme performansını olumsuz etkileyebilir. Bu sorunu gidermek için **SMOTE (Synthetic Minority Over-sampling Technique)** ve **Random Under-Sampling** yöntemleri kullanılmıştır.

**SMOTE Kullanımı:**

- SMOTE, azınlık sınıfına ait yeni örnekler oluşturur ve veri setini dengeler.
- Bu yöntemi kullanarak daha hassas bir model eğitilmesi hedeflenmiştir.

### Model Eğitimi

Aşağıdaki modeller kullanılmış ve en iyi performans, **Random Forest** modeli ile elde edilmiştir:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machines (SVM)
4. Random Forest
5. Decision Tree

---

## En İyi Model Sonuçları

Random Forest modeli ile yapılan eğitim sonuçları aşağıdaki gibidir:

**Confusion Matrix:**

```
 [[85288    19] 
 [   16   120]]
```

**Classification Report:**

```
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.86      0.88      0.87       136

    accuracy                           1.00     85443
   macro avg       0.93      0.94      0.94     85443
weighted avg       1.00      1.00      1.00     85443
```

**ROC-AUC Skoru:** 0.9780

---

## Analiz ve Sonuç

Random Forest modeli ile eğitim yaparak elde ettiğimiz sonuçlar, modelimizin dolandırıcılık tespitinde oldukça başarılı olduğunu göstermektedir.

- **ROC-AUC Skoru** modelin dolandırıcılığı tespit etme yeteneğinin güçlü olduğunu işaret etmektedir.
- Test setindeki sınıf dengesizliğine rağmen, dolandırıcılık ve normal işlemleri ayırt etmede etkili bir sonuç elde edilmiştir.
- Ancak dolandırıcılık sınıfında (`Class 1`) bazı işlemlerin gözden kaçabileceğini unutmamak gerekir.(Recall = 0.88) 

**Sonuç:**

Yaptığımız analizler sonucunda, en uygun modelin Random Forest olduğunu belirledik. Ancak veri setindeki ciddi sınıf dengesizliği, modelin dolandırıcılığı tespit etme performansını sınırlamaktadır. Bu sorunun üstesinden gelmek için farklı yöntemler veya daha büyük ve dengeli bir veri seti kullanılabilir.

---

## Proje Detayları

- Veri Seti: [Kaggle](https://www.kaggle.com/) üzerinden temin edilmiştir.
- Python Sürümü: 3.10.14
- Lisans: Yok

Proje hakkında daha fazla bilgi veya kodlar için [Kaggle Projesi Linki](#) adresine göz atabilirsiniz.

