# كاشف إعلانات الوظائف المزيفة

موقع بسيط بيحط نموذج `fake_job_postings.ipynb` قدام الناس: بيلصقوا تفاصيل إعلان
وظيفة، والموقع بيرجّع احتمال إن الإعلان نصب + الحاجات الملفتة فيه.

```
api/index.py          الـ API (FastAPI) - بيعيد بناء نفس الـ pipeline بتاع النوتبوك
public/index.html     الواجهة (عربي RTL، صفحة واحدة)
models/               مكان ملفات الـ .pkl - فاضي دلوقتي، لازم تملاه
export_for_vercel.py  الخلية اللي تشغّلها في Colab عشان تطلّع الملفات دي
```

## خطوة 1 — طلّع الموديل من Colab

افتح النوتبوك، شغّله كله، وبعدين ضيف **خلية جديدة في الآخر** وحط فيها محتوى
`export_for_vercel.py` وشغّلها. هتنزّلك `models.zip`.

الخلية دي بتعمل تلات حاجات مهمة:

- بتدرّب **Logistic Regression** مش XGBoost. السبب: حزمة xgboost لوحدها حوالي
  300MB مفكوكة، وحد الـ function على Vercel هو 250MB. النوتبوك نفسه لاحظ إن
  LogReg قريب من XGBoost في الـ PR-AUC، فده أرخص تنازل من إن الموقع ما يترفعش أصلًا.
- بتحفظ `top_countries`. النوتبوك مكنش بيحفظها، ومن غيرها مفيش طريقة تعرف بيها
  أنهي دولة تتحوّل لـ `other` وقت الفحص — وده كان هيبوّظ النتايج في صمت.
- بتطبع أرقام إصدارات المكتبات.

## خطوة 2 — حط الملفات وثبّت الإصدارات

فك `models.zip` جوه فولدر `models/`. المفروض يبقى عندك:

```
models/model.pkl
models/tfidf_vectorizer.pkl
models/onehot_encoder.pkl
models/standard_scaler.pkl
models/metadata.json
```

بعدين افتح `requirements.txt` وغيّر إصدارات `scikit-learn` و `scipy` و `numpy`
و `joblib` للأرقام اللي الخلية طبعتها. **الخطوة دي مش اختيارية** — لو الإصدار
مختلف، `joblib` إما هيرفض يفتح الملفات أو هيفتحها غلط.

## خطوة 3 — ارفع

```bash
git init && git add -A && git commit -m "fake job detector"
git remote add origin <رابط الريبو بتاعك>
git push -u origin main
```

وبعدين على vercel.com: New Project ← اختار الريبو ← Deploy. من غير أي إعدادات
زيادة، `vercel.json` فيه كل حاجة.

أو من التيرمنال:

```bash
npm i -g vercel
vercel
```

> ملفات الـ `.pkl` لازم تتعمللها commit في git. Vercel بتبني من الريبو، والـ
> function بتقراها من الديسك.

## تجربة محلي

```bash
pip install -r requirements.txt
uvicorn api.index:app --reload --port 8000
```

وافتح `public/index.html` في المتصفح (غيّر `/api/predict` لـ
`http://localhost:8000/api/predict` مؤقتًا).

للتأكد إن الموديل اتحمّل صح: `curl http://localhost:8000/api/health`

## حاجات ممكن تعدّلها

- **حد القرار**: متغير بيئة `FRAUD_THRESHOLD` على Vercel (الافتراضي `0.5`).
  الداتا غير متوازنة جدًا — لو عايز تمسك إعلانات نصب أكتر على حساب إنذارات
  كاذبة أكتر، نزّله لـ `0.3`. منحنى الـ precision-recall في النوتبوك هو اللي
  يقوللك تختار كام.
- **الملاحظات تحت النتيجة** في دالة `explain` في `api/index.py`. دي قواعد
  مكتوبة بإيدي من الـ EDA، مش خارجة من الموديل.

## تنبيه

الموديل اتدرّب على إعلانات **إنجليزية**. إعلان عربي هيدخل TF-IDF ويطلع شبه
فاضي، والنتيجة ساعتها هتبقى مبنية على الخانات الناقصة بس — يعني مش موثوقة.
لو عايز تدعم العربي، ده محتاج داتا تدريب عربية، مش تعديل في الكود.
