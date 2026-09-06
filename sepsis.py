import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#cleaning data
# ۱. بارگذاری دیتاست
df = pd.read_csv('C:\\Users\\user\\Downloads\\sepsis.csv')
# ۲. تبدیل مقادیر قراردادی (مثل ۰ در ستون‌های حیاتی یا علامت سوال) به NaN
# مثلاً ضربان قلب (HR) یا فشار خون (MAP) نمی‌تواند ۰ باشد.
cols_to_fix = df.columns.drop('SepsisLabel') # همه ستون‌ها جز هدف
for col in cols_to_fix:
# اگر در ستون‌های پزشکی عدد ۰ دیدی، یعنی داده ثبت نشده (چون ضربان قلب ۰ یعنی مرگ!)
    df[col] = df[col].replace({0: np.nan, '?': np.nan, 'NULL': np.nan, ' ': np.nan})
print(f"تعداد واقعی مقادیر خالی پیدا شده: {df.isnull().sum().sum()}")
# ۳. حالا عملیات پر کردن را انجام می‌دهیم
# پر کردن بر اساس سوابق قبلی (Forward Fill)
df = df.ffill()
# ۴. پر کردن باقی‌مانده‌ها (مواردی که از ابتدا خالی بودند) با میانه
df = df.fillna(df.median())
print(f"تعداد مقادیر خالی بعد از اصلاح نهایی: {df.isnull().sum().sum()}")
# چک کردن چند سطر اول برای اطمینان
print(df.head(5))
#scaling data
x=df.drop("SepsisLabel",axis=1)
y=df["SepsisLabel"]
x1_train,x1_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x1_train_scaled=scaler.fit_transform(x1_train)
x1_test_scaled=scaler.transform(x1_test)
print("مقیاس بندی با موفقیت انجام شد")
print(f"میانگین دادههای اسکیل شده:{x1_train_scaled.mean()}")
print(f"انحراف معیار داده های اسکیل شده:{x1_train_scaled.std():.2f}")
x1_train_final=pd.DataFrame(x1_train_scaled,columns=x.columns)
print("نمونه ی داده های اسکیل شده\n:")
print(x1_train_final.head())
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
X_train = x1_train.loc[:, x1_train.nunique() > 1]
X_test = x1_test[X_train.columns]
# ۲. محاسبه p-value ها به صورت یک لیست تخت (Flat List)
features_list = list(X_train.columns)
p_values = []
for col in features_list:
    g0 = X_train[col][y_train == 0]
    g1 = X_train[col][y_train == 1]
# انجام آزمون و گرفتن فقط مقدار p-value
    _, p = mannwhitneyu(g0, g1, alternative='two-sided')
    p_values.append(p)
# ۳. تبدیل به آرایه تک‌بعدی و اطمینان از طول صحیح
    p_values_array = np.array(p_values).flatten()
print(f"تعداد ویژگی‌ها: {len(features_list)}")
print(f"تعداد p-valueهای محاسبه شده: {len(p_values_array)}")
# ۴. اجرای تعدیل Holm با اطمینان از ورودی صحیح
# خروجی 'rejected' یک لیست از True/False به طول تعداد ویژگی‌هاست
rejected, p_corrected, _, _ = multipletests(p_values_array, alpha=0.05, method='holm')
# ۵. انتخاب ویژگی‌ها با استفاده از لیست پایتونی (برای امنیت بیشتر و جلوگیری از خطای پانداز)
selected_features = [f for f, r in zip(features_list, rejected) if r]
# ۶. اعمال خروجی روی دیتافریم‌ها
if len(selected_features) > 0:
    X_train_stat = X_train[selected_features]
    X_test_stat = X_test[selected_features]
    print(f"✅ با موفقیت انجام شد! {len(selected_features)} ویژگی معنادار انتخاب شدند.")
    print(f"نمونه ویژگی‌ها: {selected_features[:5]}")
else:
    print("⚠️ هیچ ویژگی از سد آزمون سخت‌گیرانه Holm رد نشد. پیشنهاد می‌کنم alpha را روی 0.1 تست کنی.")
    # ۷. نمایش توزیع کلاس‌ها برای آمادگی مرحله SMOTE
    print(f"\nتوزیع کلاس‌ها در آموزش: {dict(y_train.value_counts())}")
from imblearn.over_sampling import SMOTE
from collections import Counter
# ۱. تعریف الگوریتم SMOTE
# random_state را ثابت می‌گذاریم تا نتایج در دفعات بعد هم یکسان باشد
sm = SMOTE(random_state=42)
# ۲. ایجاد نمونه‌های مصنوعی (Resampling)
# دقت کن که فقط روی داده‌های آموزش (X_train_stat) این کار را انجام می‌دهیم
X_res, y_res = sm.fit_resample(X_train_stat, y_train)
# ۳. گزارش وضعیت توازن جدید
print("زهرا جان، وضعیت توازن داده‌ها اصلاح شد:")
print(f"قبل از متعادل‌سازی: {Counter(y_train)}")
print(f"بعد از متعادل‌سازی: {Counter(y_res)}")
# ۴. بررسی ابعاد نهایی
print("-" * 30)
print(f"تعداد ردیف‌های جدید برای آموزش: {X_res.shape[0]}")
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
# ۱. تعریف مدل‌ها
# مدل اول: Random Forest (با ۱۰۰ درخت تصمیم)
rf_model = RandomForestClassifier(
n_estimators=100,
class_weight={0: 1, 1: 5}, # یعنی اشتباه روی بیمار ۵ برابر سنگین‌تر از اشتباه روی فرد سالم است
random_state=42
)
# ۲. آموزش مجدد
rf_model.fit(X_res, y_res)
# ۳. پیش‌بینی
rf_pred = rf_model.predict(X_test_stat)
print("✅ مدل با جریمه سنگین‌تر برای تشخیص ندادن بیمار آموزش دید!")
# مدل دوم: Neural Network (یک شبکه عصبی با دو لایه مخفی ۱۰۰ و ۵۰ نرونی)
# ما از max_iter بالا استفاده می‌کنیم تا شبکه فرصت یادگیری داشته باشد
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
# ۲. آموزش مدل‌ها با داده‌های متعادل شده (SMOTE شده)
print("در حال آموزش مدل‌ها... کمی صبر کن زهرا جان.")
rf_model.fit(X_res, y_res)
nn_model.fit(X_res, y_res)
# ۳. پیش‌بینی روی داده‌های تست (داده‌های واقعی و دست‌نخورده)
rf_pred = rf_model.predict(X_test_stat)
nn_pred = nn_model.predict(X_test_stat)
# ۴. گزارش نتایج برای مقایسه
print("\n" + "="*20 + " نتایج Random Forest " + "="*20)
print(classification_report(y_test, rf_pred))
print("\n" + "="*20 + " نتایج Neural Network " + "="*20)
print(classification_report(y_test, nn_pred))
# ۵. محاسبه نهایی برای اعلام برنده
rf_f1 = f1_score(y_test, rf_pred)
nn_f1 = f1_score(y_test, nn_pred)
print("\n" + "*"*10 + " نتیجه نهایی رقابت " + "*"*10)
print(f"امتیاز F1 برای جنگل تصادفی: {rf_f1:.2f}")
print(f"امتیاز F1 برای شبکه عصبی: {nn_f1:.2f}")
if rf_f1 > nn_f1:
    print("برنده: Random Forest! (برای داده‌های جدولی پایدارتر عمل کرد)")
else:
    print("برنده: Neural Network! (الگوهای پیچیده را بهتر شناسایی کرد)")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# ۱. محاسبه ماتریس آشفتگی برای مدل برنده (Random Forest)
cm = confusion_matrix(y_test, rf_pred)
# ۲. رسم نمودار
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Sepsis'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Final Diagnosis Confusion Matrix (Random Forest)')
plt.show()
# ۳. نمایش اهمیت ویژگی‌ها (بخش جذاب برای گزارش)
importances = rf_model.feature_importances_
feat_importances = pd.Series(importances, index=X_train_stat.columns)
feat_importances.nlargest(10).plot(kind='barh', color='teal')
plt.title('Top 10 Most Important Medical Features')
plt.show()