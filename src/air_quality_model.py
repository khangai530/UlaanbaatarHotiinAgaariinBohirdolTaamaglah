# ============================================================
# Улаанбаатар хотын агаарын бохирдлын таамаглал
# Машин сургалтын загвар: Linear Regression, Decision Tree,
#                         Bayesian Ridge
# Өгөгдөл: Open-Meteo (цаг агаар + PM2.5)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. ӨГӨГДӨЛ АЧААЛАХ
# ============================================================

print("=" * 60)
print("1. ӨГӨГДӨЛ АЧААЛАХ")
print("=" * 60)

train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

print(f"Сургалтын өгөгдөл: {train_df.shape[0]} мөр × {train_df.shape[1]} багана")
print(f"Туршилтын өгөгдөл: {test_df.shape[0]} мөр × {test_df.shape[1]} багана")
print(f"\nБаганууд: {list(train_df.columns)}")

# ============================================================
# 2. FEATURE / TARGET ЯЛГАХ
# ============================================================

print("\n" + "=" * 60)
print("2. FEATURE / TARGET ЯЛГАХ")
print("=" * 60)

FEATURES = [
    'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
    'precipitation', 'surface_pressure',
    'month', 'hour', 'heating_season', 'temp_x_wind'
]
TARGET = 'pm2_5'

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_test  = test_df[FEATURES]
y_test  = test_df[TARGET]

print(f"Оролтын хувьсагчид ({len(FEATURES)} ш): {FEATURES}")
print(f"Зорилтот хувьсагч: {TARGET}")

# ============================================================
# 3. СТАНДАРТЧЛАЛ (Linear Regression болон Bayesian Ridge-д хэрэгтэй)
# ============================================================

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ============================================================
# 4. ЗАГВАРУУД СУРГАХ
# ============================================================

print("\n" + "=" * 60)
print("3. ЗАГВАРУУД СУРГАХ")
print("=" * 60)

# --- 4.1 Linear Regression ---
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)
print("✓ Linear Regression сургагдлаа")

# --- 4.2 Bayesian Ridge Regression ---
br = BayesianRidge(max_iter=500)
br.fit(X_train_sc, y_train)
y_pred_br = br.predict(X_test_sc)
print("✓ Bayesian Ridge Regression сургагдлаа")

# --- 4.3 Decision Tree Regressor ---
dt = DecisionTreeRegressor(max_depth=6, min_samples_leaf=50, random_state=42)
dt.fit(X_train, y_train)          # Decision Tree стандартчлал шаардахгүй
y_pred_dt = dt.predict(X_test)
print("✓ Decision Tree Regressor сургагдлаа")

# ============================================================
# 5. ҮНЭЛГЭЭ
# ============================================================

print("\n" + "=" * 60)
print("4. ЗАГВАРУУДЫН ҮНЭЛГЭЭ")
print("=" * 60)

def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n[{name}]")
    print(f"  MAE  (Дундаж абсолют алдаа)  : {mae:.4f} µg/m³")
    print(f"  RMSE (Квадрат алдааны язгуур): {rmse:.4f} µg/m³")
    print(f"  R²   (Тайлбарлах чадвар)     : {r2:.4f}")
    return {'Загвар': name, 'MAE': round(mae,4), 'RMSE': round(rmse,4), 'R²': round(r2,4)}

results = []
results.append(evaluate("Linear Regression",  y_test, y_pred_lr))
results.append(evaluate("Bayesian Ridge",      y_test, y_pred_br))
results.append(evaluate("Decision Tree",       y_test, y_pred_dt))

results_df = pd.DataFrame(results)
print("\n" + "=" * 40)
print("ХАРЬЦУУЛАЛТЫН ХҮСНЭГТ")
print("=" * 40)
print(results_df.to_string(index=False))

# ============================================================
# 6. LINEAR REGRESSION — КОЭФФИЦИЕНТҮҮД
# ============================================================

print("\n" + "=" * 60)
print("5. LINEAR REGRESSION — КОЭФФИЦИЕНТҮҮД")
print("=" * 60)

coef_df = pd.DataFrame({
    'Хувьсагч': FEATURES,
    'Коэффициент': lr.coef_
}).sort_values('Коэффициент', key=abs, ascending=False)

print(coef_df.to_string(index=False))
print(f"\nОгтлол (intercept): {lr.intercept_:.4f}")

# ============================================================
# 7. DECISION TREE — FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("6. DECISION TREE — FEATURE IMPORTANCE")
print("=" * 60)

imp_df = pd.DataFrame({
    'Хувьсагч': FEATURES,
    'Importance': dt.feature_importances_
}).sort_values('Importance', ascending=False)

print(imp_df.to_string(index=False))

# Decision Tree дүрмийн эхний 3 давхарга
print("\n--- Шийдвэрийн модны дүрэм (depth=3 хүртэл) ---")
print(export_text(dt, feature_names=FEATURES, max_depth=3))

# ============================================================
# 8. БАЙЕСЫН RIDGE — ТОДОРХОЙГҮЙН ИНТЕРВАЛ
# ============================================================

print("\n" + "=" * 60)
print("7. BAYESIAN RIDGE — УРЬДЧИЛСАН ТААМАГЛАЛ + ТОДОРХОЙГҮЙ")
print("=" * 60)

y_pred_br_mean, y_pred_br_std = br.predict(X_test_sc, return_std=True)
sample_idx = [0, 1, 2, 3, 4]
print(f"{'Бодит':>10}  {'Таамаглал':>10}  {'Std':>8}  {'95% CI':>20}")
print("-" * 55)
for i in sample_idx:
    lo = y_pred_br_mean[i] - 1.96 * y_pred_br_std[i]
    hi = y_pred_br_mean[i] + 1.96 * y_pred_br_std[i]
    print(f"{y_test.iloc[i]:>10.2f}  {y_pred_br_mean[i]:>10.2f}  "
          f"{y_pred_br_std[i]:>8.2f}  [{lo:.2f}, {hi:.2f}]")

# ============================================================
# 9. ГРАФИК ЗУРАХ
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Улаанбаатар PM2.5 таамаглал — Загваруудын харьцуулалт", fontsize=14)

models = [
    ("Linear Regression",  y_pred_lr),
    ("Bayesian Ridge",     y_pred_br),
    ("Decision Tree",      y_pred_dt),
]

# Дээд мөр: Predicted vs Actual scatter
for ax, (name, y_pred) in zip(axes[0], models):
    ax.scatter(y_test, y_pred, alpha=0.3, s=8, color='steelblue')
    lim = [0, 90]
    ax.plot(lim, lim, 'r--', lw=1.5, label="Тохирол шугам")
    ax.set_xlabel("Бодит PM2.5")
    ax.set_ylabel("Таамагласан PM2.5")
    r2 = r2_score(y_test, y_pred)
    ax.set_title(f"{name}\nR²={r2:.3f}")
    ax.legend(fontsize=8)

# Доод мөр, 0: Загваруудын R² харьцуулалт
ax = axes[1][0]
colors = ['#4C72B0', '#55A868', '#C44E52']
bars = ax.bar(results_df['Загвар'], results_df['R²'], color=colors)
ax.set_ylim(0, 1)
ax.set_ylabel("R² оноо")
ax.set_title("Загваруудын R² харьцуулалт")
for bar, val in zip(bars, results_df['R²']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha='center', fontsize=10)

# Доод мөр, 1: Feature Importance (Decision Tree)
ax = axes[1][1]
ax.barh(imp_df['Хувьсагч'], imp_df['Importance'], color='#55A868')
ax.set_xlabel("Importance")
ax.set_title("Decision Tree — Feature Importance")
ax.invert_yaxis()

# Доод мөр, 2: Bayesian Ridge uncertainty (эхний 200 бичлэг)
ax = axes[1][2]
n = 200
idx = range(n)
ax.plot(idx, y_test.iloc[:n].values, 'k-', lw=1, label="Бодит", alpha=0.7)
ax.plot(idx, y_pred_br_mean[:n], 'b-', lw=1, label="Таамаглал")
ax.fill_between(idx,
                y_pred_br_mean[:n] - 1.96 * y_pred_br_std[:n],
                y_pred_br_mean[:n] + 1.96 * y_pred_br_std[:n],
                alpha=0.3, color='blue', label="95% CI")
ax.set_xlabel("Цаг (index)")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title("Bayesian Ridge — Тодорхойгүйн интервал")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n✓ График 'results.png' болгон хадгалагдлаа")

print("\n" + "=" * 60)
print("АЖИЛ ДУУСЛАА")
print("=" * 60)