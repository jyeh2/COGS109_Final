import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 10)

df = pd.read_csv('./finaldata/combined_events.csv')
df_processed = df.copy()
df_processed = df_processed.dropna()

categorical_cols = ['Location_Type', 'A.S. Advertisement Pass', 'Event_Type', 'Day_of_Week']
label_encoders = {}
for col in categorical_cols:
    if col in df_processed.columns:
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le

df_processed['log_AWARDED'] = np.log1p(df_processed['AWARDED'])
df_processed['DATE'] = pd.to_datetime(df_processed['DATE'])
df_processed['Month'] = df_processed['DATE'].dt.month
le_month = LabelEncoder()
df_processed['Month_encoded'] = le_month.fit_transform(df_processed['Month'].astype(str))

features = ['log_AWARDED', 'Month_encoded', 'Event_Type_encoded', 'Location_Type_encoded']
X = df_processed[features]
y = df_processed['Attendance_Ratio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsRegressor(n_neighbors=3, weights='distance', metric='manhattan')
knn.fit(X_train_scaled, y_train)

y_train_pred = knn.predict(X_train_scaled)
y_test_pred = knn.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

n = len(y_test)
p = len(features)
adj_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)

cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='neg_root_mean_squared_error')
cv_rmse = -cv_scores.mean()

print("VARIANT 3: OPTIMIZED KNN")
print("="*50)
print(f"Features: {features}")
print(f"K: 3, Weights: distance, Metric: Manhattan")
print(f"\nTrain R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Adjusted R²: {adj_r2:.4f}")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"CV RMSE: {cv_rmse:.4f}")

perm_importance = permutation_importance(knn, X_test_scaled, y_test, n_repeats=30, random_state=42)
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': perm_importance.importances_mean,
    'Std': perm_importance.importances_std
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
for _, row in feature_importance_df.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

train_resid = y_train - y_train_pred
test_resid = y_test - y_test_pred

fig, axes = plt.subplots(3, 2, figsize=(16, 18))

axes[0, 0].scatter(y_train, y_train_pred, alpha=0.6, s=50, color='steelblue')
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Predicted', fontsize=12, fontweight='bold')
axes[0, 0].set_title(f'Training Set: Predicted vs Actual\nR² = {train_r2:.4f}', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(y_test, y_test_pred, alpha=0.6, s=50, color='coral')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Predicted', fontsize=12, fontweight='bold')
axes[0, 1].set_title(f'Test Set: Predicted vs Actual\nR² = {test_r2:.4f}', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(y_train_pred, train_resid, alpha=0.6, s=50, color='steelblue')
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Residuals', fontsize=12, fontweight='bold')
axes[1, 0].set_title(f'Training Set: Residual Plot\nRMSE = {train_rmse:.4f}', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(y_test_pred, test_resid, alpha=0.6, s=50, color='coral')
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Residuals', fontsize=12, fontweight='bold')
axes[1, 1].set_title(f'Test Set: Residual Plot\nRMSE = {test_rmse:.4f}', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

axes[2, 0].hist(train_resid, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[2, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[2, 0].set_xlabel('Residuals', fontsize=12, fontweight='bold')
axes[2, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[2, 0].set_title(f'Training Set: Distribution of Residuals\nMean = {train_resid.mean():.4f}', fontsize=13, fontweight='bold')
axes[2, 0].grid(True, alpha=0.3, axis='y')

axes[2, 1].hist(test_resid, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
axes[2, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[2, 1].set_xlabel('Residuals', fontsize=12, fontweight='bold')
axes[2, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[2, 1].set_title(f'Test Set: Distribution of Residuals\nMean = {test_resid.mean():.4f}', fontsize=13, fontweight='bold')
axes[2, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('variant3_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(12, 8))
colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in feature_importance_df['Importance']]
plt.barh(range(len(feature_importance_df)), feature_importance_df['Importance'], 
         xerr=feature_importance_df['Std'], color=colors, alpha=0.8, edgecolor='black')
plt.yticks(range(len(feature_importance_df)), feature_importance_df['Feature'])
plt.xlabel('Permutation Importance', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Variant 3: Feature Importance Analysis', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', lw=0.5)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('variant3_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()