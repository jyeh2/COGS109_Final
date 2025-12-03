import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

file_path = './finaldata/combined_events.csv'
df = pd.read_csv(file_path)

print("="*80)
print("VARIANT 2: KNN WITH MANHATTAN DISTANCE (L1)")
print("="*80)

df_processed = df.copy()
categorical_cols = [
    'Location_Type', 'A.S. Advertisement Pass', 
    'Event_Type', 'Day_of_Week', 'Month'
]

label_encoders = {}
for col in categorical_cols:
    if col in df_processed.columns:
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le

target = 'Attendance_Ratio'
predictors = [
    'AWARDED', 'log_AWARDED', 'Location_Type_encoded',
    'A.S. Advertisement Pass_encoded', 'Event_Type_encoded',
    'Day_of_Week_encoded', 'Month_encoded'
]

k_value = 5
use_scaling = True

print(f"\n⚙️ Manhattan Distance Variant Config:")
print(f"  • Target: {target}")
print(f"  • Predictors: {predictors}")
print(f"  • K: {k_value}")
print(f"  • Distance Metric: Manhattan (L1)")
print(f"  • Scaling: Yes\n")

X = df_processed[predictors].copy()
y = df_processed[target].copy()

X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if use_scaling:
    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train)
    X_test_final = scaler.transform(X_test)
else:
    X_train_final = X_train.values
    X_test_final = X_test.values

print("\n🤖 Training KNN (Manhattan Distance)...")

knn = KNeighborsRegressor(
    n_neighbors=k_value,
    metric='manhattan'
)

knn.fit(X_train_final, y_train)
print("✓ Model trained!")

print("\n🔮 Making predictions...")

y_train_pred = knn.predict(X_train_final)
y_test_pred = knn.predict(X_test_final)

print("✓ Predictions complete!")

print("\n📈 Evaluating model performance...")

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

n_train = len(y_train)
n_test = len(y_test)
p = len(predictors)

train_adj_r2 = 1 - (1 - train_r2) * (n_train - 1) / (n_train - p - 1)
test_adj_r2 = 1 - (1 - test_r2) * (n_test - 1) / (n_test - p - 1)

print("\n" + "="*80)
print("PERFORMANCE METRICS — Manhattan Distance")
print("="*80)
print(f"\n{'Metric':<30} {'Training':<20} {'Test':<20}")
print("-" * 70)
print(f"{'RMSE':<30} {train_rmse:<20.4f} {test_rmse:<20.4f}")
print(f"{'MAE':<30} {train_mae:<20.4f} {test_mae:<20.4f}")
print(f"{'R²':<30} {train_r2:<20.4f} {test_r2:<20.4f}")
print(f"{'Adjusted R²':<30} {train_adj_r2:<20.4f} {test_adj_r2:<20.4f}")
print("-" * 70)

print("\n🔄 Performing 5-fold cross-validation...")
kfold = KFold(n_splits=5, shuffle=True, random_state=456)
cv_scores = cross_val_score(
    knn, X_train_final, y_train, 
    cv=kfold, scoring='neg_mean_squared_error'
)
cv_rmse = np.sqrt(-cv_scores.mean())

print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS")
print("="*80)
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

print("\n🔍 Calculating feature importance...")

perm_importance = permutation_importance(
    knn, X_test_final, y_test, n_repeats=10, random_state=42
)

feature_importance_df = pd.DataFrame({
    'Feature': predictors,
    'Importance': perm_importance.importances_mean,
    'Std': perm_importance.importances_std
}).sort_values('Importance', ascending=False)

print("\n" + "="*80)
print("FEATURE IMPORTANCE — Manhattan Distance")
print("="*80)
print(feature_importance_df.to_string(index=False))

train_resid = y_train - y_train_pred
test_resid = y_test - y_test_pred

print("\n📊 Residual Statistics (Test Set):")
print(f"  • Mean: {test_resid.mean():.4f}")
print(f"  • Std Dev: {test_resid.std():.4f}")
print(f"  • Min: {test_resid.min():.4f}")
print(f"  • Max: {test_resid.max():.4f}")

print("\n🎨 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
ax1.scatter(y_test, y_test_pred, alpha=0.6, color='coral', edgecolor='k', linewidth=0.5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel("Actual Attendance Ratio", fontsize=12, fontweight='bold')
ax1.set_ylabel("Predicted Attendance Ratio", fontsize=12, fontweight='bold')
ax1.set_title(f"Test Set: Predicted vs Actual\nR² = {test_r2:.4f}", fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.scatter(y_test_pred, test_resid, alpha=0.6, color='coral', edgecolor='k', linewidth=0.5)
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel("Predicted Attendance Ratio", fontsize=12, fontweight='bold')
ax2.set_ylabel("Residuals", fontsize=12, fontweight='bold')
ax2.set_title(f"Test Set: Residual Plot\nRMSE = {test_rmse:.4f}", fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
ax3.hist(test_resid, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
ax3.axvline(x=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel("Residuals", fontsize=12, fontweight='bold')
ax3.set_ylabel("Frequency", fontsize=12, fontweight='bold')
ax3.set_title(f"Test Set: Distribution of Residuals\nMean = {test_resid.mean():.4f}", fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

ax4 = axes[1, 1]
ax4.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], alpha=0.8, edgecolor='black')
ax4.set_xlabel("Permutation Importance", fontsize=12, fontweight='bold')
ax4.set_ylabel("Features", fontsize=12, fontweight='bold')
ax4.set_title("Feature Importance", fontsize=13, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('variant2_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()

print("\n✓ Variant 2 (Manhattan Distance) Complete!")

print("\n" + "="*80)
print("✅ VARIANT 2 ANALYSIS COMPLETE!")
print("="*80)
print(f"\n🎯 VARIANT 2 CROSS-VALIDATION RESULT:")
print(f"   5-Fold CV RMSE: {cv_rmse:.4f}")
print("="*80)
