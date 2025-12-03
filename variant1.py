import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from itertools import combinations

file_path = './finaldata/combined_events.csv'
df = pd.read_csv(file_path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("=" * 80)
print("VARIANT 1: ALL FEATURES, K=5, STANDARDIZED")
print("=" * 80)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

df_processed = df.copy()

print("\n🔄 Encoding categorical variables...")
label_encoders = {}
categorical_cols = ['Location_Type', 'A.S. Advertisement Pass', 'Event_Type', 'Day_of_Week', 'Month']

for col in categorical_cols:
    if col in df_processed.columns:
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le

target = 'Attendance_Ratio'
predictors = ['AWARDED', 'log_AWARDED', 'Location_Type_encoded', 
              'A.S. Advertisement Pass_encoded', 'Event_Type_encoded', 
              'Day_of_Week_encoded', 'Month_encoded']
k_value = 5
use_scaling = True

print(f"\n⚙️ Configuration:")
print(f"  • Target variable: {target}")
print(f"  • Number of predictors: {len(predictors)}")
print(f"  • Predictors: {predictors}")
print(f"  • K neighbors: {k_value}")
print(f"  • Data scaling: {'Yes (StandardScaler)' if use_scaling else 'No'}")

print("\n📊 Preparing data...")

X = df_processed[predictors].copy()
y = df_processed[target].copy()

X = X.fillna(X.mean())

print(f"  • Features shape: {X.shape}")
print(f"  • Target shape: {y.shape}")
print(f"  • Missing values: {X.isnull().sum().sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✂️ Data split:")
print(f"  • Training samples: {len(X_train)}")
print(f"  • Test samples: {len(X_test)}")

if use_scaling:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_final = X_train_scaled
    X_test_final = X_test_scaled
    print(f"\n✓ Data standardized using StandardScaler")
else:
    X_train_final = X_train.values
    X_test_final = X_test.values
    print(f"\n✓ Using raw data (no scaling)")

print(f"\n🤖 Training KNN model with k={k_value}...")

knn = KNeighborsRegressor(n_neighbors=k_value)
knn.fit(X_train_final, y_train)

print("✓ Model trained successfully!")

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

print("\n🔄 Performing 5-fold cross-validation...")
kfold = KFold(n_splits=5, shuffle=True, random_state=123)
cv_scores = cross_val_score(knn, X_train_final, y_train, cv=kfold, 
                            scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores.mean())

print("\n" + "=" * 80)
print("PERFORMANCE METRICS")
print("=" * 80)
print(f"\n{'Metric':<30} {'Training':<20} {'Test':<20}")
print("-" * 70)
print(f"{'RMSE':<30} {train_rmse:<20.4f} {test_rmse:<20.4f}")
print(f"{'MAE':<30} {train_mae:<20.4f} {test_mae:<20.4f}")
print(f"{'R²':<30} {train_r2:<20.4f} {test_r2:<20.4f}")
print(f"{'Adjusted R²':<30} {train_adj_r2:<20.4f} {test_adj_r2:<20.4f}")
print(f"{'5-Fold CV RMSE':<30} {cv_rmse:<20.4f} {'N/A':<20}")
print("-" * 70)

print("\n🔍 Calculating feature importance...")

perm_importance = permutation_importance(knn, X_test_final, y_test, 
                                         n_repeats=10, random_state=42)

feature_importance_df = pd.DataFrame({
    'Feature': predictors,
    'Importance': perm_importance.importances_mean,
    'Std': perm_importance.importances_std
}).sort_values('Importance', ascending=False)

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE (Permutation)")
print("=" * 80)
print(feature_importance_df.to_string(index=False))

summary = f"""
VARIANT 1: KNN MODEL SUMMARY
{'=' * 80}

MODEL CONFIGURATION:
  • Algorithm: K-Nearest Neighbors Regression
  • K neighbors: {k_value}
  • Number of predictors: {len(predictors)}
  • Data scaling: {'StandardScaler' if use_scaling else 'None'}
  
PREDICTORS:
{chr(10).join([f'  • {p}' for p in predictors])}

PERFORMANCE METRICS:
{'─' * 80}
Training Set:
  • RMSE: {train_rmse:.4f}
  • MAE: {train_mae:.4f}
  • R²: {train_r2:.4f}
  • Adjusted R²: {train_adj_r2:.4f}

Test Set:
  • RMSE: {test_rmse:.4f}
  • MAE: {test_mae:.4f}
  • R²: {test_r2:.4f}
  • Adjusted R²: {test_adj_r2:.4f}

Cross-Validation:
  • 5-Fold CV RMSE: {cv_rmse:.4f}

FEATURE IMPORTANCE:
{'─' * 80}
{feature_importance_df.to_string(index=False)}

INTERPRETATION:
  • The model explains {test_r2*100:.1f}% of variance in attendance
  • Average prediction error (RMSE): {test_rmse:.2f} attendance ratio units
  • Most important feature: {feature_importance_df.iloc[0]['Feature']}
"""
print(summary)

print("\n📊 Performing residual analysis...")

train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

print(f"\nResidual Statistics (Test Set):")
print(f"  • Mean: {test_residuals.mean():.4f}")
print(f"  • Std Dev: {test_residuals.std():.4f}")
print(f"  • Min: {test_residuals.min():.4f}")
print(f"  • Max: {test_residuals.max():.4f}")

print("\n🎨 Creating visualizations...")

fig, axes = plt.subplots(3, 2, figsize=(16, 18))

ax1 = axes[0, 0]
ax1.scatter(y_train, y_train_pred, alpha=0.6, s=50, edgecolor='k', linewidth=0.5)
ax1.plot([y_train.min(), y_train.max()], 
         [y_train.min(), y_train.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Attendance Ratio', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted Attendance Ratio', fontsize=12, fontweight='bold')
ax1.set_title(f'Training Set: Predicted vs Actual\nR² = {train_r2:.4f}', 
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.scatter(y_test, y_test_pred, alpha=0.6, s=50, 
           color='coral', edgecolor='k', linewidth=0.5)
ax2.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Attendance Ratio', fontsize=12, fontweight='bold')
ax2.set_ylabel('Predicted Attendance Ratio', fontsize=12, fontweight='bold')
ax2.set_title(f'Test Set: Predicted vs Actual\nR² = {test_r2:.4f}', 
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
ax3.scatter(y_train_pred, train_residuals, alpha=0.6, s=50, edgecolor='k', linewidth=0.5)
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Predicted Attendance Ratio', fontsize=12, fontweight='bold')
ax3.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax3.set_title(f'Training Set: Residual Plot\nRMSE = {train_rmse:.4f}', 
              fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
ax4.scatter(y_test_pred, test_residuals, alpha=0.6, s=50, 
           color='coral', edgecolor='k', linewidth=0.5)
ax4.axhline(y=0, color='r', linestyle='--', lw=2)
ax4.set_xlabel('Predicted Attendance Ratio', fontsize=12, fontweight='bold')
ax4.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax4.set_title(f'Test Set: Residual Plot\nRMSE = {test_rmse:.4f}', 
              fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

ax5 = axes[2, 0]
ax5.hist(train_residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax5.axvline(x=0, color='r', linestyle='--', lw=2)
ax5.set_xlabel('Residuals', fontsize=12, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax5.set_title(f'Training Set: Distribution of Residuals\nMean = {train_residuals.mean():.4f}', 
              fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

ax6 = axes[2, 1]
ax6.hist(test_residuals, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
ax6.axvline(x=0, color='r', linestyle='--', lw=2)
ax6.set_xlabel('Residuals', fontsize=12, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax6.set_title(f'Test Set: Distribution of Residuals\nMean = {test_residuals.mean():.4f}', 
              fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
plt.close()
print("✓ Comprehensive analysis plot displayed!")

plt.figure(figsize=(12, 8))
plt.barh(range(len(feature_importance_df)), feature_importance_df['Importance'], 
         xerr=feature_importance_df['Std'], color='steelblue', alpha=0.8, edgecolor='black')
plt.yticks(range(len(feature_importance_df)), feature_importance_df['Feature'])
plt.xlabel('Permutation Importance', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Variant 1: Feature Importance Analysis', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()
print("✓ Feature importance plot displayed!")

print("\n" + "=" * 80)
print("✅ VARIANT 1 ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\n🎯 VARIANT 1 CROSS-VALIDATION RESULT:")
print(f"   5-Fold CV RMSE: {cv_rmse:.4f}")
print("=" * 80)