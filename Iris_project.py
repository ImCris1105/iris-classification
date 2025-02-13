import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Tạo DataFrame và kiểm tra dữ liệu
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Kiểm tra NaN và kiểu dữ liệu
print("NaN values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

# Trực quan hóa
sns.pairplot(
    data=df,
    hue='species',
    markers=['o', 's', 'D'],
    diag_kind='auto',
    plot_kws={'s': 20}
)
plt.show(block=True)

plt.figure(figsize=(8, 6))
sns.heatmap(
    df.drop('species', axis=1).corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f"
)
plt.title("Correlation Matrix")
plt.show(block=True)
# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện Logistic Regression
lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X_train_scaled, y_train)

# Huấn luyện SVM
svm = SVC(kernel='linear', C=1, random_state=42)
svm.fit(X_train_scaled, y_train)


# Hàm đánh giá
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=target_names)

    print(f"Accuracy: {accuracy:.2f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    return accuracy


# Đánh giá các mô hình
print("Logistic Regression:")
lr_accuracy = evaluate_model(lr, X_test_scaled, y_test)

print("\nSVM:")
svm_accuracy = evaluate_model(svm, X_test_scaled, y_test)

# Tối ưu SVM (optional)
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_svm = grid_search.best_estimator_
print("\nBest SVM Parameters:", grid_search.best_params_)
evaluate_model(best_svm, X_test_scaled, y_test)

# Dự đoán mẫu mới
new_sample = [[5.1, 3.5, 1.4, 0.2]]  # setosa
new_sample_scaled = scaler.transform(new_sample)
print("\nDự đoán mẫu mới:")
print("Logistic Regression →", target_names[lr.predict(new_sample_scaled)][0])
print("SVM →", target_names[svm.predict(new_sample_scaled)][0])