# Iris Flower Classification Project 🌸

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview
Classification of Iris flower species (setosa, versicolor, virginica) based on petal/sepal measurements using Scikit-learn.

## Dataset
- **Source:** Built-in Iris Dataset from Scikit-learn
- **Samples:** 150
- **Features:**  
  `sepal length (cm)`, `sepal width (cm)`, `petal length (cm)`, `petal width (cm)`
- **Target Classes:** 3 species (`setosa`, `versicolor`, `virginica`)

## Installation
```bash
git clone https://github.com/NguyenBui/iris-classification.git
cd iris-classification
pip install -r requirements.txt
```

## Usage
Run the classification script:
```bash
python iris.py
```

## Code Example
```python
# Sample prediction
new_sample = [[5.1, 3.5, 1.4, 0.2]]  # setosa
new_sample_scaled = scaler.transform(new_sample)
print("SVM prediction:", target_names[svm.predict(new_sample_scaled)[0]])
```

## Results
| Model               | Accuracy | Confusion Matrix                          |
|---------------------|----------|-------------------------------------------|
| Logistic Regression | 97.5%    | [[10  0   0]<br> [ 0  9   1]<br> [ 0  0 10]] |
| SVM                 | 100%     | [[10  0   0]<br> [ 0 10   0]<br> [ 0  0 10]] |

## Author
**Nguyen Bui**  
📧 [Email](mailto:buidangnguyen01012005@gmail.com)  
💻 [GitHub](https://github.com/NguyenBui)  
🔗 [LinkedIn](https://linkedin.com/in/nguyenbui) *(optional)*