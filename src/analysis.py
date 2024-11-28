import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy import stats

class DerivativeAnalysis:
    @staticmethod
    def compute_gradient(image):
        # Convert image to grayscale if it's not already
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute the gradient of the image (numerical derivative)
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
        return gradient_magnitude

def extract_image_features(folder):
    data = []
    for file in os.listdir(folder):
        if file.endswith('_thermal.jpg'):
            thermal_path = os.path.join(folder, file)
            rgb_path = os.path.join(folder, file.replace('_thermal', '_rgb'))
            if os.path.exists(rgb_path):
                thermal_img = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
                rgb_img = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
                # Compute gradient magnitudes for both images
                rgb_gradient_magnitude = DerivativeAnalysis.compute_gradient(rgb_img)
                thermal_gradient_magnitude = DerivativeAnalysis.compute_gradient(thermal_img)
                data.append({'ID': file.split('_')[0], 'Class': 'Thermal',
                             'Mean': thermal_img.mean(), 'Variance': thermal_img.var(),
                             'Gradient_Mean': rgb_gradient_magnitude.mean()})
                data.append({'ID': file.split('_')[0], 'Class': 'RGB',
                             'Mean': rgb_img.mean(), 'Variance': rgb_img.var(),
                             'Gradient_Mean': thermal_gradient_magnitude.mean()})
    return pd.DataFrame(data)

def fisher_discriminant_ratio(df):
    thermal_data = df[df['Class'] == 'Thermal']
    rgb_data = df[df['Class'] == 'RGB']
    return ((thermal_data['Mean'].mean() - rgb_data['Mean'].mean()) ** 2) / (
        thermal_data['Variance'].var() + rgb_data['Variance'].var()
    )

def auc_score(y_test, y_pred_prob):
    auc = roc_auc_score(y_test, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    return auc

def hypothesis_test(df):
    t_stat, p_val = stats.ttest_ind(
        df[df['Class'] == 'Thermal']['Mean'], 
        df[df['Class'] == 'RGB']['Mean']
    )
    return t_stat, p_val

def main_analysis(folder):
    df = extract_image_features(folder)
    df.dropna(inplace=True)
    for col in ['Mean', 'Variance', 'Gradient_Mean']:
        df[col] = df.groupby('Class')[col].transform(lambda x: (x - x.mean()) / x.std())

    le = LabelEncoder()
    df['Class_Label'] = le.fit_transform(df['Class'])
    X = df[['Mean', 'Variance', 'Gradient_Mean']].values
    y = df['Class_Label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    fdr = fisher_discriminant_ratio(df)
    auc = auc_score(y_test, y_pred_prob)
    t_stat, p_val = hypothesis_test(df)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return {
        'Fisher Discriminant Ratio': fdr,
        'AUC Score': auc,
        'T-Statistic': t_stat,
        'P-Value': p_val,
        'Accuracy': accuracy,
        'Confusion Matrix': conf_matrix
    }

if __name__ == '__main__':
    folder = '/content/drive/MyDrive/Programacion Cientifica/Database'  
    results = main_analysis(folder)
    for key, value in results.items():
        print(f'{key}: {value}')
