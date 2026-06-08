import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from blink_utils import aggregate_bilateral_blinks_new, analyze_csv_blinks
from comparacoes import agrupa_marcacao, agrupa_olhos_abertos, calculate_confusion_matrix, calculate_frame_based_confusion_matrix, check_intersection_summary_blinks, load_data_unifesp

# Define the directory containing the files
directory_path = './projeto/videos/'

# Initialize empty dataframes to accumulate results
all_df_intersection_tag_to_app = pd.DataFrame()
all_df_intersection_app_to_tag = pd.DataFrame()
all_df_intersection_open_eyes_tag_to_app = pd.DataFrame()

# Loop through all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith("marcacoes.txt"):
        file_path_unifesp = os.path.join(directory_path, filename)
        print(f"Processing file: {file_path_unifesp}")

        df = load_data_unifesp(file_path_unifesp)

        df_agregado = agrupa_marcacao(df)

        #df_blinks = load_excel_to_dataframe(file_path_unifesp.replace('marcacoes.txt', 'blinks.xlsx'))

        # Algoritmo usando somente probabilidade de olhos abertos
        df_blinks = analyze_csv_blinks(file_path_unifesp.replace('marcacoes.txt', 'output_abertura.csv'))

        # Algoritmo usando probabilidade de olhos abertos e EAR
        #df_blinks = analyze_excel_blinks_EAR(file_path_unifesp.replace('marcacoes.txt', 'analise.xlsx'))

        df_bilateral = aggregate_bilateral_blinks_new(df_blinks)
        #df_bilateral = aggregate_bilateral_blinks(df_blinks)

        df_intersection_tag_to_app = check_intersection_summary_blinks(df_agregado, df_bilateral)
        df_intersection_app_to_tag = check_intersection_summary_blinks(df_bilateral, df_agregado)
        df_agregado_olhos_abertos = agrupa_olhos_abertos(df)
        df_intersection_open_eyes_tag_to_app = check_intersection_summary_blinks(df_agregado_olhos_abertos, df_bilateral)


        # Accumulate the results
        all_df_intersection_tag_to_app = pd.concat([all_df_intersection_tag_to_app, df_intersection_tag_to_app], ignore_index=True)
        all_df_intersection_app_to_tag = pd.concat([all_df_intersection_app_to_tag, df_intersection_app_to_tag], ignore_index=True)
        all_df_intersection_open_eyes_tag_to_app = pd.concat([all_df_intersection_open_eyes_tag_to_app, df_intersection_open_eyes_tag_to_app], ignore_index=True)


# Now you have the accumulated dataframes:
# all_df_intersection_tag_to_app
# all_df_intersection_app_to_tag
# all_df_intersection_open_eyes_tag_to_app

print("\n--- Accumulated Results ---")
print("Intersection Tag to App:")
print(all_df_intersection_tag_to_app.head())
print("\nIntersection App to Tag:")
print(all_df_intersection_app_to_tag.head())
print("\nIntersection Open Eyes Tag to App:")
print(all_df_intersection_open_eyes_tag_to_app.head())

# You can now use these accumulated dataframes for further analysis (e.g., calculating overall confusion matrix)

# Calcula a matriz de confusão com base em quantidade de piscadas
"""
confusion_matrix_results = calculate_confusion_matrix(
    all_df_intersection_tag_to_app,
    all_df_intersection_app_to_tag,
    all_df_intersection_open_eyes_tag_to_app
)
print(confusion_matrix_results)
"""

# Calcula a matriz de confusão com base em quantidade de frames
confusion_matrix_results = calculate_frame_based_confusion_matrix(
    all_df_intersection_tag_to_app,
    all_df_intersection_app_to_tag,
    all_df_intersection_open_eyes_tag_to_app
)
print(confusion_matrix_results)

# Get the counts from the confusion matrix results
tp = confusion_matrix_results['True Positives (TP)']
fn = confusion_matrix_results['False Negatives (FN)']
fp = confusion_matrix_results['False Positives (FP)']
tn = confusion_matrix_results['True Negatives (TN)']

# Calculate metrics
accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
# F1-Score is the harmonic mean of Precision and Recall
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
# Specificity or True Negative Rate
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
# False Positive Rate
false_positive_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
# False Negative Rate
false_negative_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
# True Positive Rate (same as Recall)
true_positive_rate = recall

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity/True Positive Rate): {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")
print(f"Specificity (True Negative Rate): {specificity:.4f}")
print(f"False Positive Rate: {false_positive_rate:.4f}")
print(f"False Negative Rate: {false_negative_rate:.4f}")

# Get the confusion matrix counts from the results
tp = confusion_matrix_results['True Positives (TP)']
fn = confusion_matrix_results['False Negatives (FN)']
fp = confusion_matrix_results['False Positives (FP)']
tn = confusion_matrix_results['True Negatives (TN)']

# Create the confusion matrix array
# The order is typically [[TN, FP], [FN, TP]] for binary classification
confusion_matrix_array = np.array([[tn, fp],
                                   [fn, tp]])

# Create labels for the heatmap with counts
labels = np.array([
    [f'True Negatives\n{tn}', f'False Positives\n{fp}'],
    [f'False Negatives\n{fn}', f'True Positives\n{tp}']
])

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_array, annot=labels, fmt='s', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])

plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()