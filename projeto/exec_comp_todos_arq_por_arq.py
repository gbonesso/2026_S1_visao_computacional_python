import os
import pandas as pd
import re

from comparacoes import agrupa_marcacao, agrupa_olhos_abertos, calculate_frame_based_confusion_matrix_by_file_number, check_intersection_summary_blinks, load_data_unifesp
from blink_utils import aggregate_bilateral_blinks_new, analyze_csv_blinks

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
        numero = re.search(r'/(\d+)_', file_path_unifesp)
        file_number = numero.group(1)
        print(f"Processing file #{file_number}: {file_path_unifesp}")

        df = load_data_unifesp(file_path_unifesp)

        df_agregado = agrupa_marcacao(df)

        #df_blinks = load_excel_to_dataframe(file_path_unifesp.replace('marcacoes.txt', 'blinks.xlsx'))
        #df_blinks = analyze_excel_blinks(file_path_unifesp.replace('marcacoes.txt', 'analise.xlsx'))
        df_blinks = analyze_csv_blinks(file_path_unifesp.replace('marcacoes.txt', 'output_abertura.csv'))

        df_bilateral = aggregate_bilateral_blinks_new(df_blinks)
        #df_bilateral = aggregate_bilateral_blinks(df_blinks)

        df_intersection_tag_to_app = check_intersection_summary_blinks(df_agregado, df_bilateral)
        df_intersection_app_to_tag = check_intersection_summary_blinks(df_bilateral, df_agregado)
        df_agregado_olhos_abertos = agrupa_olhos_abertos(df)
        df_intersection_open_eyes_tag_to_app = check_intersection_summary_blinks(df_agregado_olhos_abertos, df_bilateral)

        df_intersection_tag_to_app['file_number'] = file_number
        df_intersection_app_to_tag['file_number'] = file_number
        df_intersection_open_eyes_tag_to_app['file_number'] = file_number

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

confusion_matrix_results = calculate_frame_based_confusion_matrix_by_file_number(
    all_df_intersection_tag_to_app,
    all_df_intersection_app_to_tag,
    all_df_intersection_open_eyes_tag_to_app
)
print(confusion_matrix_results)

# Initialize an empty list to store the results for each file
metrics_list = []

# Iterate through each row of the confusion matrix results dataframe
for index, row in confusion_matrix_results.iterrows():
    file_number = row['file_number']
    tp = row['True Positives (TP)']
    fn = row['False Negatives (FN)']
    fp = row['False Positives (FP)']
    tn = row['True Negatives (TN)']

    # Calculate metrics for the current file
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_positive_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
    false_negative_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
    true_positive_rate = recall

    # Append the results to the list
    metrics_list.append({
        'file_number': file_number,
        'tp': tp,
        'fn': fn,
        'fp': fp,
        'tn': tn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'Specificity': specificity,
        'False Positive Rate': false_positive_rate,
        'False Negative Rate': false_negative_rate,
        'True Positive Rate': true_positive_rate
    })

# Create a new DataFrame from the list of metrics
metrics_df = pd.DataFrame(metrics_list)

# Display the resulting DataFrame
print(metrics_df.head(100))

# Export the metrics dataframe to Excel
metrics_output_path = './projeto/videos/metrics_df.xlsx'
try:
    metrics_df.to_excel(metrics_output_path, index=False)
    print(f"Metrics DataFrame successfully exported to {metrics_output_path}")
except Exception as e:
    print(f"An error occurred while exporting metrics_df: {e}")

