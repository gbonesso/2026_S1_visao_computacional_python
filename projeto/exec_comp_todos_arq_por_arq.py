import os
import pandas as pd
import re
from pathlib import Path

from comparacoes import agrupa_marcacao, agrupa_olhos_abertos, calculate_frame_based_confusion_matrix_by_file_number_v2, check_intersection_summary_blinks, load_data_unifesp
from blink_utils import aggregate_bilateral_blinks_new, analyze_csv_blinks

def processar_diretorio(directory_path, salvar_arquivos_intermediarios=False):
    directory_path = Path(directory_path)

    if not directory_path.is_dir():
        raise NotADirectoryError(f"Diretório inválido: {directory_path}")

    all_df_intersection_tag_to_app = pd.DataFrame()
    all_df_intersection_app_to_tag = pd.DataFrame()
    all_df_intersection_open_eyes_tag_to_app = pd.DataFrame()
    all_df_matriz_confusao = pd.DataFrame()

    for filename in os.listdir(directory_path):
        if filename.endswith("marcacoes.txt"):
            file_path_unifesp = directory_path / filename
            numero = re.search(r'^(\d+)_', file_path_unifesp.name)
            file_number = numero.group(1)
            print(f"Processing file #{file_number}: {file_path_unifesp}")

            df = load_data_unifesp(str(file_path_unifesp))
            df_matriz_confusao = df.copy()  # Create a copy for confusion matrix calculations
            df_agregado = agrupa_marcacao(df)

            df_blinks = analyze_csv_blinks(
                str(file_path_unifesp.with_name(file_path_unifesp.name.replace('marcacoes.txt', 'output_abertura.csv')))
            )

            df_bilateral = aggregate_bilateral_blinks_new(df_blinks)

            # Acrescente uma coluna blink_detected em df_matriz_confusao para indicar se houve detecção de blink
            df_matriz_confusao['blink_detected'] = df_matriz_confusao.apply(
                lambda row: any(
                    (row['Frame_ID'] >= blink['start_frame'] and row['Frame_ID'] <= blink['end_frame'])
                    for _, blink in df_bilateral.iterrows()
                ),
                axis=1
            )

            # Acrescente uma coluna resultado em df_matriz_confusao marcando como 'TP' caso
            # blink_detected seja true e o Blink_ID seja diferente de -1. Marque como 'FN'caso Blink_ID 
            # seja diferente de -1 e blink_detected seja false. Marque como 'FP' caso Blink_ID seja -1 e 
            # blink_detected seja true. Marque como 'TN' caso Blink_ID seja -1 e blink_detected seja false.
            def classify_row(row):
                if row['blink_detected'] and row['Blink_ID'] != -1:
                    return 'TP'
                elif not row['blink_detected'] and row['Blink_ID'] != -1:
                    return 'FN'
                elif row['blink_detected'] and row['Blink_ID'] == -1:
                    return 'FP'
                elif not row['blink_detected'] and row['Blink_ID'] == -1:
                    return 'TN'
                else:
                    return 'Unknown'
                
            df_matriz_confusao['resultado'] = df_matriz_confusao.apply(classify_row, axis=1)

            df_intersection_tag_to_app = check_intersection_summary_blinks(df_agregado, df_bilateral)
            df_intersection_app_to_tag = check_intersection_summary_blinks(df_bilateral, df_agregado)
            df_agregado_olhos_abertos = agrupa_olhos_abertos(df)
            df_intersection_open_eyes_tag_to_app = check_intersection_summary_blinks(df_agregado_olhos_abertos, df_bilateral)

            df_intersection_tag_to_app['file_number'] = file_number
            df_intersection_app_to_tag['file_number'] = file_number
            df_intersection_open_eyes_tag_to_app['file_number'] = file_number
            df_matriz_confusao['file_number'] = file_number

            if salvar_arquivos_intermediarios:
                df_agregado.to_excel(directory_path / f"{file_number}_df_agregado.xlsx", index=False)
                df_blinks.to_excel(directory_path / f"{file_number}_df_blinks.xlsx", index=False)
                df_bilateral.to_excel(directory_path / f"{file_number}_df_bilateral.xlsx", index=False)
                df_intersection_tag_to_app.to_excel(directory_path / f"{file_number}_df_intersection_tag_to_app.xlsx", index=False)
                df_intersection_app_to_tag.to_excel(directory_path / f"{file_number}_df_intersection_app_to_tag.xlsx", index=False)
                df_agregado_olhos_abertos.to_excel(directory_path / f"{file_number}_df_agregado_olhos_abertos.xlsx", index=False)
                df_intersection_open_eyes_tag_to_app.to_excel(
                    directory_path / f"{file_number}_df_intersection_open_eyes_tag_to_app.xlsx",
                    index=False,
                )
                df_matriz_confusao.to_excel(directory_path / f"{file_number}_df_matriz_confusao.xlsx", index=False)

            all_df_intersection_tag_to_app = pd.concat([all_df_intersection_tag_to_app, df_intersection_tag_to_app], ignore_index=True)
            all_df_intersection_app_to_tag = pd.concat([all_df_intersection_app_to_tag, df_intersection_app_to_tag], ignore_index=True)
            all_df_intersection_open_eyes_tag_to_app = pd.concat([all_df_intersection_open_eyes_tag_to_app, df_intersection_open_eyes_tag_to_app], ignore_index=True)
            all_df_matriz_confusao = pd.concat([all_df_matriz_confusao, df_matriz_confusao], ignore_index=True)

    print("\n--- Accumulated Results ---")
    print("Intersection Tag to App:")
    print(all_df_intersection_tag_to_app.head())
    print("\nIntersection App to Tag:")
    print(all_df_intersection_app_to_tag.head())
    print("\nIntersection Open Eyes Tag to App:")
    print(all_df_intersection_open_eyes_tag_to_app.head())
    print("\nConfusion Matrix Results:")
    print(all_df_matriz_confusao.head())

    confusion_matrix_results = calculate_frame_based_confusion_matrix_by_file_number_v2(
        all_df_matriz_confusao,
    )
    print(confusion_matrix_results)

    metrics_list = []

    for index, row in confusion_matrix_results.iterrows():
        file_number = row['file_number']
        tp = row['True Positives (TP)']
        fn = row['False Negatives (FN)']
        fp = row['False Positives (FP)']
        tn = row['True Negatives (TN)']

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_positive_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
        false_negative_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
        true_positive_rate = recall

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
            'True Positive Rate': true_positive_rate,
        })

    metrics_df = pd.DataFrame(metrics_list)
    print(metrics_df.head(100))

    metrics_output_path = directory_path / 'metrics_df.xlsx'
    try:
        metrics_df.to_excel(metrics_output_path, index=False)
        print(f"Metrics DataFrame successfully exported to {metrics_output_path}")
    except Exception as e:
        print(f"An error occurred while exporting metrics_df: {e}")

    return {
        'all_df_intersection_tag_to_app': all_df_intersection_tag_to_app,
        'all_df_intersection_app_to_tag': all_df_intersection_app_to_tag,
        'all_df_intersection_open_eyes_tag_to_app': all_df_intersection_open_eyes_tag_to_app,
        'all_df_matriz_confusao': all_df_matriz_confusao,
        'confusion_matrix_results': confusion_matrix_results,
        'metrics_df': metrics_df,
        'metrics_output_path': metrics_output_path,
    }


if __name__ == '__main__':
    processar_diretorio(
        './projeto/videos_output_com_rotacao_com_med_ang_3/', 
        salvar_arquivos_intermediarios=True,
    )

