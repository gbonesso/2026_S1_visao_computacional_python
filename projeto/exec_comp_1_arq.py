from comparacoes import agrupa_olhos_abertos, check_intersection_summary_blinks, load_data_unifesp, agrupa_marcacao
from blink_utils import aggregate_bilateral_blinks_new, analyze_csv_blinks

file_path_unifesp = './projeto/videos/14_marcacoes.txt' # Replace with your actual file path
df = load_data_unifesp(file_path_unifesp)

if df is None:
    print(f"Error: Unable to load data from {file_path_unifesp}")
else:
    print(df.head())

    df_agregado = agrupa_marcacao(df)

    print(df_agregado.head(50))

    #df_blinks = load_excel_to_dataframe(file_path_unifesp.replace('marcacoes.txt', 'blinks.xlsx'))
    #df_blinks = analyze_excel_blinks(file_path_unifesp.replace('marcacoes.txt', 'analise.xlsx'))
    df_blinks = analyze_csv_blinks(file_path_unifesp.replace('marcacoes.txt', 'output_abertura.csv'))

    #print(df_blinks.head())
    print(df_blinks.head(50))

    #df_bilateral = aggregate_bilateral_blinks(df_blinks)
    df_bilateral = aggregate_bilateral_blinks_new(df_blinks)

    print(df_bilateral.head(50))

    df_intersection_tag_to_app = check_intersection_summary_blinks(df_agregado, df_bilateral)

    print(df_intersection_tag_to_app.head())

    # Export the DataFrame to an Excel file
    try:
      df_intersection_tag_to_app.to_excel(file_path_unifesp.replace('marcacoes.txt', 'intersection_tag_to_app.xlsx'), index=False)
      print("DataFrame successfully exported to intersection_tag_to_app.xlsx")
    except Exception as e:
      print(f"An error occurred while exporting the DataFrame: {e}")

    df_intersection_app_to_tag = check_intersection_summary_blinks(df_bilateral, df_agregado)

    print(df_intersection_app_to_tag.head())

    # Export the DataFrame to an Excel file
    try:
      df_intersection_app_to_tag.to_excel(file_path_unifesp.replace('marcacoes.txt', 'intersection_app_to_tag.xlsx'), index=False)
      print("DataFrame successfully exported to intersection_app_to_tag.xlsx")
    except Exception as e:
      print(f"An error occurred while exporting the DataFrame: {e}")

    df_agregado_olhos_abertos = agrupa_olhos_abertos(df)

    print('df_agregado_olhos_abertos')
    print(df_agregado_olhos_abertos.head())

    df_intersection_open_eyes_tag_to_app = check_intersection_summary_blinks(df_agregado_olhos_abertos, df_bilateral)

    print(df_intersection_open_eyes_tag_to_app.head())

    # Export the DataFrame to an Excel file
    try:
      df_intersection_open_eyes_tag_to_app.to_excel(file_path_unifesp.replace('marcacoes.txt', 'intersection_open_eyes_tag_to_app.xlsx'), index=False)
      print("DataFrame successfully exported to intersection_open_eyes_tag_to_app.xlsx")
    except Exception as e:
      print(f"An error occurred while exporting the DataFrame: {e}")