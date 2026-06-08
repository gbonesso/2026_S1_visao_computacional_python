import re
import pandas as pd

def convert_blink_format(blink_string):
    """
    Converts a string of blink annotations in the format '#start-end_type'
    to lines in the .tag file format, including frames with no blink information
    and ensuring sequential blink IDs.

    Args:
        blink_string: A string containing blink annotations.

    Returns:
        A dictionary where keys are frame numbers and values are lists of
        information for each frame in the .tag format.
    """
    frame_data = {}
    # Split the input string by '#' to get individual blink annotations
    blinks = blink_string.strip().split(' ')[1:]  # Ignore the first empty split

    blink_counter = 1 # Initialize blink counter for sequential IDs

    # First pass to populate blink information for frames within blink ranges
    for blink in blinks:
        match = re.match(r'(\d+)-(\d+)([ci])', blink)
        if match:
            start_frame = int(match.group(1))
            end_frame = int(match.group(2))
            blink_type = match.group(3)

            for frame_id in range(start_frame, end_frame + 1):
                # Construct the .tag line based on blink type
                # Assuming columns are: Frame_ID:Blink_ID:NFF:LE_FC:LE_NV:RE_FC:RE_NV:...
                if blink_type == 'c':  # Complete blink
                    # Assuming 'C' for closed, 'X' for not applicable/unknown
                    tag_line_parts = [str(frame_id), str(blink_counter), 'X', 'C', 'X', 'C', 'X'] # Taking first 7 columns as per previous code
                elif blink_type == 'i':  # Incomplete blink
                     # Assuming 'X' for not applicable/unknown
                    tag_line_parts = [str(frame_id), str(blink_counter), 'X', 'X', 'X', 'X', 'X'] # Taking first 7 columns as per previous code

                # Store data for each frame. Using a list to match the structure from the load_data function
                frame_data[frame_id] = tag_line_parts

            blink_counter += 1 # Increment blink counter for the next blink

        else:
            print(f"Warning: Could not parse blink annotation: {blink}")

    # Find the maximum frame ID from the blink annotations
    max_frame = 0
    for blink in blinks:
        match = re.match(r'(\d+)-(\d+)([ci])', blink)
        if match:
            max_frame = max(max_frame, int(match.group(2)))

    # Populate frames without blink information
    for frame_id in range(max_frame + 1): # Iterate up to and including the max_frame
        if frame_id not in frame_data:
            # Assign -1 to frames with no blink information, and 'X' to other columns
            frame_data[frame_id] = [str(frame_id), '-1', 'X', 'X', 'X', 'X', 'X'] # Taking first 7 columns

    # Sort the frames by Frame_ID
    sorted_frame_ids = sorted(frame_data.keys())

    # Generate the .tag lines in order
    tag_lines = [":".join(frame_data[frame_id]) for frame_id in sorted_frame_ids]


    return tag_lines

"""
A função load_data_unifesp carrega um arquivo de texto com marcações no formato definido pela Bárbara.
Ela lê o conteúdo do arquivo, converte as marcações usando a função convert_blink_format, 
e então processa as linhas resultantes para criar um DataFrame do pandas. 
O DataFrame contém colunas para Frame_ID, Blink_ID, NFF, LE_FC, LE_NV, RE_FC e RE_NV, onde Frame_ID e Blink_ID 
são convertidos para valores numéricos. A função também inclui tratamento de erros para lidar com problemas na 
leitura do arquivo.

# Example usage (replace with your actual file path):
# file_path_unifesp = '/content/path/to/your/unifesp_blink_file.txt' # Replace with your actual file path
# df_unifesp = load_data_unifesp(file_path_unifesp)

# if df_unifesp is not None:
#     print(df_unifesp.head())
"""
def load_data_unifesp(file_path):
    """
    Loads blink annotations from a text file, converts them using convert_blink_format,
    and creates a pandas DataFrame based on the .tag format rules.

    Args:
        file_path: The path to the text file containing blink annotations.

    Returns:
        A pandas DataFrame containing the converted blink data, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as f:
            blink_string = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

    print(f"Successfully read blink annotations from {file_path}")
    print(f"Blink annotations string: {blink_string[:100]}...")  # Print the first 100 characters for verification

    # Convert the blink annotations string to .tag format lines
    tag_lines = convert_blink_format(blink_string)

    print(f"Converted blink annotations to .tag format with {len(tag_lines)} lines.")
    print(f"First 5 lines of .tag format:\n{tag_lines[:5]}")  # Print the first 5 lines for verification

    # Process the .tag lines to create a DataFrame
    data = []
    for line in tag_lines:
        parts = line.strip().split(":", 7)  # Split by ":", limit to 7 columns (first 7)
        if len(parts) >= 7:  # Ensure we have at least 7 elements before appending
            data.append(parts[:7]) # Take only the first 7

    # Define columns based on the .tag format description
    df = pd.DataFrame(
        data,
        columns=['Frame_ID', 'Blink_ID', 'NFF', 'LE_FC', 'LE_NV', 'RE_FC', 'RE_NV'])

    # Convert Frame_ID and Blink_ID to numeric, coercing errors
    df['Frame_ID'] = pd.to_numeric(df['Frame_ID'], errors='coerce')
    df['Blink_ID'] = pd.to_numeric(df['Blink_ID'], errors='coerce')

    return df

def agrupa_marcacao(df):
    """
    Agrupa o DataFrame por Blink_ID, excluindo -1, e retorna um DataFrame
    com os quadros inicial e final de cada blink.

    Args:
        df: DataFrame de entrada com as colunas 'Frame_ID' e 'Blink_ID'.

    Returns:
        DataFrame com as colunas 'start_frame' e 'end_frame' agregadas por Blink_ID.
    """
    # Filter out blink_id == -1
    df_filtered = df[df['Blink_ID'] != -1].copy()

    # Aggregate data
    blink_summary = df_filtered.groupby('Blink_ID')['Frame_ID'].agg(['min', 'max'])
    blink_summary.columns = ['start_frame', 'end_frame']

    return blink_summary

def check_intersection_summary_blinks(blink_summary, df_blinks):
    """
    Checks for intersection between blinks in blink_summary and df_blinks based on frame ranges.

    Args:
        blink_summary: DataFrame with 'start_frame' and 'end_frame' for blinks from summary.
        df_blinks: DataFrame with 'Initial_Frame' and 'End_Frame' for blinks from df_blinks.

    Returns:
        DataFrame with intersection results, including original and intersecting frames,
        and 'no intersection' if no intersection is found.
    """
    blink_summary['start_frame'] = pd.to_numeric(blink_summary['start_frame'], errors='coerce')
    blink_summary['end_frame']   = pd.to_numeric(blink_summary['end_frame'], errors='coerce')
    df_blinks['start_frame']     = pd.to_numeric(df_blinks['start_frame'], errors='coerce')
    df_blinks['end_frame']       = pd.to_numeric(df_blinks['end_frame'], errors='coerce')

    results = []
    for index, row in blink_summary.iterrows():
        start_frame_summary = row['start_frame']
        end_frame_summary = row['end_frame']

        intersection_found = False
        for _, blink_row in df_blinks.iterrows():
            initial_frame = blink_row['start_frame']
            end_frame = blink_row['end_frame']

            '''
            print(type(start_frame_summary), start_frame_summary)
            print(type(end_frame_summary), end_frame_summary)
            print(type(initial_frame), initial_frame)
            print(type(end_frame), end_frame)
            '''

            if (start_frame_summary <= end_frame) and (end_frame_summary >= initial_frame):
                results.append({
                    'id_summary': index,
                    'start_frame': start_frame_summary,
                    'end_frame': end_frame_summary,
                    'target_start_frame': initial_frame,
                    'target_end_frame': end_frame
                })
                intersection_found = True
                break  # Consider just the first intersection

        if not intersection_found:
            results.append({
                'id_summary': index,
                'start_frame': start_frame_summary,
                'end_frame': end_frame_summary,
                'target_start_frame': "no intersection",
                'target_end_frame': "no intersection"
            })

    result_df = pd.DataFrame(results)
    return result_df

def agrupa_olhos_abertos(df):
    """
    Identifica períodos contínuos onde Blink_ID é -1 (olhos abertos)
    e atribui um ID único sequencial a cada período, começando por 1.

    Args:
        df: DataFrame de entrada com as colunas 'Frame_ID' e 'Blink_ID'.

    Returns:
        DataFrame com as colunas 'id', 'start_frame', e 'end_frame'
        para cada período de olhos abertos.
    """
    df_open = df[df['Blink_ID'] == -1].copy()

    if df_open.empty:
        return pd.DataFrame(columns=['id', 'start_frame', 'end_frame'])

    # Ensure 'Frame_ID' is numeric and sorted
    df_open['Frame_ID'] = pd.to_numeric(df_open['Frame_ID'])
    df_open = df_open.sort_values(by='Frame_ID')


    # Calculate the difference between consecutive frame_ids
    df_open['frame_diff'] = df_open['Frame_ID'].diff().fillna(1)

    # Identify the start of new open eye periods
    # A new period starts when the frame_diff is greater than 1
    df_open['new_period'] = (df_open['frame_diff'] > 1).cumsum()

    # Group by the new period and get the min and max frame_id
    open_eye_periods = df_open.groupby('new_period')['Frame_ID'].agg(['min', 'max']).reset_index()
    open_eye_periods.columns = ['id', 'start_frame', 'end_frame']

    # Assign sequential IDs starting from 1
    open_eye_periods['id'] = open_eye_periods.index + 1

    return open_eye_periods

def calculate_confusion_matrix(df_intersection_tag_to_app, df_intersection_app_to_tag, df_intersection_open_eyes_tag_to_app):
    """
    Calculates the confusion matrix based on intersection results from dataframes.

    Args:
        df_intersection_tag_to_app: DataFrame containing intersections from tag to app.
                                    True Positives are rows with intersection.
                                    False Negatives are rows with 'no intersection'.
        df_intersection_app_to_tag: DataFrame containing intersections from app to tag.
                                    False Positives are rows with 'no intersection'.
        df_intersection_open_eyes_tag_to_app: DataFrame containing intersections of open eyes (tag) to app blinks.
                                             True Negatives are rows with 'no intersection'.

    Returns:
        A dictionary or a confusion matrix object containing TP, FN, FP, TN counts.
    """
    # True Positives: Intersections in df_intersection_tag_to_app
    true_positives = df_intersection_tag_to_app[
        df_intersection_tag_to_app['target_start_frame'] != 'no intersection'
    ].shape[0]

    # False Negatives: No intersections in df_intersection_tag_to_app
    false_negatives = df_intersection_tag_to_app[
        df_intersection_tag_to_app['target_start_frame'] == 'no intersection'
    ].shape[0]

    # False Positives: No intersections in df_intersection_app_to_tag
    false_positives = df_intersection_app_to_tag[
        df_intersection_app_to_tag['target_start_frame'] == 'no intersection'
    ].shape[0]

    # True Negatives: No intersections in df_intersection_open_eyes_tag_to_app
    true_negatives = df_intersection_open_eyes_tag_to_app[
         df_intersection_open_eyes_tag_to_app['target_start_frame'] == 'no intersection'
    ].shape[0]


    # You can return these counts as a dictionary or use sklearn's confusion_matrix
    # To use sklearn's confusion_matrix, you typically need true labels and predicted labels.
    # Given the structure, returning counts might be more direct based on the request.
    confusion_matrix_counts = {
        'True Positives (TP)': true_positives,
        'False Negatives (FN)': false_negatives,
        'False Positives (FP)': false_positives,
        'True Negatives (TN)': true_negatives
    }

    # Alternatively, to use sklearn.metrics.confusion_matrix, you would need to
    # construct arrays of true and predicted labels based on the intersection logic.
    # This would be more complex and might not directly map to the user's definition
    # based on the three separate dataframes.

    return confusion_matrix_counts

def calculate_frame_based_confusion_matrix(df_intersection_tag_to_app, df_intersection_app_to_tag, df_intersection_open_eyes_tag_to_app):
    """
    Calculates the confusion matrix based on summing the duration of frames
    within intersecting or non-intersecting periods of the intersection dataframes,
    according to the user's definitions.

    Args:
        df_intersection_tag_to_app: DataFrame containing intersections from tag to app.
                                    Used for TP and FN based on 'target_start_frame'.
        df_intersection_app_to_tag: DataFrame containing intersections from app to tag.
                                    Used for FP based on 'target_start_frame'.
        df_intersection_open_eyes_tag_to_app: DataFrame containing intersections of open eyes (tag) to app blinks.
                                             Used for TN based on 'target_start_frame'.


    Returns:
        A dictionary containing TP, FN, FP, TN counts based on frames.
    """
    # Calculate True Positives: Sum of frame durations for rows in df_intersection_tag_to_app with intersection
    df_tp_periods = df_intersection_tag_to_app[
        df_intersection_tag_to_app['target_start_frame'] != 'no intersection'
    ].copy()
    df_tp_periods['duration'] = df_tp_periods['end_frame'] - df_tp_periods['start_frame'] + 1
    true_positives = df_tp_periods['duration'].sum()

    # Calculate False Negatives: Sum of frame durations for rows in df_intersection_tag_to_app with 'no intersection'
    df_fn_periods = df_intersection_tag_to_app[
        df_intersection_tag_to_app['target_start_frame'] == 'no intersection'
    ].copy()
    df_fn_periods['duration'] = df_fn_periods['end_frame'] - df_fn_periods['start_frame'] + 1
    false_negatives = df_fn_periods['duration'].sum()


    # Calculate False Positives: Sum of frame durations for rows in df_intersection_app_to_tag with 'no intersection'
    df_fp_periods = df_intersection_app_to_tag[
        df_intersection_app_to_tag['target_start_frame'] == 'no intersection'
    ].copy()
    df_fp_periods['duration'] = df_fp_periods['end_frame'] - df_fp_periods['start_frame'] + 1
    false_positives = df_fp_periods['duration'].sum()


    # Calculate True Negatives: Sum of frame durations for rows in df_intersection_open_eyes_tag_to_app with 'no intersection'
    df_tn_periods = df_intersection_open_eyes_tag_to_app[
         df_intersection_open_eyes_tag_to_app['target_start_frame'] == 'no intersection'
    ].copy()
    df_tn_periods['duration'] = df_tn_periods['end_frame'] - df_tn_periods['start_frame'] + 1
    true_negatives = df_tn_periods['duration'].sum()


    confusion_matrix_counts = {
        'True Positives (TP)': int(true_positives),
        'False Negatives (FN)': int(false_negatives),
        'False Positives (FP)': int(false_positives),
        'True Negatives (TN)': int(true_negatives)
    }

    return confusion_matrix_counts

def calculate_frame_based_confusion_matrix_by_file_number(df_intersection_tag_to_app, df_intersection_app_to_tag, df_intersection_open_eyes_tag_to_app):
    """
    Calculates the confusion matrix based on summing the duration of frames
    within intersecting or non-intersecting periods of the intersection dataframes,
    according to the user's definitions, providing a row for each file number.

    Args:
        df_intersection_tag_to_app: DataFrame containing intersections from tag to app.
                                    Used for TP and FN based on 'target_start_frame'.
                                    Should include 'file_number' column.
        df_intersection_app_to_tag: DataFrame containing intersections from app to tag.
                                    Used for FP based on 'target_start_frame'.
                                    Should include 'file_number' column.
        df_intersection_open_eyes_tag_to_app: DataFrame containing intersections of open eyes (tag) to app blinks.
                                             Used for TN based on 'target_start_frame'.
                                             Should include 'file_number' column.


    Returns:
        A DataFrame containing TP, FN, FP, TN counts based on frames for each file_number.
    """
    results = []

    # Get all unique file numbers from the input dataframes
    all_file_numbers = pd.concat([
        df_intersection_tag_to_app['file_number'],
        df_intersection_app_to_tag['file_number'],
        df_intersection_open_eyes_tag_to_app['file_number']
    ]).unique()

    for file_number in all_file_numbers:
        # Filter dataframes for the current file number
        df_tag_to_app_file = df_intersection_tag_to_app[
            df_intersection_tag_to_app['file_number'] == file_number
        ].copy()
        df_app_to_tag_file = df_intersection_app_to_tag[
            df_intersection_app_to_tag['file_number'] == file_number
        ].copy()
        df_open_eyes_file = df_intersection_open_eyes_tag_to_app[
            df_intersection_open_eyes_tag_to_app['file_number'] == file_number
        ].copy()

        # Calculate True Positives for the current file: Sum of frame durations for rows in df_tag_to_app_file with intersection
        df_tp_periods_file = df_tag_to_app_file[
            df_tag_to_app_file['target_start_frame'] != 'no intersection'
        ].copy()
        if not df_tp_periods_file.empty:
            df_tp_periods_file['duration'] = df_tp_periods_file['end_frame'] - df_tp_periods_file['start_frame'] + 1
            true_positives = df_tp_periods_file['duration'].sum()
        else:
            true_positives = 0


        # Calculate False Negatives for the current file: Sum of frame durations for rows in df_tag_to_app_file with 'no intersection'
        df_fn_periods_file = df_tag_to_app_file[
            df_tag_to_app_file['target_start_frame'] == 'no intersection'
        ].copy()
        if not df_fn_periods_file.empty:
            df_fn_periods_file['duration'] = df_fn_periods_file['end_frame'] - df_fn_periods_file['start_frame'] + 1
            false_negatives = df_fn_periods_file['duration'].sum()
        else:
            false_negatives = 0


        # Calculate False Positives for the current file: Sum of frame durations for rows in df_app_to_tag_file with 'no intersection'
        df_fp_periods_file = df_app_to_tag_file[
            df_app_to_tag_file['target_start_frame'] == 'no intersection'
        ].copy()
        if not df_fp_periods_file.empty:
            df_fp_periods_file['duration'] = df_fp_periods_file['end_frame'] - df_fp_periods_file['start_frame'] + 1
            false_positives = df_fp_periods_file['duration'].sum()
        else:
            false_positives = 0


        # Calculate True Negatives for the current file: Sum of frame durations for rows in df_open_eyes_file with 'no intersection'
        df_tn_periods_file = df_open_eyes_file[
             df_open_eyes_file['target_start_frame'] == 'no intersection'
        ].copy()
        if not df_tn_periods_file.empty:
            df_tn_periods_file['duration'] = df_tn_periods_file['end_frame'] - df_tn_periods_file['start_frame'] + 1
            true_negatives = df_tn_periods_file['duration'].sum()
        else:
            true_negatives = 0

        results.append({
            'file_number': file_number,
            'True Positives (TP)': int(true_positives),
            'False Negatives (FN)': int(false_negatives),
            'False Positives (FP)': int(false_positives),
            'True Negatives (TN)': int(true_negatives)
        })

    return pd.DataFrame(results)
