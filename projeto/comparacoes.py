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