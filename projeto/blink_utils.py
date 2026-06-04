import pandas as pd
from enum import Enum
from typing import Dict
import logging
from typing import List

class Eye(Enum):
    LEFT = "Eye.left"
    RIGHT = "Eye.right"

class BlinkType(Enum):
    INCOMPLETE = "BlinkType.incomplete"
    COMPLETE = "BlinkType.complete"

class BlinkEntity:
    def __init__(self, eye: Eye, blink_type: BlinkType, blink_sequence: int, first_frame: int, last_frame: int):
        self.eye = eye
        self.blink_type = blink_type
        self.blink_sequence = blink_sequence
        self.first_frame = first_frame
        self.last_frame = last_frame

    def to_json(self) -> Dict[str, any]:
        return {
            'eye': self.eye.value,
            'blink_type': self.blink_type.value,
            'id': self.blink_sequence,
            'start_frame': self.first_frame,
            'end_frame': self.last_frame
        }

    @classmethod
    def from_json(cls, json_data: Dict[str, any]):
        eye = Eye.LEFT if json_data['eye'] == Eye.LEFT.value else Eye.RIGHT
        blink_type = BlinkType.INCOMPLETE if json_data['blink_type'] == BlinkType.INCOMPLETE.value else BlinkType.COMPLETE
        return cls(
            eye=eye,
            blink_type=blink_type,
            blink_sequence=json_data['id'],
            first_frame=json_data['start_frame'],
            last_frame=json_data['end_frame']
        )
    
class FrameEntity:
    def __init__(self, frame_number: int, left_eye_openness: float, right_eye_openness: float):
        self.frame_number = frame_number
        self.left_eye_openness = left_eye_openness
        self.right_eye_openness = right_eye_openness

#from blink_entity import BlinkEntity, Eye, BlinkType
log = logging.getLogger('BlinkLogger')

class BlinkUtils:
    @staticmethod
    def analyse_blinks(frame_list: List[FrameEntity]) -> List[BlinkEntity]:
        both_eyes_blink_list = []
        left_eye_blink_list = BlinkUtils._analyse_blinks_for_one_eye(frame_list, Eye.LEFT)
        right_eye_blink_list = BlinkUtils._analyse_blinks_for_one_eye(frame_list, Eye.RIGHT)
        both_eyes_blink_list.extend(left_eye_blink_list)
        both_eyes_blink_list.extend(right_eye_blink_list)
        return both_eyes_blink_list

    @staticmethod
    def _get_smoothed_openness(frame_list: List[FrameEntity], which_eye: Eye, frame_index: int) -> float:
        window_size = 3
        half_window_size = window_size // 2
        first_frame = frame_index - half_window_size
        last_frame = frame_index + half_window_size
        total_frames = len(frame_list)
        sum_openness = 0.0
        frames_used = 0
        for i in range(first_frame, last_frame + 1):
            if 0 <= i < total_frames:
                eye_openness = frame_list[i].left_eye_openness if which_eye == Eye.LEFT else frame_list[i].right_eye_openness
                sum_openness += eye_openness
                frames_used += 1
        return sum_openness / frames_used if frames_used > 0 else 0.0

    @staticmethod
    def _analyse_blinks_for_one_eye(frame_list: List[FrameEntity], which_eye: Eye) -> List[BlinkEntity]:
        consecutive_closed_for_blink = 1
        partial_blink_threshold = 0.50
        complete_blink_threshold = 0.40
        end_blink_threshold = 0.50

        partial_blink_started = False
        complete_blink_started = False
        blink_ended = False
        first_blink_frame = -1
        last_blink_frame = -1
        blink_sequence = 0

        eye_blink_list = []

        for frame_data in frame_list:
            eye_openness = frame_data.left_eye_openness if which_eye == Eye.LEFT else frame_data.right_eye_openness
            smoothed_openness = BlinkUtils._get_smoothed_openness(frame_list, which_eye, frame_data.frame_number)

            log.debug(f'Frame: {frame_data.frame_number}, Eye: {which_eye}, '
                      f'Openness: {eye_openness}, Smoothed Openness: {smoothed_openness}')

            if eye_openness <= partial_blink_threshold:
                if not partial_blink_started:
                    first_blink_frame = frame_data.frame_number
                    partial_blink_started = True
                if eye_openness <= complete_blink_threshold:
                    if partial_blink_started and not complete_blink_started:
                        complete_blink_started = True
                    elif not partial_blink_started:
                        first_blink_frame = frame_data.frame_number
                        complete_blink_started = True
            else:
                if eye_openness >= end_blink_threshold:
                    if partial_blink_started and not complete_blink_started:
                        last_blink_frame = frame_data.frame_number
                        blink_ended = True
                    elif complete_blink_started:
                        last_blink_frame = frame_data.frame_number
                        blink_ended = True

            if blink_ended:
                blink_sequence += 1
                blink = BlinkEntity(
                    eye=which_eye,
                    blink_type=BlinkType.COMPLETE if complete_blink_started else BlinkType.INCOMPLETE,
                    first_frame=first_blink_frame,
                    last_frame=last_blink_frame,
                    blink_sequence=blink_sequence
                )
                if blink.last_frame - blink.first_frame >= consecutive_closed_for_blink:
                    eye_blink_list.append(blink)
                    log.info(f'Blink detected: {blink.to_json()}')
                else:
                    blink_sequence -= 1

                partial_blink_started = False
                complete_blink_started = False
                blink_ended = False

        if partial_blink_started or complete_blink_started:
            last_blink_frame = frame_list[-1].frame_number
            blink_sequence += 1
            blink = BlinkEntity(
                eye=which_eye,
                blink_type=BlinkType.COMPLETE if complete_blink_started else BlinkType.INCOMPLETE,
                first_frame=first_blink_frame,
                last_frame=last_blink_frame,
                blink_sequence=blink_sequence
            )
            if blink.last_frame - blink.first_frame >= consecutive_closed_for_blink:
                eye_blink_list.append(blink)
                log.info(f'Blink detected: {blink.to_json()}')
            else:
                blink_sequence -= 1

        return eye_blink_list
    
def analyze_csv_blinks(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file with frame-level eye openness, normalizes the openness
    columns to the [0, 1] range, computes a centered 5-frame moving average,
    and returns a DataFrame of blink entities.

    Expected input columns:
        frame,left_opening,right_opening

    The function also accepts the legacy typo `right_opennning` and will use it
    if `right_opening` is not present.

    Args:
        file_path: The path to the CSV file.

    Returns:
        A pandas DataFrame containing blink entity information.
    """
    df = pd.read_csv(file_path)

    if 'right_opening' not in df.columns and 'right_opennning' in df.columns:
        df = df.rename(columns={'right_opennning': 'right_opening'})

    required_columns = {'frame', 'left_opening', 'right_opening'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f'Missing required columns in CSV: {sorted(missing_columns)}')

    def normalize_series(series: pd.Series) -> pd.Series:
        min_value = series.min()
        max_value = series.max()
        if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
            return pd.Series(0.0, index=series.index)
        return (series - min_value) / (max_value - min_value)

    df['left_opening'] = pd.to_numeric(df['left_opening'], errors='coerce')
    df['right_opening'] = pd.to_numeric(df['right_opening'], errors='coerce')
    df['frame'] = pd.to_numeric(df['frame'], errors='coerce')

    df = df.sort_values('frame').reset_index(drop=True)

    df['left_opening_norm'] = normalize_series(df['left_opening'])
    df['right_opennning_norm'] = normalize_series(df['right_opening'])

    df['left_opening_med'] = df['left_opening_norm'].rolling(window=5, center=True, min_periods=1).mean()
    df['right_opening_med'] = df['right_opennning_norm'].rolling(window=5, center=True, min_periods=1).mean()

    frame_entities = []
    for _, row in df.iterrows():
        frame_entities.append(
            FrameEntity(
                frame_number=int(row['frame']),
                left_eye_openness=row['left_opening_med'],
                right_eye_openness=row['right_opening_med'],
            )
        )

    blink_entities = BlinkUtils.analyse_blinks(frame_entities)

    blink_data = []
    for blink in blink_entities:
        blink_data.append(blink.to_json())

    # Create a pandas DataFrame from the blink data
    blink_df = pd.DataFrame(blink_data)

    return blink_df


def aggregate_bilateral_blinks_new(df_blinks, only_complete_blinks=False):
    if only_complete_blinks:
        df_blinks = df_blinks[df_blinks['blink_type'] == 'complete'].copy()

    df_left = df_blinks[df_blinks['eye'] == 'Eye.left'].copy()
    df_right = df_blinks[df_blinks['eye'] == 'Eye.right'].copy()

    # Merge cruzado (combina todas as piscadas)
    df_cross = df_left.merge(df_right, how='cross', suffixes=('_left', '_right'))

    # Filtra onde há interseção temporal (blinks que se sobrepõem)
    bilateral_blinks = df_cross[
        (df_cross['start_frame_left'] <= df_cross['end_frame_right']) &
        (df_cross['end_frame_left']  >= df_cross['start_frame_right'])
    ]

    # Usa o intervalo total da piscada bilateral (min start, max end)
    aggregated_blinks = bilateral_blinks.assign(
        start_frame = bilateral_blinks[['start_frame_left', 'start_frame_right']].min(axis=1),
        end_frame   = bilateral_blinks[['end_frame_left',  'end_frame_right']].max(axis=1)
    )[['start_frame', 'end_frame']].drop_duplicates().reset_index(drop=True)

    # Adiciona id sequencial
    aggregated_blinks['id'] = aggregated_blinks.index + 1

    # Reordena colunas
    return aggregated_blinks[['id', 'start_frame', 'end_frame']]