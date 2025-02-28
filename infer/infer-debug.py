import argparse
import os
from pathlib import Path
import librosa
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset import TestDataset
from model import AttModel
import sys
from datetime import datetime, timezone
from sqlalchemy.exc import IntegrityError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monsoon_biodiversity_common.config import cfg as CFG
from monsoon_biodiversity_common.db_model_debug import init_database, RpiDevices, SpeciesDetections, AudioFiles

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Create full database path
db_path = 'infer/app-data/score-training-data-score-5-debug.db'
DATABASE_URL = f'sqlite:///{db_path}'
print(f"Database URL: {DATABASE_URL}")

Session = init_database(DATABASE_URL)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-W", "--weight", help="Weight file of sound classifier model",
                    default="./weights/ait_bird_local_eca_nfnet_l0/latest-model-lin.pt")
parser.add_argument("-I", "--input_directory", type=str, required=True,
                    help="Path to a device directory containing the audio files to be processed")
parser.add_argument("-C", "--bio_class", type=int, required=True,
                    help="Biodiversity ground truth class/level.")
parser.add_argument("--db-filename", type=str, default="score-training-data.db",
                    help="Filename of SQLite database file to be created")
parser.add_argument("--pi-type", type=int, choices=[0, 1], required=True,  
                    help="Specify the Raspberry Pi device type: 0 for mobile, 1 for station")  

parser_args = parser.parse_args()

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(parser_args.weight, map_location=device)['state_dict']
model = AttModel(
    backbone=CFG.backbone,
    num_class=CFG.num_classes,
    infer_period=5,
    cfg=CFG,
    training=False,
    device=device
)
model.load_state_dict(state_dict)
model = model.to(device)
model.logmelspec_extractor = model.logmelspec_extractor.to(device)
model.eval()

def prediction_for_clip(audio_path):
    prediction_dict = {}
    classification_dict = {}

    clip, _ = librosa.load(audio_path, sr=32000)
    duration = librosa.get_duration(y=clip, sr=32000)
    seconds = list(range(5, int(duration), 5))

    filename = Path(audio_path).stem
    row_ids = [filename + f"_{second}" for second in seconds]

    test_df = pd.DataFrame({"row_id": row_ids, "seconds": seconds})
    dataset = TestDataset(df=test_df, clip=clip, cfg=CFG)
    loader = torch.utils.data.DataLoader(dataset, **CFG.loader_params['valid'])

    for inputs in tqdm(loader):
        row_ids = inputs.pop('row_id')
        with torch.no_grad():
            output = model(inputs)['logit']
            for row_id_idx, row_id in enumerate(row_ids):
                logits = output[row_id_idx, :].sigmoid().detach().cpu().numpy()
                prediction_dict[row_id] = {CFG.target_columns[i]: logits[i] for i in range(len(CFG.target_columns))}
                classification_dict[row_id] = {
                    'row_id': row_id,
                    'Class': CFG.target_columns[np.argmax(logits)],
                    'Score': np.max(logits)
                }
    return prediction_dict, classification_dict

def save_predictions_to_db(audio_path, classification_dict, class_level, pi_type, session):  
    path_parts = Path(audio_path).parts
    
    if len(path_parts) < 3:
        return
    
    pi_id = path_parts[-3]
    recording_date_str = path_parts[-2]

    try:
        recording_date = datetime.strptime(recording_date_str, "%Y-%m-%d").date()
    except ValueError:
        return
    
    audio_filename = Path(audio_path).name
    unique_file_key = f"{pi_id}_{recording_date}_{audio_filename}"  # Ensure uniqueness

    device = session.query(RpiDevices).filter_by(pi_id=pi_id).first()
    if not device:
        device = RpiDevices(pi_id=pi_id, pi_type=pi_type)  
        session.add(device)
        session.commit()

    audio_file = session.query(AudioFiles).filter_by(
        file_key=unique_file_key,  
        device_id=device.id,
        recording_date=recording_date  
    ).first()

    if not audio_file:
        audio_file = AudioFiles(
            device_id=device.id,
            recording_date=recording_date,
            file_key=unique_file_key
        )
        session.add(audio_file)
        session.commit()

    new_detections = []

    for row_id, classification in classification_dict.items():
        existing_detection = session.query(SpeciesDetections).filter(
            SpeciesDetections.audio_file_id == audio_file.id,
            SpeciesDetections.time_segment_id == row_id
        ).first()

        if not existing_detection:
            new_detections.append(SpeciesDetections(
                audio_file_id=audio_file.id,
                time_segment_id=row_id,
                species_class=classification['Class'],
                confidence_score=classification['Score'],
                created_at=datetime.now(timezone.utc),
                collected_biodiversity_level=class_level
            ))

    session.add_all(new_detections)
    session.commit()


# Process input directory
input_dir = parser_args.input_directory
session = Session()

print(f"Processing files in {input_dir}")
for audio_file in sorted(os.listdir(input_dir)):
    if audio_file.endswith(".wav"):
        audio_path = os.path.join(input_dir, audio_file)
        print(f"Processing {audio_path}")
        prediction_dict, classification_dict = prediction_for_clip(audio_path)
        save_predictions_to_db(audio_path, classification_dict, parser_args.bio_class, parser_args.pi_type, session)

session.close()
print("Inference complete!")
