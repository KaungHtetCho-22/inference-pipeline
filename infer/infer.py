import argparse
import sqlite3
import os
import time

# Configuration
DB_FILENAME = "app-data/soundscape.db"
WEIGHTS = "infer/weights/ait_bird_local_eca_nfnet_l0/latest-model-lin.pt"
LOG_FILE = "logs/processing_log.txt"


def log_message(message):
    """Logs messages to the log file."""
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    with open(LOG_FILE, "a") as log:
        log.write(f"{timestamp} {message}\n")


def check_file_in_db(rpi_id, file_name):
    """Check if the file exists in the database."""
    conn = sqlite3.connect(DB_FILENAME)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM RpiDevices WHERE pi_id = ?", (rpi_id,))
    device = cursor.fetchone()

    if not device:
        conn.close()
        return False

    device_id = device[0]
    cursor.execute("SELECT COUNT(*) FROM AudioFiles WHERE device_id = ? AND file_key = ?", (device_id, file_name))
    file_count = cursor.fetchone()[0]

    conn.close()
    return file_count > 0


def process_file(audio_path, rpi_id):
    """Process the file and update the database."""
    file_name = os.path.basename(audio_path)

    if check_file_in_db(rpi_id, file_name):
        log_message(f"[SKIPPED] {file_name} already exists in DB.")
        return

    log_message(f"[PROCESSING] Running model on {file_name}...")

    # Simulate inference (replace with actual model execution)
    time.sleep(2)

    # Insert into database (mocked here)
    conn = sqlite3.connect(DB_FILENAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO AudioFiles (device_id, file_key) VALUES ((SELECT id FROM RpiDevices WHERE pi_id = ?), ?)", (rpi_id, file_name))
    conn.commit()
    conn.close()

    log_message(f"[SUCCESS] Processed {file_name} and saved to DB.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="Path to the audio file")
    parser.add_argument("-r", "--rpi", required=True, help="Raspberry Pi ID")
    args = parser.parse_args()

    process_file(args.file, args.rpi)
