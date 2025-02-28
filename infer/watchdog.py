import os
import time
import sqlite3
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuration
BASE_DIR = "../bio-data"
DB_FILENAME = "database/score-training-data-score-5-debug.db"
INFER_SCRIPT = "infer.py"
LOG_FILE = "logs/processing_log.txt"

# Mapping of Raspberry Pi devices to bio class and PI type
RPi_BIO_CLASSES = {
    "RPiID-0000000090d15aba": 5,
    "RPiID-00000000a1007a14": 5
}

RPi_PI_TYPE = {
    "RPiID-0000000090d15aba": 1,
    "RPiID-00000000a1007a14": 1
}


def log_message(message):
    """Logs messages to the log file and prints to console."""
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    message = f"{timestamp} {message}"
    with open(LOG_FILE, "a") as log:
        log.write(message + "\n")
    print(message)


def check_file_in_db(rpi_id, file_name):
    """Check if the audio file is already in the database."""
    conn = sqlite3.connect(DB_FILENAME)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM RpiDevices WHERE pi_id = ?", (rpi_id,))
    device = cursor.fetchone()

    if not device:
        conn.close()
        return False  # Device not found in DB

    device_id = device[0]
    cursor.execute("SELECT COUNT(*) FROM AudioFiles WHERE device_id = ? AND file_key = ?", (device_id, file_name))
    file_count = cursor.fetchone()[0]

    conn.close()
    return file_count > 0  # True if file exists


def process_audio_file(audio_path):
    """Processes a newly added audio file using infer.py."""
    rpi_id = os.path.basename(os.path.dirname(os.path.dirname(audio_path)))  # Extract RPi ID from path
    file_name = os.path.basename(audio_path)

    if rpi_id not in RPi_BIO_CLASSES or rpi_id not in RPi_PI_TYPE:
        log_message(f"[SKIPPED] {audio_path} - Unknown RPi ID")
        return

    if check_file_in_db(rpi_id, file_name):
        log_message(f"[SKIPPED] {file_name} - Already in database")
        return

    log_message(f"[PROCESSING] {file_name} - Running inference...")

    # Run inference
    command = ["python3", INFER_SCRIPT, "-f", audio_path, "-r", rpi_id]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log_message(f"[SUCCESS] Processed {file_name} and saved to DB")
    except subprocess.CalledProcessError as e:
        log_message(f"[ERROR] Inference failed for {file_name}: {e}")


class AudioFileHandler(FileSystemEventHandler):
    """Watchdog event handler for monitoring new audio files."""
    def on_created(self, event):
        if event.is_directory:
            return

        if event.src_path.endswith(".wav"):
            time.sleep(1)  # Small delay to ensure file is fully written
            process_audio_file(event.src_path)


def start_monitoring():
    """Starts monitoring the base directory for new audio files."""
    observer = Observer()
    event_handler = AudioFileHandler()
    observer.schedule(event_handler, BASE_DIR, recursive=True)
    observer.start()

    log_message(f"🚀 Monitoring directory: {BASE_DIR}")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    start_monitoring()
