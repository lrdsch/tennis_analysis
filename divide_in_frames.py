import cv2
import os
import argparse

def extract_frames(video_path):
    # Controlla che il file esista
    if not os.path.isfile(video_path):
        print(f"Errore: il file '{video_path}' non esiste.")
        return

    # Ottiene il nome del file senza estensione e il percorso
    video_dir = os.path.dirname(video_path)
    video_name_ext = os.path.basename(video_path)
    video_name, _ = os.path.splitext(video_name_ext)

    # Crea la cartella di output con lo stesso nome del video
    output_dir = os.path.join(video_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)

    # Apre il video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Errore: impossibile aprire il video '{video_path}'")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_dir, f"{frame_count:06d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Estrazione completata: {frame_count} frame salvati in '{output_dir}'")

def main():
    parser = argparse.ArgumentParser(description="Estrai i frame da un video.")
    parser.add_argument('--name', type=str, required=True, help="Percorso del file video")
    args = parser.parse_args()

    extract_frames(args.name)

if __name__ == '__main__':
    main()
