import cv2
import os

def extract_frames(video_path, output_folder):
    # Vytvoření složky pro uložení snímků, pokud ještě neexistuje
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Načtení videa
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened(): # Kontrola, zda se video podařilo otevřít
        print("Chyba: video se nepodařilo otevřít.")
        exit(1)

    frame_count = 0
    while True:
        # Přečtení dalšího snímku
        ret, frame = video_capture.read()
        if not ret:
            break

        # Uložení sníkmku jako obrázek
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    video_capture.release()
    print(f"Extracted {frame_count} frames and saved them to {output_folder}.")


extract_frames("train_video.mp4", r"L:\CVUT\1roc\projekt1\frames")