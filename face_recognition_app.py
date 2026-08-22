"""
=============================================================
  ÉTAPE 3 : APPLICATION DE RECONNAISSANCE FACIALE
  - Détection en temps réel depuis la webcam
  - Affiche le nom + le niveau de confiance
  - Commandes :
      Q  → Quitter
      S  → Capture une screenshot
      A  → Ajouter une nouvelle personne (relance dataset)
=============================================================
"""

import cv2
import os
import pickle
import numpy as np
from datetime import datetime

# ─── Configuration ────────────────────────────────────────
MODEL_FILE      = "model.yml"
LABELS_FILE     = "labels.pkl"
SCREENSHOTS_DIR = "screenshots"
CONFIDENCE_THRESHOLD = 70   # en dessous = reconnu, au dessus = inconnu
# ──────────────────────────────────────────────────────────

# Palette de couleurs par label (BGR)
COLORS = [
    (0, 255, 100),   # vert vif
    (0, 200, 255),   # cyan
    (255, 100, 0),   # bleu
    (200, 0, 255),   # violet
    (0, 165, 255),   # orange
]

def get_color(label_id):
    return COLORS[label_id % len(COLORS)]

def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=10):
    """Dessine un rectangle avec coins arrondis."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90,  0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0,   0, 90, color, thickness)

def draw_hud(frame, nb_faces, fps):
    """Affiche le HUD (interface) en overlay."""
    h, w = frame.shape[:2]

    # Fond semi-transparent en haut
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Infos texte
    now = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, f"FaceReco AI  |  {now}", (10, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(frame, f"Visages: {nb_faces}  FPS: {fps:.0f}", (w - 220, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (100, 255, 150), 1)

    # Barre commandes en bas
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 36), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
    cmds = "  [Q] Quitter    [S] Screenshot    [A] Ajouter personne"
    cv2.putText(frame, cmds, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

def run_recognition():
    # ── Vérifications ──────────────────────────────────────
    if not os.path.exists(MODEL_FILE):
        print("❌ Modèle introuvable. Lance d'abord face_train.py")
        return
    if not os.path.exists(LABELS_FILE):
        print("❌ Fichier labels introuvable. Ré-entraîne le modèle.")
        return

    # ── Chargement ─────────────────────────────────────────
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    with open(LABELS_FILE, "rb") as f:
        label_map = pickle.load(f)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la caméra !")
        return

    print("\n🚀 Application lancée ! Commandes : Q=Quitter  S=Screenshot  A=Ajouter\n")

    # Variables FPS
    prev_time = datetime.now()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # miroir horizontal
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)   # meilleure détection

        # ── Détection ──────────────────────────────────────
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        nb_faces = len(faces)

        for (x, y, w, h) in faces:
            # Rogner + redimensionner pour la reconnaissance
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (200, 200))

            label_id, confidence = recognizer.predict(roi_resized)

            # Déterminer nom et couleur
            if confidence < CONFIDENCE_THRESHOLD:
                name     = label_map.get(label_id, "Inconnu")
                color    = get_color(label_id)
                conf_pct = int(100 - confidence)
                tag      = f"{name}  {conf_pct}%"
            else:
                name     = "Inconnu"
                color    = (50, 50, 200)
                tag      = "Inconnu"

            # Rectangle arrondi autour du visage
            draw_rounded_rect(frame, (x, y), (x+w, y+h), color, 2)

            # Coins décoratifs (effet scan)
            L = 20
            cv2.line(frame, (x, y),     (x+L, y),     color, 3)
            cv2.line(frame, (x, y),     (x, y+L),     color, 3)
            cv2.line(frame, (x+w, y),   (x+w-L, y),   color, 3)
            cv2.line(frame, (x+w, y),   (x+w, y+L),   color, 3)
            cv2.line(frame, (x, y+h),   (x+L, y+h),   color, 3)
            cv2.line(frame, (x, y+h),   (x, y+h-L),   color, 3)
            cv2.line(frame, (x+w, y+h), (x+w-L, y+h), color, 3)
            cv2.line(frame, (x+w, y+h), (x+w, y+h-L), color, 3)

            # Étiquette nom + confiance
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
            lbl_x = max(x, 0)
            lbl_y = y - 12 if y > 40 else y + h + 30

            # Fond de l'étiquette
            cv2.rectangle(frame,
                          (lbl_x - 4, lbl_y - th - 6),
                          (lbl_x + tw + 4, lbl_y + 4),
                          color, -1)
            cv2.putText(frame, tag,
                        (lbl_x, lbl_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65,
                        (10, 10, 10), 1)

        # ── Calcul FPS ─────────────────────────────────────
        now = datetime.now()
        delta = (now - prev_time).total_seconds()
        if delta > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / delta)
        prev_time = now

        # ── HUD ────────────────────────────────────────────
        draw_hud(frame, nb_faces, fps)

        cv2.imshow("🎯 FaceReco AI — Détection & Reconnaissance en temps réel", frame)

        # ── Gestion touches ────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:   # Q ou Echap
            print("\n👋 Application fermée.")
            break

        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOTS_DIR, f"capture_{ts}.jpg")
            cv2.imwrite(path, frame)
            print(f"📷 Screenshot sauvegardé : {path}")

        elif key == ord('a'):
            print("\n➕ Ajout d'une nouvelle personne — relance face_dataset.py manuellement.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_recognition()
