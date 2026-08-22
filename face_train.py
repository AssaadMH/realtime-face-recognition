"""
=============================================================
  ÉTAPE 2 : ENTRAÎNEMENT DU MODÈLE
  - Lance ce script après avoir capturé le dataset
  - Génère un fichier 'model.yml' et 'labels.txt'
=============================================================
"""

import cv2
import os
import numpy as np
import pickle

# ─── Configuration ────────────────────────────────────────
DATASET_DIR = "dataset"
MODEL_FILE  = "model.yml"
LABELS_FILE = "labels.pkl"
# ──────────────────────────────────────────────────────────

def train_model():
    print("\n🔄 Chargement des données...\n")

    if not os.path.exists(DATASET_DIR):
        print("❌ Dossier 'dataset' introuvable. Lance d'abord face_dataset.py")
        return

    persons = [d for d in os.listdir(DATASET_DIR)
               if os.path.isdir(os.path.join(DATASET_DIR, d))]

    if not persons:
        print("❌ Aucune personne trouvée dans le dataset !")
        return

    print(f"👥 Personnes détectées : {', '.join(persons)}\n")

    faces_data  = []
    labels_data = []
    label_map   = {}   # id -> nom

    for idx, person in enumerate(persons):
        label_map[idx] = person
        person_dir = os.path.join(DATASET_DIR, person)
        images = [f for f in os.listdir(person_dir)
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        if not images:
            print(f"⚠️  Aucune image pour {person}, ignoré.")
            continue

        print(f"  📂 {person} : {len(images)} images (label={idx})")

        for img_file in images:
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))
            faces_data.append(img)
            labels_data.append(idx)

    if len(faces_data) == 0:
        print("❌ Aucune image valide trouvée !")
        return

    print(f"\n📊 Total images chargées : {len(faces_data)}")
    print("🧠 Entraînement du modèle LBPH en cours...")

    # Modèle LBPH (Local Binary Pattern Histogram) — léger et rapide
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    recognizer.train(faces_data, np.array(labels_data))

    # Sauvegarder le modèle et le mapping des labels
    recognizer.save(MODEL_FILE)
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(label_map, f)

    print(f"\n✅ Modèle sauvegardé : '{MODEL_FILE}'")
    print(f"✅ Labels sauvegardés : '{LABELS_FILE}'")
    print("\n➡️  Lance maintenant : python face_recognition_app.py\n")


if __name__ == "__main__":
    train_model()
