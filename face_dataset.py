"""
=============================================================
  ÉTAPE 1 : CAPTURE DU DATASET FACIAL
  - Lance ce script en premier
  - Entre ton nom quand demandé
  - Le programme prend 50 photos de ton visage
=============================================================
"""

import cv2
import os

# ─── Configuration ────────────────────────────────────────
DATASET_DIR = "dataset"
NB_PHOTOS   = 50          # nombre de photos par personne
DELAY_MS    = 100         # délai entre chaque capture (ms)
# ──────────────────────────────────────────────────────────

def create_dataset():
    # Détecteur de visage Haar Cascade (fourni avec OpenCV)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Saisie du nom de la personne
    name = input("\n👤 Entre ton prénom/nom : ").strip()
    if not name:
        print("❌ Nom invalide !")
        return

    # Créer le dossier de sauvegarde
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    # Ouvrir la caméra
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la caméra !")
        return

    print(f"\n📸 Capture en cours pour : {name}")
    print("   Regarde bien la caméra... (appuie sur Q pour arrêter)\n")

    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if count < NB_PHOTOS:
                # Sauvegarder le visage recadré
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))
                filename = os.path.join(person_dir, f"{count:03d}.jpg")
                cv2.imwrite(filename, face_img)
                count += 1

            # Dessiner le rectangle + compteur
            color = (0, 255, 0) if count < NB_PHOTOS else (0, 200, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{count}/{NB_PHOTOS}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Barre de progression
        progress = int((count / NB_PHOTOS) * frame.shape[1])
        cv2.rectangle(frame, (0, frame.shape[0]-20),
                      (progress, frame.shape[0]), (0, 255, 0), -1)
        cv2.putText(frame, f"Dataset : {name}  [{count}/{NB_PHOTOS}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("📸 Capture Dataset - Appuie sur Q pour quitter", frame)

        if count >= NB_PHOTOS:
            print(f"✅ {NB_PHOTOS} photos sauvegardées dans '{person_dir}'")
            cv2.waitKey(1500)
            break

        key = cv2.waitKey(DELAY_MS) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n➡️  Lance maintenant : python face_train.py\n")


if __name__ == "__main__":
    create_dataset()
