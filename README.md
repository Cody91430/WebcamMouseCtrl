# WebcamMouseCtrl

Prototype Python pour le suivi des mains (une ou deux) via MediaPipe Hands avec filtrage One Euro et HUD (landmarks, bbox, curseur lisse). Pas de clic/pincement dans cette version.

## Prerequis
- Python 3.10+ (teste sur Windows).
- Webcam fonctionnelle.

## Installation (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si l'activation est bloquee (execution policy), lancer:
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
.\.venv\Scripts\Activate.ps1
```

## Lancer le suivi
```powershell
python main.py --show-fps
```

Options utiles:
- `--camera-index`: index de la webcam (0 par defaut).
- `--width` / `--height`: resolution demandee.
- `--show-fps`: overlay FPS.
- `--no-mirror`: desactive le flip horizontal du flux.
- `--min-cutoff`, `--beta`, `--d-cutoff`: reglages du filtre One Euro.

Quitter: touche ESC dans la fenetre.

## Structure actuelle
- `main.py`: capture OpenCV, detection MediaPipe Hands (2 mains), filtrage One Euro, overlay landmarks/bbox/curseur lisse.
- `requirements.txt`: OpenCV, MediaPipe.
- `project.md`: cadrage du projet (pipeline, gestes, plan).

## HUD et suivi
- Landmarks + connexions MediaPipe.
- Bbox et label main (Left/Right + score).
- Croix verte sur le bout de l'index lisse (filtre One Euro).
