# WebcamMouseCtrl

Document d'introduction et cadrage pour une app Python qui controle la souris via la webcam.

## Objectif
- Suivre en temps reel un point ou des gestes de la main/du visage pour deplacer le curseur et declencher clics/defilement sans materiel externe.
- Priorite a la stabilite (peu de faux positifs), a la reactivite (>= 20-30 fps) et a une sortie facile (touche d'arret).

## Termes et notions
- **Capture** : flux video brut recupere par la webcam (OpenCV).
- **Landmarks** : points de reperage 2D/3D (doigts, nez, yeux) fournis par un modele (ex: MediaPipe Hands/FaceMesh/Pose).
- **ROI** (region of interest) : zone utile dans l'image pour cadrer/recadrer l'analyse.
- **Gesture** : motif detecte (pincement pouce-index, clignement, maintien doigt leve).
- **Mapping** : conversion coordonnees camera -> coordonnees ecran avec mise a l'echelle et offset.
- **Filtrage** : lissage pour reduire le bruit (moyenne mobile, One Euro filter, seuils de mouvement).

## Pipeline general
1) Capture video: lecture webcam, eventuelle reduction de resolution pour le CPU.
2) Detection: modele leger (MediaPipe) pour extraire landmarks mains/visage.
3) Mapping: normaliser, puis projeter sur l'ecran selon une zone de calibration.
4) Filtrage: limiter les sauts, plafonner la vitesse, debouncer les clics.
5) Actions souris: deplacement, clic gauche/droit, drag, defilement via pyautogui/pynput.
6) Feedback: overlay visuel (curseur, box de calibration, etat du geste) et raccourci d'arret (ESC).

## Gestes cibles (MVP)
- Deplacement: suivre le bout de l'index (main levee) ou la pointe du nez (mode tete).
- Clic gauche: pincement pouce-index maintenu bref.
- Clic droit: pincement avec delai plus long ou pouce-majeur.
- Drag: maintenir pincement pendant le mouvement.
- Defilement: distance pouce-index mappee a la molette (ou geste vertical du poignet).

## Composants logiciels
- Capture/affichage: OpenCV.
- Detection landmarks: MediaPipe (Hands ou FaceMesh); option: modele ultraleger si besoin sans GPU.
- Actions souris: pyautogui ou pynput (pynput souvent plus fiable sous Windows).
- Numerique: numpy; filtrage optionnel (One Euro filter custom).

## Contraintes UX et securite
- Toujours offrir une touche d'arret (ESC) et/ou une combinaison clavier.
- Afficher un indicateur clair de l'etat (tracking actif, geste detecte, clic en cours).
- Ajouter une phase de calibration initiale (zone de travail, main droite/gauche).
- Eviter les clics fantomes: seuils de distance, hysteresis, debounce temporel.

## Plan de dev propose
- Etape 1: squelette capture + affichage + sortie ESC.
- Etape 2: integration MediaPipe Hands; extraire l'index; mapping simple centre-ecran.
- Etape 3: filtrage basique (moyenne mobile) + mapping ecran en respectant le ratio.
- Etape 4: gestes pincement -> clic/drag; molette optionnelle.
- Etape 5: overlay HUD (curseur, statut geste) et petite calibration.
- Etape 6: packaging (requirements.txt), guide d'usage, tests manuels checklist.

## Tests manuels rapides
- Latence ressentie < 50-70 ms.
- Deplacement stable sans tremblement visible.
- Aucun clic fantome en restant immobile.
- Raccourci ESC fonctionne a tout moment.
- Comportement correct en cas de perte de main/visage (pause, pas de clic).
