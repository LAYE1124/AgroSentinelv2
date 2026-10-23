# AgriSentinel - Prototype de diagnostic phytosanitaire

Prototype de solution intelligente pour le diagnostic des maladies foliaires de la tomate à partir d'une simple image de feuille.

## Objectif

AgriSentinel vise à aider les agriculteurs, agents de terrain et exploitants agricoles à identifier rapidement les principales pathologies foliaires sur tomate, en fournissant :

- un diagnostic visuel assisté par IA,
- un niveau de confiance estimé,
- une carte d'attention Grad-CAM pour expliquer la décision du modèle,
- des recommandations agronomiques adaptées à chaque maladie.

Le prototype est conçu pour une démonstration locale et un usage de preuve de concept, avec un focus sur les cas de tomate rencontrés dans les zones maraîchères comme les Niayes au Sénégal.

## Fonctionnalités

- Import d'une image de feuille de tomate au format JPG, JPEG ou PNG
- Analyse locale avec un modèle PyTorch
- Détection de 8 classes associées à des états foliaires de tomate :
  - Alternariose
  - Septoriose
  - Feuille saine
  - Tache bactérienne
  - Mildiou tardif
  - Mosaïque virale
  - Virus jaune
  - Feuille moisie
- Calcul de confiance sur la prédiction
- Visualisation Grad-CAM pour rendre le diagnostic explicable
- Affichage de symptômes typiques et recommandations de traitement biologique/chimique

## Architecture du prototype

Le prototype utilise :

- Streamlit pour l'interface utilisateur
- PyTorch et torchvision pour le modèle de classification
- EfficientNet-B0 comme architecture de base
- OpenCV pour le traitement visuel
- Grad-CAM pour l'explicabilité
- Pillow pour le prétraitement des images

## Structure du projet

```text
agrosentinel-demo/
├── app.py                  # Application Streamlit principal
├── best_model_efficientnet.pth  # Poids du modèle entraîné
├── requirements.txt        # Dépendances Python
├── style.css               # Styles personnalisés de l'interface
├── logo                    # Ressource visuelle du projet
├── .gitignore              # Fichiers ignorés par Git
└── readme.md               # Documentation du prototype
```

## Prérequis

- Python 3.10+
- pip
- environnement virtuel recommandé

## Installation

1. Ouvrir un terminal dans le dossier du projet.
2. Créer un environnement virtuel :

```bash
python -m venv .venv
```

3. Activer l'environnement :

Sous Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

Sous Windows CMD :

```cmd
.venv\Scripts\activate.bat
```

4. Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

Ensuite, ouvrir l'URL indiquée par Streamlit dans le navigateur (généralement http://localhost:8501).

## Notes importantes

- Le fichier `best_model_efficientnet.pth` doit être présent à la racine du projet pour que le modèle soit chargé correctement.
- Le prototype est destiné à une démonstration locale et à une validation fonctionnelle, pas encore à un déploiement production complet.
- Le système est optimisé pour le diagnostic de feuilles de tomate, avec des recommandations agronomiques orientées pratique et décision support.

## Exemple d'usage

1. Télécharger une image d'une feuille de tomate.
2. Cliquer sur le bouton d'analyse.
3. Consulter :
   - la maladie prédite,
   - le score de confiance,
   - la zone d'intérêt visuelle via Grad-CAM,
   - les symptômes et recommandations associées.

## Démo

Le prototype présente une interface moderne et orientée agronomie, avec un design inspiré des outils de décision et d'analyse de terrain. Il offre une première expérience utilisateur permettant d'évaluer la qualité d'un diagnostic IA pour des cultures maraîchères.

## Licence

Ce dépôt correspond à un prototype de démonstration. La licence exacte dépendra de la stratégie de diffusion du projet final.
