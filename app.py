import streamlit as st
import requests
from PIL import Image
import base64
import io
from streamlit_lottie import st_lottie
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import gc

# 1. Configuration de la page
st.set_page_config(
    page_title="AgriSentinel - Diagnostic Phytosanitaire",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour charger le fichier CSS externe
def load_css(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Fichier de style {file_name} introuvable.")

# Chargement du CSS et de FontAwesome
load_css("style.css")
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">', unsafe_allow_html=True)

# Fonction utilitaire pour charger une animation Lottie
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_scanning = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_w51pcehl.json")

# 2. Définition des classes
CLASSES_NAMES = [
    'Tomato_Early_blight_leaf',
    'Tomato_Septoria_leaf_spot',
    'Tomato_leaf',
    'Tomato_leaf_bacterial_spot',
    'Tomato_leaf_late_blight',
    'Tomato_leaf_mosaic_virus',
    'Tomato_leaf_yellow_virus',
    'Tomato_mold_leaf'
]

# Classe Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        with torch.enable_grad():
            output = self.model(input_tensor)
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
                
            score = output[0, class_idx]
            self.model.zero_grad()
            score.backward()
            
        if self.gradients is None or self.activations is None:
            raise ValueError("Impossible de capturer les gradients ou les activations.")
            
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.squeeze().cpu().numpy()
        return cam, class_idx

# 3. Chargement mis en cache du modèle et du module Grad-CAM (Optimisation RAM)
@st.cache_resource
def load_model_and_gradcam():
    DEVICE = torch.device("cpu")
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(256, len(CLASSES_NAMES))
    )
    
    weights_path = "best_model_efficientnet.pth"
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    target_layer = model.features[-1]
    grad_cam_extractor = GradCAM(model, target_layer)
    return model, grad_cam_extractor, DEVICE

model, grad_cam_extractor, DEVICE = load_model_and_gradcam()

# Pipeline de transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Base de connaissances agronomiques
RECOMMENDATIONS = {
    "Tomato_Early_blight_leaf": {
        "titre": "Alternariose (Early Blight)",
        "severite": "Modérée",
        "badge_class": "badge-moderate",
        "border_color": "#F59E0B",
        "symptomes": [
            "Taches concentriques sombres sur les feuilles anciennes",
            "Jaunissement progressif du feuillage autour des lésions",
            "Chute prématurée des feuilles infectées"
        ],
        "traitement_bio": "Élimination des feuilles atteintes, rotation des cultures, et pulvérisation préventive à base de cuivre.",
        "traitement_chimique": "Application de fongicides homologués (ex: chlorothalonil ou mancozèbe) dès l'apparition des premiers symptômes."
    },
    "Tomato_Septoria_leaf_spot": {
        "titre": "Septoriose (Septoria Leaf Spot)",
        "severite": "Modérée",
        "badge_class": "badge-moderate",
        "border_color": "#F59E0B",
        "symptomes": [
            "Petites taches circulaires avec un centre grisâtre et un bord sombre",
            "Présence de petits points noirs (pycnides) au centre des taches",
            "Assèchement progressif des feuilles de la base vers le haut"
        ],
        "traitement_bio": "Nettoyage des débris végétaux après récolte, espacement des plants pour l'aération, arrosage au pied.",
        "traitement_chimique": "Fongicides à base de cuivre ou de mancozèbe en cas de forte pression climatique."
    },
    "Tomato_leaf": {
        "titre": "Feuille Saine (Healthy)",
        "severite": "Nulle",
        "badge_class": "badge-healthy",
        "border_color": "#10B981",
        "symptomes": [
            "Feuillage vert vif et uniformément réparti",
            "Absence de taches, de nécrose ou de déformation",
            "Tiges robustes et vigoureuses"
        ],
        "traitement_bio": "Aucun traitement requis. Maintenir une bonne fertilisation, un suivi régulier de l'irrigation et des inspections préventives.",
        "traitement_chimique": "Aucun traitement chimique nécessaire."
    },
    "Tomato_leaf_bacterial_spot": {
        "titre": "Gale bactérienne (Bacterial Spot)",
        "severite": "Élevée",
        "badge_class": "badge-critical",
        "border_color": "#EF4444",
        "symptomes": [
            "Petites taches aqueuses devenant brunes et anguleuses sur les feuilles",
            "Lésions liégeuses ou croûteuses sur les fruits",
            "Défoliation partielle en cas d'attaque sévère"
        ],
        "traitement_bio": "Utilisation exclusive de semences certifiées saines, éviter de manipuler les plants mouillés, désinfection des outils.",
        "traitement_chimique": "Traitements préventifs à base de cuivre ou d'antibiotiques agricoles autorisés selon la législation locale."
    },
    "Tomato_leaf_late_blight": {
        "titre": "Mildiou de la tomate (Late Blight)",
        "severite": "Critique",
        "badge_class": "badge-critical",
        "border_color": "#EF4444",
        "symptomes": [
            "Taches brunes huileuses sur les feuilles inférieures",
            "Duvet blanc-gris sur le dessous des feuilles",
            "Taches brunes sur les fruits qui deviennent dures",
            "Flétrissement rapide des tiges",
            "Nécrose des feuilles à partir des bords"
        ],
        "traitement_bio": "Bouillie bordelaise (cuivre + chaux) en préventif. Extraits de prêle des champs en pulvérisation. Lutte biologique avec Trichoderma.",
        "traitement_chimique": "Mancozèbe, métalaxyl-M, cyazofamide. Traitement tous les 7-10 jours en conditions humides. Alternance des matières actives pour éviter la résistance."
    },
    "Tomato_leaf_mosaic_virus": {
        "titre": "Mosaïque de la tomate (Tomato Mosaic Virus)",
        "severite": "Élevée",
        "badge_class": "badge-critical",
        "border_color": "#EF4444",
        "symptomes": [
            "Motifs alternant zones claires et sombres (effet mosaïque) sur le limbe",
            "Déformation des jeunes feuilles (aspect filiforme)",
            "Croissance ralentie et rendements fortement diminués"
        ],
        "traitement_bio": "Arrachage immédiat des plants infectés, lavage rigoureux des mains et désinfection fréquente des outils de taille.",
        "traitement_chimique": "Aucun traitement curatif chimique antiviral direct. Lutte axée sur l'éradication des vecteurs et des plants malades."
    },
    "Tomato_leaf_yellow_virus": {
        "titre": "Enroulement jaune (TYLCV)",
        "severite": "Élevée",
        "badge_class": "badge-critical",
        "border_color": "#EF4444",
        "symptomes": [
            "Enroulement des marges des feuilles vers le haut (forme de cuillère)",
            "Jaunissement prononcé entre les nervures",
            "Nanisme de la plante et arrêt de la fructification"
        ],
        "traitement_bio": "Installation de pièges collants jaunes, pose de filets anti-insectes pour bloquer les aleurodes (mouches blanches).",
        "traitement_chimique": "Application d'insecticides ciblés contre les mouches blanches ou utilisation de savons noirs pour réduire les populations vectrices."
    },
    "Tomato_mold_leaf": {
        "titre": "Moisissure olivaire (Mold Leaf)",
        "severite": "Modérée",
        "badge_class": "badge-moderate",
        "border_color": "#F59E0B",
        "symptomes": [
            "Taches jaunes pâles sur la face supérieure des feuilles",
            "Feutrage olivâtre ou grisâtre sur la face inférieure",
            "Sécheresse et chute des feuilles atteintes"
        ],
        "traitement_bio": "Amélioration de la ventilation sous serre, réduction de l'humidité relative, espacement des pieds.",
        "traitement_chimique": "Application de fongicides adaptés (soufre ou produits homologués contre la cladosporiose) en cas de persistance."
    }
}

# --- BARRE LATÉRALE (Sidebar) ---
with st.sidebar:
    st.image("logo/logo-agrosentinel.png", width=130, caption="AgriSentinel - Diagnostic Phytosanitaire")
    st.markdown("---")
    # Mode sombre toggle
    darkMode = st.toggle("🌙 Mode Sombre", value=False)
    
    if darkMode:
        st.markdown("""
            <style>
            [data-testid="stAppViewContainer"] {
                background-color: #121212 !important;
                color: #E0E0E0 !important;
            }
            [data-testid="stAppViewContainer"] p, 
            [data-testid="stAppViewContainer"] span, 
            [data-testid="stAppViewContainer"] label {
                color: #E0E0E0 !important;
            }
            [data-testid="stSidebar"] {
                background-color: #1A2621 !important;
                border-right: 1px solid #2C3E35 !important;
            }
            [data-testid="stSidebar"] *, 
            [data-testid="stSidebar"] p, 
            [data-testid="stSidebar"] span, 
            [data-testid="stSidebar"] label, 
            [data-testid="stSidebar"] h3, 
            [data-testid="stSidebar"] div {
                color: #E0E0E0 !important;
            }
            .legend-color-box {
                color: var(--color-override) !important;
            }
            </style>
        """, unsafe_allow_html=True)
    
    st.markdown("### <i class='fas fa-info-circle'></i> À propos du Projet", unsafe_allow_html=True)
    st.markdown("Application de démonstration dédiée au diagnostic intelligent des cultures maraîchères (Cas pilote : zone des **Niayes, Sénégal**).")
    
    st.markdown("### <i class='fas fa-brain'></i> Choix du Modèle & IA", unsafe_allow_html=True)
    st.markdown("""
    - **Architecture** : EfficientNet-B0 (PyTorch)
    - **Explicabilité** : Grad-CAM intégré pour cartographier les zones d'attention.
    - **Classes** : 8 pathologies de tomates prises en charge.
    """)
    
    st.markdown("---")
    st.markdown("### <i class='fas fa-server'></i> État du Système", unsafe_allow_html=True)
    st.success("Moteur IA : Chargé localement (Optimisé)")

# --- CORPS PRINCIPAL DE L'APPLICATION ---
st.markdown("""
    <div class="header-container">
        <h1><i class="fas fa-leaf"></i> AgriSentinel : Assistant Phytosanitaire Intelligent</h1>
        <p>Soumettez un cliché de feuille de tomate pour obtenir un diagnostic instantané, une analyse explicable par IA et des recommandations ciblées.</p>
    </div>
""", unsafe_allow_html=True)

col_upload, col_preview = st.columns([1, 1], gap="large")
# --- SECTION D'IMPORTATION ET D'APERÇU DE L'IMAGE ---
with col_upload:
    st.markdown("### <i class='fas fa-cloud-upload-alt'></i> 1. Importer un cliché", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choisissez une photo de feuille de tomate...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        analyze_btn = st.button("Lancer l'analyse intelligente", type="primary")
    else:
        analyze_btn = False
        if "analysis_done" in st.session_state:
            del st.session_state.analysis_done
# --- SECTION D'APERÇU DE L'IMAGE ---
with col_preview:
    if uploaded_file is not None:
        st.markdown("### <i class='fas fa-image'></i> 2. Aperçu de l'image source", unsafe_allow_html=True)
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True, caption="Cliché soumis")

# Traitement de l'inférence locale
if analyze_btn and uploaded_file is not None:
    with st.container():
        col_lot1, col_lot2, col_lot3 = st.columns([1, 2, 1])
        with col_lot2:
            if lottie_scanning:
                st_lottie(lottie_scanning, height=180, key="scan_animation")
            st.markdown("<p style='text-align: center; color: #555;'><i class='fas fa-spinner fa-spin'></i> Analyse neuronale et calcul Grad-CAM en cours...</p>", unsafe_allow_html=True)
        # --- INFÉRENCE ET GRAD-CAM ---
        try:
            pil_img = Image.open(uploaded_file).convert("RGB")
            input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

            # 1. Inférence
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence, predicted_idx = torch.max(probabilities, 0)

            predicted_class = CLASSES_NAMES[predicted_idx.item()]
            conf_score = confidence.item() * 100

            # 2. Génération Grad-CAM
            encoded_image = ""
            try:
                input_tensor.requires_grad_()
                cam, _ = grad_cam_extractor.generate_cam(input_tensor, predicted_idx.item())

                img_cv2 = np.array(pil_img)
                img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
                h, w, _ = img_cv2.shape

                cam_resized = cv2.resize(cam, (w, h))
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                superimposed_img = cv2.addWeighted(img_cv2, 0.6, heatmap, 0.4, 0)

                _, buffer = cv2.imencode(".jpg", superimposed_img)
                encoded_image = base64.b64encode(buffer).decode("utf-8")
            except Exception as cam_error:
                # Fallback image d'origine si Grad-CAM échoue
                img_cv2 = np.array(pil_img)
                img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode(".jpg", img_cv2)
                encoded_image = base64.b64encode(buffer).decode("utf-8")

            gc.collect()

            # Stockage dans la session
            st.session_state.analysis_result = {
                "prediction": {
                    "class_name": predicted_class,
                    "confidence_score": round(conf_score, 2),
                },
                "grad_cam_image_base64": encoded_image
            }
            st.session_state.analysis_done = True

        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image : {str(e)}")
            st.session_state.analysis_done = False

# --- AFFICHAGE DES RÉSULTATS ---
if st.session_state.get("analysis_done", False):
    result = st.session_state.analysis_result
    pred_info = result["prediction"]
    
    class_name = pred_info["class_name"]
    confidence = pred_info["confidence_score"]
    reco = RECOMMENDATIONS.get(class_name, {})
    
    badge_cls = reco.get("badge_class", "badge-moderate")
    border_col = reco.get("border_color", "#F59E0B")
    severite_text = reco.get("severite", "Inconnue")

    st.markdown("---")
    st.markdown("## <i class='fas fa-chart-bar' style='color:#1E3D59;'></i> Résultats du Diagnostic", unsafe_allow_html=True)

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(
            f"**Diagnostic Principal**<br><span style='font-size: 1.5rem; font-weight: 600;'>"
            f"{reco.get('titre', class_name.replace('_', ' '))}</span>",
            unsafe_allow_html=True
        )
    with m_col2:
        st.metric(label="Indice de Confiance", value=f"{confidence}%")
    with m_col3:
        st.markdown(f"**Niveau de Sévérité**<br><br><span class='{badge_cls}'>{severite_text}</span>", unsafe_allow_html=True)

    res_col1, res_col2 = st.columns(2, gap="large")

    with res_col1:
        st.markdown("### <i class='fas fa-eye'></i> Explicabilité Visuelle (Grad-CAM)", unsafe_allow_html=True)
        st.markdown("Les zones en **rouge/jaune** mettent en évidence les caractéristiques pathologiques détectées par le modèle.")
        base64_img = result.get("grad_cam_image_base64")
        if base64_img:
            img_bytes = base64.b64decode(base64_img)
            grad_cam_image = Image.open(io.BytesIO(img_bytes))
            st.image(grad_cam_image, use_container_width=True, caption="Carte d'activation thermique")

    with res_col2:
        st.markdown("### <i class='fas fa-clipboard-list'></i> Recommandations & Diagnostic", unsafe_allow_html=True)
        if reco:
            symptomes_html = "".join([f"<li style='margin-bottom: 6px;'><i class='fas fa-exclamation-triangle' style='color: {border_col}; margin-right: 8px;'></i>{s}</li>" for s in reco.get("symptomes", [])])
            
            st.markdown(f"""
                <div style="background-color: #FFFFFF; padding: 1.5rem; border-radius: 8px; border-left: 6px solid {border_col}; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 1rem;">
                    <h4 style="margin-top:0; color: #1E3D59;"><i class='fas fa-search'></i> symptômes typiques :</h4>
                    <ul style="padding-left: 0; list-style: none; color: #333333; line-height: 1.5;">
                        {symptomes_html}
                    </ul>
                </div>
                
                <div style="background-color: #FFFFFF; padding: 1.2rem; border-radius: 8px; border-left: 6px solid #10B981; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 1rem;">
                    <h4 style="margin-top:0; color: #10B981;"><i class='fas fa-leaf'></i> Traitement Biologique :</h4>
                    <p style="color: #333333; margin-bottom: 0; line-height: 1.5;">{reco.get('traitement_bio', 'Non spécifié.')}</p>
                </div>

                <div style="background-color: #FFFFFF; padding: 1.2rem; border-radius: 8px; border-left: 6px solid #EF4444; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
                    <h4 style="margin-top:0; color: #EF4444;"><i class='fas fa-flask'></i> Traitement Chimique :</h4>
                    <p style="color: #333333; margin-bottom: 0; line-height: 1.5;">{reco.get('traitement_chimique', 'Non spécifié.')}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Suivez les bonnes pratiques phytosanitaires habituelles pour cette culture.")