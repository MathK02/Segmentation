import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Importez votre modèle et les fonctions nécessaires
from reseau import UNet, rgb_to_class_idx  # Assurez-vous que reseau.py est dans le même dossier

# Paramètres
image_size = 256
num_classes = 8
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Paramètre pour choisir le modèle (changez cette valeur pour utiliser un modèle différent)
# Exemples: 'best_model_1.pth', 'best_model_2.pth', 'latest_best_model.pth'
MODEL_CHOICE = 'latest_best_model.pth'

# Chemin vers le modèle sauvegardé
model_path = MODEL_CHOICE

# Vérifier si le modèle existe
if not os.path.exists(model_path):
    print(f"Erreur: Le fichier modèle '{model_path}' n'existe pas!")
    print("Répertoire de travail actuel :", os.getcwd())
    
    # Afficher les modèles disponibles
    available_models = [f for f in os.listdir('.') if f.endswith('.pth')]
    if available_models:
        print("\nModèles disponibles:")
        for model in available_models:
            print(f"  - {model}")
        print("\nChangez la valeur de MODEL_CHOICE pour utiliser un de ces modèles.")
    exit(1)

# Créer une instance du modèle avec bilinear=True pour correspondre au modèle entraîné
model = UNet(n_channels=3, n_classes=num_classes, bilinear=True).to(device)

# Charger les poids du modèle
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Modèle '{model_path}' chargé avec succès!")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
    exit(1)

# Mettre le modèle en mode évaluation
model.eval()

# Fonction pour prédire sur une seule image
def predict_image(image_path):
    # Vérifier si l'image existe
    if not os.path.exists(image_path):
        print(f"Erreur: L'image '{image_path}' n'existe pas!")
        return None
    
    # Charger et prétraiter l'image
    img = Image.open(image_path)
    img = transforms.Resize((image_size, image_size))(img)
    img_display = np.array(img)  # Pour l'affichage
    
    img = np.array(img)
    img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    img = img / 255.0  # Normalisation
    
    # Prédiction
    with torch.no_grad():
        img = img.to(device)
        output = model(img)
        pred = torch.argmax(output, dim=1)
    
    # Convertir en numpy pour l'affichage
    pred = pred.cpu().numpy()[0]
    
    # Afficher l'image originale et la prédiction
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_display)
    plt.title('Image originale')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(pred)
    plt.title('Prédiction')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return pred

# Fonction pour prédire sur un dossier d'images
def predict_folder(folder_path, max_images=5):
    # Vérifier si le dossier existe
    if not os.path.exists(folder_path):
        print(f"Erreur: Le dossier '{folder_path}' n'existe pas!")
        return
    
    # Lister les images dans le dossier
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Aucune image trouvée dans le dossier '{folder_path}'!")
        return
    
    # Limiter le nombre d'images à traiter
    image_files = image_files[:max_images]
    
    # Prédire sur chaque image
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        print(f"Prédiction pour l'image: {img_file}")
        predict_image(img_path)

# Exemple d'utilisation
if __name__ == "__main__":
    # Utiliser les mêmes chemins que dans reseau.py
    path = 'E:/Documents/IOGS/Machine learning/Projet'  # Chemin absolu vers votre dossier de projet
    test_path = os.path.join(path, 'TEST')
    test_images_path = os.path.join(test_path, 'images')
    
    # Prédire sur un dossier d'images
    predict_folder(test_images_path, max_images=3)