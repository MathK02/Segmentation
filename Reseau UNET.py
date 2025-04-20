import logging
import os
import torch
import numpy
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from collections import OrderedDict
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

batch_size = 4
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
epochs = 20
path = 'E:/Documents/IOGS/Machine learning/Projet'  # Chemin absolu vers votre dossier de projet
train_val_path = os.path.join(path, 'train_val')
test_path = os.path.join(path, 'TEST')
image_size = 256
learning_rate = 5e-4
num_classes = 8
weight_decay = 1e-4

# Vérifiez la structure exacte de vos dossiers
train_images_path = os.path.join(train_val_path, 'images')
train_masks_path = os.path.join(train_val_path, 'masks')
test_images_path = os.path.join(test_path, 'images')
test_masks_path = os.path.join(test_path, 'masks')

# Ajoutez cette fonction pour vérifier les chemins avant de les utiliser
def verify_paths():
    paths = [train_images_path, train_masks_path, test_images_path, test_masks_path]
    for p in paths:
        if not os.path.exists(p):
            print(f"ATTENTION: Le chemin {p} n'existe pas!")
        else:
            print(f"Le chemin {p} existe.")

# Vérifiez les chemins avant de créer les datasets
verify_paths()

#         DataAugment
def random_rot_flip(image, label):
    k = numpy.random.randint(0, 4)
    image = numpy.rot90(image, k)
    label = numpy.rot90(label, k)
    axis = numpy.random.randint(0, 2)
    image = numpy.flip(image, axis=axis).copy()
    label = numpy.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = numpy.random.randint(-40, 40)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def randomGaussian(image, label, mean=0.2, sigma=0.3):
    def gaussianNoisy(im, mean=0.2, sigma=0.3):
        """
        Gaussian noise processing of images
        :param im: Single-channel images
        :param mean: Offset
        :param sigma: Standard deviation
        :return:
        """
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

     # Converting images into arrays
    img = numpy.asarray(image)
    img.flags.writeable = True  # Changing arrays to read and write mode
    width, height = img.shape[:2]
    img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])
    return numpy.uint8(img), label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Augmentation plus agressive et diversifiée
        if random.random() > 0.3:  # Augmenté la probabilité
            image, label = random_rot_flip(image, label)
        if random.random() > 0.3:  # Augmenté la probabilité
            image, label = random_rotate(image, label)
        if random.random() > 0.3:  # Augmenté la probabilité
            image, label = randomGaussian(image, label)
        
        # Ajout de variations de luminosité et de contraste
        if random.random() > 0.5:
            # Ajuster la luminosité aléatoirement
            brightness_factor = random.uniform(0.7, 1.3)
            image = numpy.clip(image * brightness_factor, 0, 255).astype(numpy.uint8)
        
        if random.random() > 0.5:
            # Ajuster le contraste aléatoirement
            contrast_factor = random.uniform(0.7, 1.3)
            mean = numpy.mean(image, axis=(0, 1), keepdims=True)
            image = numpy.clip((image - mean) * contrast_factor + mean, 0, 255).astype(numpy.uint8)

        return image, label

def rgb_to_class_idx(rgb_mask):
    """
    Convertit un masque RGB en indices de classe
    """
    # Définir la correspondance entre couleurs RGB et indices de classe
    # Ces valeurs sont à adapter selon votre jeu de données
    color_map = {
        (0, 0, 0): 0,       # Fond (noir)
        (255, 0, 0): 1,     # Rouge
        (0, 255, 0): 2,     # Vert
        (0, 0, 255): 3,     # Bleu
        (255, 255, 0): 4,   # Jaune
        (255, 0, 255): 5,   # Magenta
        (0, 255, 255): 6,   # Cyan
        (255, 255, 255): 7  # Blanc
    }
    
    # Créer un masque vide pour les indices de classe
    h, w, c = rgb_mask.shape
    class_mask = numpy.zeros((h, w), dtype=numpy.uint8)
    
    # Convertir chaque pixel RGB en indice de classe
    for color, idx in color_map.items():
        # Créer un masque booléen pour les pixels de cette couleur
        mask = numpy.all(rgb_mask == color, axis=2)
        # Assigner l'indice de classe aux pixels correspondants
        class_mask[mask] = idx
    
    return class_mask

class SUIM(Dataset):
    def __init__(self, img_path, label_path, transform = None):
        self.img_path = img_path
        self.label_path = label_path
        
        # Filtrer pour ne garder que les fichiers (pas les dossiers)
        all_items = os.listdir(self.label_path)
        self.label_data = [item for item in all_items if not os.path.isdir(os.path.join(self.label_path, item))]
        
        print(f"Total d'éléments: {len(all_items)}, Après filtrage des dossiers: {len(self.label_data)}")
        
        self.transform = transform # Data enhancement
        self.resize = transforms.Resize((image_size, image_size)) # Trimming of data
    def __len__(self):
        return len(self.label_data)  # Number of data returned

    def __getitem__(self, item):
        # Obtenir le nom du fichier de masque
        mask_filename = self.label_data[item]
        label_data = os.path.join(self.label_path, mask_filename)
        
        # Liste des préfixes connus (identiques pour l'entraînement et le test)
        known_prefixes = ['d_r_', 'f_r_', 'n_l_', 'w_r_']
        
        # Essayer d'abord la correspondance exacte
        img_filename = mask_filename
        img_data = os.path.join(self.img_path, img_filename)
        
        if not os.path.exists(img_data):
            # Si pas de correspondance exacte, essayer avec le même préfixe mais extension différente
            prefix = os.path.splitext(mask_filename)[0]  # Nom sans extension
            found = False
            
            for ext in ['.jpg', '.png', '.jpeg']:
                potential_img = os.path.join(self.img_path, prefix + ext)
                if os.path.exists(potential_img):
                    img_data = potential_img
                    found = True
                    break
            
            # Si toujours pas trouvé, essayer de décomposer le nom selon le format préfixenombre_
            if not found:
                # Extraire le préfixe et le numéro du masque
                mask_base = os.path.splitext(mask_filename)[0]  # Nom sans extension
                
                # Pour chaque préfixe connu
                for prefix in known_prefixes:
                    # Vérifier si le nom du masque commence par ce préfixe
                    if mask_base.startswith(prefix):
                        # Extraire la partie numérique (tout ce qui est entre le préfixe et le dernier underscore)
                        number_part = mask_base[len(prefix):-1]  # -1 pour ignorer le dernier underscore
                        
                        # Essayer de reconstruire le nom avec chaque préfixe connu
                        for test_prefix in known_prefixes:
                            for ext in ['.jpg', '.png', '.jpeg']:
                                potential_name = f"{test_prefix}{number_part}_{ext}"
                                potential_img = os.path.join(self.img_path, potential_name)
                                if os.path.exists(potential_img):
                                    img_data = potential_img
                                    found = True
                                    break
                            if found:
                                break
                    if found:
                        break
            
            # Si toujours pas trouvé, afficher une erreur détaillée
            if not found:
                available_files = os.listdir(self.img_path)
                print(f"Fichiers disponibles dans {self.img_path} (premiers 10): {available_files[:10]}...")
                raise FileNotFoundError(f"Image introuvable pour le masque {mask_filename}")
        
        # Charger les images
        img = Image.open(img_data)
        label = Image.open(label_data)
        
        # Reste du code inchangé
        img = self.resize(img)
        label = self.resize(label)
        img = numpy.array(img)
        label = numpy.array(label)
        
        # Convertir le masque RGB en indices de classe si nécessaire
        if label.ndim == 3 and label.shape[2] == 3:  # Si le masque est en RGB
            label = rgb_to_class_idx(label)
        
        sample = {'image': img, 'label': label}
        if self.transform:
            img, label = self.transform(sample)
        
        # Conversion en tenseurs PyTorch
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # Conversion en format CHW
        label = torch.from_numpy(label).long()  # Les étiquettes doivent être de type long
        
        img = img / 255.0  # Normalisation simple
        
        return img, label


'''The dataloader is loaded, and when it is loaded, it is processed in the trainloader.（img，label）'''

train_dataset = SUIM(train_images_path, train_masks_path,
                    transform=transforms.Compose(
                        [RandomGenerator(output_size=[image_size, image_size])]))

test_data = SUIM(test_images_path, test_masks_path, transform=None)

train_data, val_data = torch.utils.data.random_split(train_dataset, [1220, 305], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_data, batch_size=batch_size,
                          shuffle=True, num_workers=0,
                          drop_last=True)

val_loader = DataLoader(val_data, batch_size=batch_size,
                          shuffle=False, num_workers=0,
                          drop_last=True)

test_loader = DataLoader(test_data, batch_size=1,
                          shuffle=False, num_workers=0,
                          drop_last=True)

for data, mask in train_loader:
    data1 = data[0]
    # Convertir le tenseur du format (C, H, W) au format (H, W, C) pour l'affichage
    data1_np = data1.permute(1, 2, 0).numpy()  # Réorganiser les dimensions et convertir en numpy
    plt.imshow(data1_np)
    plt.show()
    plt.imshow(mask[0])
#     print(mask[0],mask[0].shape)
#     print(mask[0].numpy())
    plt.show()
    break

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # Ajout de dropout
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)   # Ajout de dropout
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=8, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# Fonction pour visualiser les prédictions
def visualize_predictions(model, test_loader, num_samples=5):
    model.eval()
    count = 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            if count >= num_samples:
                break
                
            images = images.float().to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Affichage des résultats
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(images[0].cpu().permute(1, 2, 0))
            axs[0].set_title('Image originale')
            axs[0].axis('off')
            
            axs[1].imshow(masks[0].cpu())
            axs[1].set_title('Masque réel')
            axs[1].axis('off')
            
            axs[2].imshow(preds[0].cpu())
            axs[2].set_title('Prédiction')
            axs[2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            count += 1

# Fonction d'évaluation avec barre de progression
def evaluate_model(model, test_loader):
    model.eval()
    dice_scores = []
    
    print("Évaluation du modèle...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images = images.float().to(device)
            masks = masks.long().to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Calcul du score Dice pour chaque classe
            for cls in range(num_classes):
                pred_cls = (preds == cls).float()
                mask_cls = (masks == cls).float()
                
                intersection = (pred_cls * mask_cls).sum()
                union = pred_cls.sum() + mask_cls.sum()
                
                if union > 0:
                    dice = (2 * intersection) / union
                    dice_scores.append(dice.item())
    
    mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
    print(f"Score Dice moyen: {mean_dice:.4f}")
    return mean_dice

# Fonction d'entraînement avec barre de progression
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 5  # Nombre d'époques à attendre avant early stopping
    patience_counter = 0
    
    # Scheduler pour réduire le taux d'apprentissage quand la perte stagne
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Trouver le prochain numéro de modèle disponible
    model_number = 1
    while os.path.exists(f'best_model_{model_number}.pth'):
        model_number += 1
    
    print(f"Ce sera le modèle numéro {model_number}")
    
    for epoch in range(num_epochs):
        # Mode entraînement
        model.train()
        running_loss = 0.0
        
        # Barre de progression pour l'entraînement
        print(f"Époque {epoch+1}/{num_epochs} - Entraînement")
        for images, masks in tqdm(train_loader):
            images = images.float().to(device)
            masks = masks.long().to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward et optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Mode validation
        model.eval()
        val_loss = 0.0
        
        print(f"Époque {epoch+1}/{num_epochs} - Validation")
        with torch.no_grad():
            for images, masks in tqdm(val_loader):
                images = images.float().to(device)
                masks = masks.long().to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        # Mettre à jour le scheduler
        scheduler.step(epoch_val_loss)
        
        print(f'Époque {epoch+1}/{num_epochs}, Perte entraînement: {epoch_train_loss:.4f}, Perte validation: {epoch_val_loss:.4f}')
        
        # Sauvegarde du meilleur modèle
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0  # Réinitialiser le compteur
            
            # Sauvegarder avec un nom incrémental
            model_path = f'best_model_{model_number}.pth'
            torch.save(model.state_dict(), model_path)
            
            # Créer également un lien symbolique ou une copie nommée "latest_best_model.pth"
            latest_path = 'latest_best_model.pth'
            if os.path.exists(latest_path):
                os.remove(latest_path)
            torch.save(model.state_dict(), latest_path)
            
            print(f'Meilleur modèle sauvegardé sous {model_path} et {latest_path}!')
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping après {epoch+1} époques")
            break
            
        # Afficher les courbes de perte à chaque époque
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(train_losses)), train_losses, label='Perte d\'entraînement')
        plt.plot(range(len(val_losses)), val_losses, label='Perte de validation')
        plt.xlabel('Époques')
        plt.ylabel('Perte')
        plt.title('Courbes de perte pendant l\'entraînement')
        plt.legend()
        plt.savefig(f'loss_curves_epoch_{epoch+1}.png')
        plt.close()
    
    return train_losses, val_losses

def analyze_dataset_format():
    """Analyse le format des images et des masques"""
    # Vérifier un exemple d'image et de masque
    img_path = os.path.join(path, 'train_val', 'images')
    mask_path = os.path.join(path, 'train_val', 'masks')
    
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print("Chemins d'accès incorrects!")
        return
    
    img_files = os.listdir(img_path)
    mask_files = os.listdir(mask_path)
    
    if len(img_files) == 0 or len(mask_files) == 0:
        print("Aucun fichier trouvé!")
        return
    
    # Ouvrir un exemple d'image et de masque
    img = Image.open(os.path.join(img_path, img_files[0]))
    mask = Image.open(os.path.join(mask_path, mask_files[0]))
    
    # Afficher les informations
    print(f"Image: format={img.format}, mode={img.mode}, taille={img.size}")
    print(f"Masque: format={mask.format}, mode={mask.mode}, taille={mask.size}")
    
    # Afficher l'image et le masque
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Image exemple")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title("Masque exemple")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_mask_colors():
    """Analyse les couleurs uniques dans les masques"""
    mask_path = train_masks_path
    mask_files = os.listdir(mask_path)
    
    if len(mask_files) == 0:
        print("Aucun fichier de masque trouvé!")
        return
    
    # Ouvrir un exemple de masque
    mask = Image.open(os.path.join(mask_path, mask_files[0]))
    mask_array = numpy.array(mask)
    
    if mask_array.ndim == 3 and mask_array.shape[2] == 3:  # Si le masque est en RGB
        # Trouver les couleurs uniques
        unique_colors = numpy.unique(mask_array.reshape(-1, mask_array.shape[2]), axis=0)
        print(f"Couleurs uniques dans le masque ({len(unique_colors)}):")
        for i, color in enumerate(unique_colors):
            print(f"  Classe {i}: RGB{tuple(color)}")
    else:
        # Si le masque est en niveaux de gris
        unique_values = numpy.unique(mask_array)
        print(f"Valeurs uniques dans le masque: {unique_values}")

# Fonction pour calculer et tracer les courbes d'accuracy
def plot_accuracy_curves(model, train_loader, val_loader, test_loader=None):
    """
    Calcule et trace les courbes d'accuracy pour le modèle
    """
    model.eval()
    datasets = [('Train', train_loader), ('Validation', val_loader)]
    if test_loader:
        datasets.append(('Test', test_loader))
    
    results = {}
    
    for name, loader in datasets:
        correct = 0
        total = 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        print(f"Calcul de l'accuracy sur l'ensemble {name}...")
        with torch.no_grad():
            for images, masks in tqdm(loader):
                images = images.float().to(device)
                masks = masks.long().to(device)
                
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                
                # Accuracy globale
                correct += (preds == masks).sum().item()
                total += masks.numel()
                
                # Accuracy par classe
                for cls in range(num_classes):
                    class_correct[cls] += ((preds == cls) & (masks == cls)).sum().item()
                    class_total[cls] += (masks == cls).sum().item()
        
        # Calcul des accuracies
        accuracy = correct / total if total > 0 else 0
        class_accuracies = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                           for i in range(num_classes)]
        
        results[name] = {
            'global': accuracy,
            'per_class': class_accuracies
        }
        
        print(f"Accuracy {name}: {accuracy:.4f}")
        for i, acc in enumerate(class_accuracies):
            print(f"  Classe {i}: {acc:.4f} ({class_total[i]} pixels)")
    
    # Tracer l'accuracy par classe
    plt.figure(figsize=(12, 8))
    
    # Préparer les données pour le graphique
    class_names = [f"Classe {i}" for i in range(num_classes)]
    x = numpy.arange(len(class_names))
    width = 0.25
    multiplier = 0
    
    # Tracer les barres pour chaque ensemble de données
    for name, result in results.items():
        offset = width * multiplier
        plt.bar(x + offset, result['per_class'], width, label=name)
        multiplier += 1
    
    # Ajouter les étiquettes et la légende
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.title('Accuracy par classe')
    plt.xticks(x + width, class_names, rotation=45)
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ajouter les valeurs d'accuracy globale
    global_acc_text = ", ".join([f"{name}: {result['global']:.4f}" for name, result in results.items()])
    plt.figtext(0.5, 0.01, f"Accuracy globale: {global_acc_text}", 
               ha="center", fontsize=10, 
               bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.savefig('accuracy_per_class.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

# Encapsuler le code d'entraînement
if __name__ == "__main__":
    # Création du modèle
    model = UNet(n_channels=3, n_classes=num_classes, bilinear=True).to(device)
    
    # Utiliser un optimiseur avec weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Utiliser une fonction de perte avec pondération des classes
    # Calculer les poids des classes en fonction de leur fréquence dans les données d'entraînement
    class_weights = None
    try:
        # Calculer les poids des classes (optionnel, peut être supprimé si cela ralentit trop)
        print("Calcul des poids des classes...")
        class_counts = torch.zeros(num_classes)
        for _, mask in tqdm(train_loader):
            for c in range(num_classes):
                class_counts[c] += (mask == c).sum().item()
        
        # Inverser les fréquences pour obtenir les poids
        class_weights = 1.0 / (class_counts + 1e-8)
        # Normaliser les poids
        class_weights = class_weights / class_weights.sum() * num_classes
        print(f"Poids des classes: {class_weights}")
        class_weights = class_weights.to(device)
    except Exception as e:
        print(f"Erreur lors du calcul des poids des classes: {e}")
        print("Utilisation de poids uniformes")

    # Fonction de perte avec pondération des classes
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Entraînement du modèle
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)
    
    # Évaluation du modèle
    dice_score = evaluate_model(model, test_loader)
    
    # Visualisation des prédictions
    visualize_predictions(model, test_loader)
    
    # Visualisation des courbes de perte
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Perte d\'entraînement')
    plt.plot(val_losses, label='Perte de validation')
    plt.xlabel('Époques')
    plt.ylabel('Perte')
    plt.legend()
    plt.title('Courbes de perte pendant l\'entraînement')
    plt.show()
    
    # Analyser le format des données
    analyze_dataset_format()
    
    # Analyser les couleurs des masques
    analyze_mask_colors()
    
    # Après l'entraînement du modèle
    # Charger le meilleur modèle
    best_model = UNet(n_channels=3, n_classes=num_classes, bilinear=True).to(device)
    best_model.load_state_dict(torch.load('latest_best_model.pth'))
    
    # Calculer et tracer les courbes d'accuracy
    accuracy_results = plot_accuracy_curves(best_model, train_loader, val_loader, test_loader)