# Show, Attend and Tell — Neural Image Caption Generation with Visual Attention

Reproduction du papier **Xu et al., ICML 2015** dans le cadre du projet de Deep Learning — 4DS5, Semaine 14.

---

## Contexte académique

| | |
|---|---|
| **Cours** | Deep Learning — 4DS5 |
| **Projet** | Reproduction d'un article scientifique |
| **Papier reproduit** | Xu et al., *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*, ICML 2015 |
| **Référence de reproduction** | Liu & Brailsford, *Reproducing "Show, Attend and Tell"*, J. Phys.: Conf. Ser. 2589 (2023) |
| **Validation** | Semaine 14 |

---

## Objectif

Ce projet implémente un modèle de **génération automatique de légendes d'images** (*image captioning*) à partir d'une image en entrée. Le modèle produit une description textuelle en anglais en s'appuyant sur un mécanisme d'**attention visuelle** qui lui permet de se concentrer sur les régions pertinentes de l'image mot par mot.

**Exemple :**
> Image d'un chien → *"a brown dog is running through the grass."*

---

## Architecture

Le pipeline est composé de deux modules :

```
Image (224×224)
    │
    ▼
┌─────────────────────────────┐
│   Encodeur CNN (VGG19)      │  Feature maps : (14 × 14 × 512)
│   Pré-entraîné ImageNet     │  Information spatiale conservée
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│   Décodeur LSTM + Soft Attention        │
│                                         │
│  Pour chaque mot généré :               │
│   1. Calcul des poids α (attention)     │
│   2. Vecteur de contexte pondéré ẑₜ    │
│   3. Prédiction du mot suivant          │
└─────────────────────────────────────────┘
    │
    ▼
Caption : "a dog is running in the grass ."
```

### Mécanisme d'attention (Soft Attention)

À chaque pas de décodage *t*, les poids d'attention αₜᵢ sont calculés par :

```
eₜᵢ = f_att(aᵢ, hₜ₋₁)
αₜᵢ = softmax(eₜᵢ)
ẑₜ  = Σᵢ αₜᵢ · aᵢ
```

Les poids α sont **interprétables visuellement** : on peut superposer la carte d'attention à l'image pour voir sur quelle région le modèle se focalise lors de la génération de chaque mot.

---

## Datasets

| Dataset | Images | Captions / image | Utilisation |
|---|---|---|---|
| **Flickr8k** | 8 091 | 5 | Implémentation principale (ce projet) |
| Flickr30k | 31 783 | 5 | Mentionné dans le papier original |
| MS COCO | ~330 000 | 5 | Mentionné dans le papier original |

**Split utilisé (Flickr8k) :**
- Train : 6 000 images
- Validation : 1 000 images
- Test : 1 091 images
- Vocabulaire : **3 006 mots** (fréquence minimale = 5)

---

## Modèles implémentés

### 1. SAT — Show, Attend and Tell *(papier principal)*
- Encodeur : **VGG19** (feature maps 14×14×512)
- Décodeur : **LSTM Cell** avec **Soft Attention**
- Régularisation doublement stochastique : `λ · Σᵢ(1 - Σₜ αₜᵢ)²`
- Paramètres entraînables (décodeur) : **7 544 767**

### 2. Baseline — Show and Tell *(Vinyals et al., 2014)*
- Encodeur : VGG19 (partagé, figé)
- Décodeur : LSTM sans attention (contexte global fixe)
- Sert de référence pour mesurer le **gain apporté par l'attention**

### 3. EncoderResNet *(contribution supplémentaire)*
- Alternative à VGG19 : **ResNet50** (feature maps 14×14×2048)
- Architecture résiduelle plus profonde

---

## Hyperparamètres

| Paramètre | Valeur |
|---|---|
| Dimension embeddings | 512 |
| Dimension LSTM (hidden) | 512 |
| Dimension attention | 512 |
| Dropout | 0.5 |
| Learning rate encodeur | 1e-4 |
| Learning rate décodeur | 4e-4 |
| Alpha_C (régularisation) | 1.0 |
| Gradient clipping | 5.0 |
| Epochs | 5 |
| Batch size | 32 |
| Taille image | 224 × 224 |

---

## Résultats

### Scores BLEU sur le Test Set (200 images)

| Modèle | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|
| **SAT (VGG19 + Soft Attention)** | **0.609** | **0.416** | **0.289** | **0.195** |
| Show and Tell (baseline) | 0.607 | — | — | 0.192 |
| *Papier original (Xu et al., 2015)* | *—* | *—* | *—* | *~0.188* |

**Gain BLEU-4 avec attention : +0.003** | **BLEU-4 obtenu > BLEU-4 du papier original**

### Évolution de la loss (5 epochs)

| Epoch | Train Loss | Val Loss | Val BLEU-4 |
|---|---|---|---|
| 1 | 4.727 | 4.076 | 0.039 |
| 2 | 3.945 | 3.785 | 0.029 |
| 3 | 3.664 | 3.655 | 0.030 |
| 4 | 3.482 | 3.583 | 0.029 |
| 5 | 3.342 | 3.548 | 0.025 |

---

## Analyse d'erreurs *(inspirée de Liu & Brailsford, 2023)*

Les erreurs les plus fréquentes dans les captions générées sont :

| Catégorie d'erreur | Description |
|---|---|
| **Objet incorrect** | Le modèle prédit un mauvais objet (ex: *"dog"* au lieu de *"cat"*) |
| **Action incorrecte** | Le verbe généré est inexact (ex: *"running"* au lieu de *"swimming"*) |
| **Objet omis** | Un élément important de l'image n'est pas mentionné |
| **Détection de genre** | Confusion homme/femme |
| **Détection de scène** | Mauvaise identification du lieu |

**Top noms incorrects :** shirt, man, woman, bench, field  
**Top verbes incorrects :** stand, jump, run

---

## Structure du projet

```
deepl/
├── Show_Attend_Tell_VSCode.ipynb   # Notebook principal (18 cellules)
├── captions.txt                    # Légendes Flickr8k (image → 5 captions)
├── Images/                         # 8 091 images Flickr8k (.jpg)
│   ├── 1000268201_693b08cb0e.jpg
│   └── ...
├── outputs/                        # Généré automatiquement à l'exécution
│   ├── sat_checkpoint.pth          # Meilleur modèle SAT (sauvegardé par BLEU-4)
│   ├── baseline_checkpoint.pth     # Modèle baseline Show and Tell
│   ├── training_curves.png         # Courbes loss + BLEU-4
│   ├── attention_viz.png           # Visualisation des cartes d'attention
│   ├── error_analysis.png          # Analyse lexicale des erreurs
│   └── qualitative_comparison.png  # Comparaison SAT vs Baseline
└── README.md
```

---

## Installation et exécution

### Prérequis

```bash
pip install torch torchvision nltk matplotlib Pillow
```

**GPU recommandé** — le notebook détecte automatiquement CUDA :
```
Device utilisé : cuda   # ~30 min/epoch sur GPU
Device utilisé : cpu    # ~2h/epoch sur CPU
```

### Données

1. Télécharger **Flickr8k** depuis Kaggle :
   [https://www.kaggle.com/datasets/adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)

2. Extraire et placer les fichiers :
   ```
   deepl/
   ├── Images/        ← toutes les images .jpg
   └── captions.txt   ← fichier des légendes
   ```

3. Vérifier les chemins dans la **Cellule 2** du notebook (`IMAGE_DIR`, `CAPTION_FILE`).

### Lancement

Ouvrir et exécuter `Show_Attend_Tell_VSCode.ipynb` cellule par cellule :

| Cellule | Contenu |
|---|---|
| 1 | Installation des dépendances, imports |
| 2 | Configuration des chemins |
| 3 | Vocabulaire et chargement des captions |
| 4 | Dataset Flickr8k et DataLoaders |
| 5 | Encodeurs CNN (VGG19 + ResNet50) |
| 6 | Module Soft Attention |
| 7 | Décodeur LSTM avec Attention |
| 8 | Configuration des hyperparamètres |
| 9 | Boucles d'entraînement et de validation |
| 10 | Lancement de l'entraînement SAT |
| 11 | Courbes d'entraînement |
| 12 | Génération de captions (greedy decoding) |
| 13 | Visualisation de l'attention |
| 14 | Évaluation BLEU (test set) |
| 15 | Modèle baseline Show and Tell |
| 16 | Comparaison SAT vs Baseline |
| 17 | Analyse d'erreurs (nouns/verbs) |
| 18 | Comparaison qualitative |

---

## Références

```
[1] Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhudinov, R., Zemel, R., & Bengio, Y.
    "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention."
    ICML 2015, PMLR, pp. 2048–2057.

[2] Liu, H., & Brailsford, T.
    "Reproducing 'Show, Attend and Tell: Neural Image Caption Generation with Visual Attention'."
    J. Phys.: Conf. Ser. 2589 (2023) 012012, ICCEE-2023.
    DOI: 10.1088/1742-6596/2589/1/012012

[3] Vinyals, O., Toshev, A., Bengio, S., & Erhan, D.
    "Show and Tell: A Neural Image Caption Generator."
    CVPR 2015, pp. 3156–3164.

[4] Simonyan, K., & Zisserman, A.
    "Very Deep Convolutional Networks for Large-Scale Image Recognition."
    arXiv:1409.1556, 2014.
```
