# Équité et Tokenisation des Langues Africaines (Projet NLP)

Ce projet vise à analyser et démontrer les inégalités de traitement des langues africaines (Swahili, Yoruba, Wolof, Fongbe) par rapport aux langues occidentales (comme le Français ou l'Anglais) par les algorithmes de tokenisation standards tels que BPE (utilisé par OpenAI avec `tiktoken`).

Les résultats de ce projet mettent en évidence la "sur-tokenisation", l'augmentation du coût d'inférence, la fragmentation morphologique (perte de sens), et les problèmes liés au traitement de l'Unicode (comme les diacritiques ou les tons lexicaux).

## 🚀 Fonctionnalités et Analyses Clés

- **Mesure de la fertilité** : Comparaison du nombre de tokens générés par mot entre différentes langues.
- **Analyse de la couverture du vocabulaire** : Évaluation du taux de mots découpés en sous-mots fragmentés (fallback to bytes).
- **Problème de coût d'inférence** : Évaluation financière de l'inégalité de tokenisation (ex: coût API GPT-4).
- **Entraînement de Tokenizer Adapté** : Création d'un tokenizer BPE spécifiquement dédié aux langues africaines (ex: Swahili) afin de réduire la fertilité.
- **Analyses phonétiques et orthographiques** : Problèmes avec les tons lexicaux (Yoruba, Fongbe) et les géminées (Wolof).

## 📁 Structure du Projet

```text
.
├── corpus/
│   └── corpus_multilingue.csv        # Le dataset contenant des phrases en Français, Swahili, Yoruba, Wolof, et Fongbe.
├── models/
│   ├── bpe_euro_large.model/.vocab   # Modèles BPE simulant un entraînement majoritairement européen
│   ├── bpe_multi.model/.vocab        # Modèles BPE multilingues "équilibrés"
│   └── bpe_swahili_adapted.model/.vocab # Modèle BPE adapté spécifiquement au Swahili géré avec sentencepiece
├── results/
│   ├── tokenization_results.csv      # Résultats bruts de tokenisation
│   ├── coverage_stats.csv            # Statistiques sur la couverture et fallback to bytes
│   ├── inference_cost.csv            # Analyse des coûts financiers de la sur-tokenisation
│   ├── adapted_tokenizer_comparison.csv # Comparatifs de performances entre le tokenizer de base et adapté
│   └── planches_tokenisation.json    # Exemples visuels et qualitatifs de la tokenisation (différences de fragmentation)
├── TP_Tokenizers_Langues_Africaines.ipynb # Le Notebook Jupyter principal contenant toute l'analyse, les expériences, et les graphiques de synthèse.
└── train_model.py                    # Script Python pour entraîner le modèle SentencePiece adapté sur le corpus swahili
```

## 🛠️ Explication Pas-à-Pas de la Démarche

1.  **Création du Corpus Multilingue** : 
    Le notebook charge et utilise un corpus structuré dans `corpus/corpus_multilingue.csv`. Ce corpus contient des phrases choisies pour leur diversité morphologique et linguistique dans cinq langues différentes.

2.  **Expérience 1 : Analyse Quantitative de la Sur-tokenisation** :
    Nous évaluons les tokenizers (ici une simulation via `SentencePiece` reflétant le comportement des algorithmes comme `tiktoken`). On compare la **fertilité** (moyenne de tokens par mot) des langues africaines face au français.

3.  **Expérience 2 : Couverture et Fallback to Bytes** :
    Nous vérifions si les caractères (ou glyphes) propres aux langues africaines (comme les lettres pointées au Yoruba ou tons au Fongbe) sont présents dans le vocabulaire de base ou s'ils sont systématiquement scindés en "octets" (bytes). Cela détruit la sémantique et augmente massivement le nombre de tokens.

4.  **Expérience 3 : Création d'un Tokenizer Adapté (Le "Correctif")** :
    Grâce au script `train_model.py`, un modèle `SentencePiece` (`bpe_swahili_adapted.model`) est entraîné spécifiquement sur les extraits swahilis. On l'intègre ensuite dans le notebook pour démontrer que l'on peut ramener la fertilité à des niveaux "normaux" comparables aux langues occidentales.

5.  **Analyse Qualitative et Éthique** :
    La dernière partie du projet (et les graphiques de sortie) détaille comment cette sur-tokenisation impacte économiquement l'adoption de l'IA en Afrique (les factures des API commerciales sont basées sur le nombre de tokens consommé, pénalisant mécaniquement ces langues).

## ⚙️ Installation et Dépendances

Assurez-vous d'avoir Python 3.8+ installé.

1. Clonez ce dépôt ou ouvrez le dossier dans votre terminal.
2. Installez les paquets requis. Le script principal et le notebook reposent notamment sur `pandas`, `sentencepiece`, et `matplotlib` pour les visualisations.

```bash
pip install pandas sentencepiece matplotlib jupyter jupyterlab
```

*Note : L'implémentation de base pour la simulation de `tiktoken` a nécessité l'usage de `SentencePiece` en raison de blocages réseau rencontrés lors du téléchargement des blobs OpenAI.*

## 📊 Pour lancer les analyses

- **Entraînement du modèle.** Si vous souhaitez relancer ou mettre à jour le modèle adapté pour le swahili, exécutez le script d'entraînement :
  ```bash
  python train_model.py
  ```
- **Visualisation et résultats.** Ouvrez le notebook :
  ```bash
  jupyter notebook TP_Tokenizers_Langues_Africaines.ipynb
  ```
  Et exécutez l'ensemble des cellules pour générer les fichiers CSV finaux dans le dossier `results/` ainsi que les graphiques comparatifs.
