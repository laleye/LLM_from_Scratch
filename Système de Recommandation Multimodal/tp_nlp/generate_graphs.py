import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configurer le style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Créer le dossier pour les graphiques
import os
os.makedirs('generated_graphs', exist_ok=True)

# ===== GRAPHIQUE 1 : Comparaison P@5 des méthodes unimodales =====
methods = ['TF-IDF', 'MiniLM', 'ResNet-50', 'ViT']
p5_values = [34.6, 33.8, 40.0, 48.5]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, p5_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
plt.ylabel('Precision@5 (%)', fontsize=12, fontweight='bold')
plt.title('Performance des Méthodes Unimodales', fontsize=14, fontweight='bold')
plt.ylim(0, 60)

# Ajouter les valeurs sur les barres
for bar, val in zip(bars, p5_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val}%', ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('generated_graphs/p5_comparison.png', bbox_inches='tight')
plt.close()
print("✅ Graphique 1 créé : p5_comparison.png")

# ===== GRAPHIQUE 2 : Comparaison Unimodal vs Multimodal =====
methods_all = ['TF-IDF\n(Unimodal T)', 'MiniLM\n(Unimodal T)', 'ResNet-50\n(Unimodal V)',
               'ViT\n(Unimodal V)', 'Concaténation\n(Multimodal)',
               'Moyenne α=0.3\n(Multimodal)', 'CLIP\n(Multimodal)']
p5_all = [34.6, 33.8, 40.0, 48.5, 53.1, 50.8, 51.5]
types = ['Unimodal T', 'Unimodal T', 'Unimodal V', 'Unimodal V',
         'Multimodal', 'Multimodal', 'Multimodal']

plt.figure(figsize=(12, 6))
colors_type = ['#3498db' if 'Unimodal' in t else '#e67e22' for t in types]
bars = plt.bar(methods_all, p5_all, color=colors_type, alpha=0.7,
               edgecolor='black', linewidth=1.5)
plt.ylabel('Precision@5 (%)', fontsize=12, fontweight='bold')
plt.title('Comparaison Unimodal vs Multimodal', fontsize=14, fontweight='bold')
plt.ylim(0, 60)

# Ajouter les valeurs
for bar, val in zip(bars, p5_all):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.grid(axis='y', alpha=0.3)
plt.axhline(y=48.5, color='gray', linestyle='--', alpha=0.5, label='Meilleur unimodal (ViT)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('generated_graphs/unimodal_vs_multimodal.png', bbox_inches='tight')
plt.close()
print("✅ Graphique 2 créé : unimodal_vs_multimodal.png")

# ===== GRAPHIQUE 3 : Courbe de sensibilité α =====
alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
p5_alpha = [50.0, 50.0, 51.5, 50.8, 50.8, 49.2, 45.4, 41.5, 35.4, 34.6, 33.8]
mrr_alpha = [0.728, 0.735, 0.759, 0.804, 0.833, 0.865, 0.827, 0.737, 0.670, 0.636, 0.628]

fig, ax1 = plt.subplots(figsize=(12, 6))

# Courbe P@5
color1 = 'tab:blue'
ax1.set_xlabel('Poids α (texte)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Precision@5 (%)', color=color1, fontsize=12, fontweight='bold')
ax1.plot(alpha_values, p5_alpha, color=color1, marker='o', linewidth=2.5,
         markersize=8, label='Precision@5')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

# Marquer l'optimum P@5
opt_p5_idx = p5_alpha.index(max(p5_alpha))
ax1.plot(alpha_values[opt_p5_idx], p5_alpha[opt_p5_idx], 'r*', markersize=20,
         label=f'Optimum P@5 (α={alpha_values[opt_p5_idx]})')
ax1.text(alpha_values[opt_p5_idx], p5_alpha[opt_p5_idx]+1,
         f'{p5_alpha[opt_p5_idx]}%', ha='center', fontweight='bold', color='red')

# Second axe pour MRR
ax2 = ax1.twinx()
color2 = 'tab:green'
ax2.set_ylabel('MRR', color=color2, fontsize=12, fontweight='bold')
ax2.plot(alpha_values, mrr_alpha, color=color2, marker='s', linewidth=2.5,
         markersize=8, label='MRR', linestyle='--')
ax2.tick_params(axis='y', labelcolor=color2)

# Marquer l'optimum MRR
opt_mrr_idx = mrr_alpha.index(max(mrr_alpha))
ax2.plot(alpha_values[opt_mrr_idx], mrr_alpha[opt_mrr_idx], 'r*', markersize=20,
         label=f'Optimum MRR (α={alpha_values[opt_mrr_idx]})')
ax2.text(alpha_values[opt_mrr_idx], mrr_alpha[opt_mrr_idx]+0.02,
         f'{mrr_alpha[opt_mrr_idx]:.3f}', ha='center', fontweight='bold', color='red')

plt.title('Analyse de Sensibilité au Poids α', fontsize=14, fontweight='bold', pad=20)

# Légendes combinées
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
           bbox_to_anchor=(0.5, -0.15), ncol=2)

plt.tight_layout()
plt.savefig('generated_graphs/alpha_sensitivity.png', bbox_inches='tight')
plt.close()
print("✅ Graphique 3 créé : alpha_sensitivity.png")

# ===== GRAPHIQUE 4 : Distribution des catégories =====
categories = ['Topwear', 'Shoes', 'Bags', 'Bottomwear', 'Watches',
              'Innerwear', 'Jewellery', 'Eyewear']
counts = [50] * 8
colors_cat = sns.color_palette("viridis", 8)

plt.figure(figsize=(10, 6))
bars = plt.barh(categories, counts, color=colors_cat, alpha=0.7,
                edgecolor='black', linewidth=1.5)
plt.xlabel('Nombre de Produits', fontsize=12, fontweight='bold')
plt.title('Distribution des Produits par Catégorie', fontsize=14, fontweight='bold')
plt.xlim(0, 60)

# Ajouter les valeurs
for bar, val in zip(bars, counts):
    plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f'{val}', ha='left', va='center', fontweight='bold')

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('generated_graphs/category_distribution.png', bbox_inches='tight')
plt.close()
print("✅ Graphique 4 créé : category_distribution.png")

print("\n✅ Tous les graphiques ont été générés dans le dossier 'generated_graphs/'")
