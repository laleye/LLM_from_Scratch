import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
st.set_page_config(page_title="Recommandation Mode", layout="wide")

st.title("🛍️ Système de Recommandation Multimodal - Mode")

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv('data/subset_fashion_dataset/products_final.csv')
    return df

@st.cache_data
def load_embeddings():
    embeddings = {}

    # Texte
    embeddings['MiniLM'] = np.load('results/text_embeddings.npy')

    # Image
    embeddings['ViT'] = np.load('results/vit_embeddings.npy')
    embeddings['ResNet-50'] = np.load('results/resnet_embeddings.npy')

    # Multimodal
    embeddings['Concaténation'] = np.load('results/strategy1_multimodal_embeddings.npy')
    embeddings['Moyenne (α=0.5)'] = np.load('results/strategy2_multimodal_embeddings.npy')
    embeddings['CLIP'] = np.load('results/clip_image_embeddings.npy')

    return embeddings

df = load_data()
embeddings = load_embeddings()
image_folder = 'data/subset_fashion_dataset/images'

# Créer mapping ID -> index
id_to_idx = {row['id']: idx for idx, row in df.iterrows()}

# Fonction pour obtenir les recommandations
def get_recommendations(product_id, method, embeddings_dict, df_data, id_to_idx_map, top_k=5):
    """Obtenir les top-K recommandations avec scores"""
    if product_id not in id_to_idx_map:
        return None

    anchor_idx = id_to_idx_map[product_id]

    if method not in embeddings_dict:
        return None

    emb = embeddings_dict[method]

    # Calculer la similarité
    similarities = cosine_similarity([emb[anchor_idx]], emb)[0]

    # Exclure le produit lui-même
    similarities[anchor_idx] = -1

    # Top-K
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_scores = similarities[top_indices]

    results = []
    for idx, score in zip(top_indices, top_scores):
        prod_id = df_data.iloc[idx]['id']
        results.append({
            'id': prod_id,
            'score': score,
            'nom': df_data.iloc[idx]['nom'],
            'categorie': df_data.iloc[idx]['categorie']
        })

    return results

# Sidebar - Sélection du produit
st.sidebar.header("🔍 Sélection du produit")

search_mode = st.sidebar.radio("Mode de recherche", ["Par ID", "Par nom"])

if search_mode == "Par ID":
    product_id = st.sidebar.number_input(
        "ID du produit",
        min_value=int(df['id'].min()),
        max_value=int(df['id'].max()),
        value=int(df['id'].iloc[0]),
        step=1
    )
else:
    product_names = sorted(df['nom'].tolist())
    selected_name = st.sidebar.selectbox("Nom du produit", product_names)
    product_id = int(df[df['nom'] == selected_name]['id'].iloc[0])

# Vérifier que le produit existe
if product_id not in id_to_idx:
    st.error(f"⚠️ Le produit {product_id} n'existe pas dans la base.")
    st.stop()

# Afficher le produit sélectionné
product_info = df[df['id'] == product_id].iloc[0]

st.header("Produit sélectionné")
col1, col2 = st.columns([1, 3])

with col1:
    img_path = os.path.join(image_folder, f"{product_id}.jpg")
    if os.path.exists(img_path):
        st.image(img_path)
    else:
        st.warning("Image non disponible")

with col2:
    st.subheader(product_info['nom'])
    st.write(f"**Catégorie**: {product_info['categorie']}")

    if 'prix' in df.columns:
        st.write(f"**Prix**: {product_info.get('prix', 'N/A')}")

    if 'description' in df.columns:
        desc = product_info.get('description', '')
        if desc and pd.notna(desc):
            with st.expander("Description"):
                st.write(desc)

st.markdown("---")

# Sélection de la méthode
st.sidebar.header("📊 Méthode de recommandation")

method_category = st.sidebar.selectbox("Catégorie", ["Texte seul", "Image seul", "Multimodal"])

if method_category == "Texte seul":
    method = st.sidebar.selectbox("Méthode", ["MiniLM"])
elif method_category == "Image seul":
    method = st.sidebar.selectbox("Méthode", ["ResNet-50", "ViT"])
else:
    method = st.sidebar.selectbox("Méthode", ["Concaténation", "Moyenne (α=0.5)", "CLIP"])

# Afficher les recommandations
st.header(f"Top-5 Recommandations - {method}")

with st.spinner("Calcul des recommandations..."):
    recommendations = get_recommendations(product_id, method, embeddings, df, id_to_idx)

if recommendations:
    # Afficher les 5 recommandations
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        if idx < len(recommendations):
            rec = recommendations[idx]

            with col:
                img_path = os.path.join(image_folder, f"{rec['id']}.jpg")
                if os.path.exists(img_path):
                    st.image(img_path)
                else:
                    st.warning("Image non disponible")

                # Colorer selon la catégorie
                if rec['categorie'] == product_info['categorie']:
                    st.markdown(f"**✅ {rec['nom']}**")
                    st.caption(f"📁 {rec['categorie']}")
                else:
                    st.markdown(f"**❌ {rec['nom']}**")
                    st.caption(f"📁 {rec['categorie']}")

                # Score de similarité
                score_pct = rec['score'] * 100
                st.metric("Similarité", f"{score_pct:.1f}%")
                st.caption(f"#{idx + 1}")
else:
    st.warning("Impossible de calculer les recommandations.")

# Comparaison des méthodes
with st.expander("🆚 Comparaison des méthodes"):
    st.write("Voici les top-3 recommandations pour chaque méthode:")

    comparison_methods = ['MiniLM', 'ResNet-50', 'ViT', 'Concaténation', 'Moyenne (α=0.5)', 'CLIP']

    for meth in comparison_methods:
        recs = get_recommendations(product_id, meth, embeddings, df, id_to_idx, top_k=3)

        if recs:
            st.subheader(f"🔹 {meth}")

            cols = st.columns(3)
            for idx, col in enumerate(cols):
                if idx < len(recs):
                    rec = recs[idx]

                    with col:
                        img_path = os.path.join(image_folder, f"{rec['id']}.jpg")
                        if os.path.exists(img_path):
                            st.image(img_path)

                        if rec['categorie'] == product_info['categorie']:
                            st.markdown(f"**✅ {rec['nom']}**")
                        else:
                            st.markdown(f"**❌ {rec['nom']}**")

                        score_pct = rec['score'] * 100
                        st.caption(f"{score_pct:.1f}%")

# Footer
st.markdown("---")
st.caption("Système de recommandation multimodal pour produits de mode")
