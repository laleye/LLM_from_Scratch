import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import ViTModel, ViTImageProcessor
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Visual Search - Phase Bonus", layout="wide")

st.title("🔍 Recherche par Image (Visual Search)")
st.markdown("---")

# Charger les données
@st.cache_resource
def load_model_and_data():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Charger ViT
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)
    model.eval()
    
    # Charger données produits
    df = pd.read_csv('data/subset_fashion_dataset/products_final.csv')
    
    # Charger embeddings ViT
    embeddings = np.load('results/vit_embeddings.npy')
    
    return processor, model, embeddings, df, device

processor, vit_model, vit_embeddings, df, device = load_model_and_data()

# Upload image
uploaded_file = st.file_uploader(
    "Uploadez une photo de produit",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(uploaded_file, caption="Image uploadée", use_container_width=True)
    
    with col2:
        with st.spinner("Recherche en cours..."):
            # Charger et prétraiter l'image
            query_image = Image.open(uploaded_file).convert('RGB')
            
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.585, 0.569, 0.589]
                )
            ])
            
            img_tensor = preprocess(query_image).unsqueeze(0).to(device)
            
            # Extraire embedding
            with torch.no_grad():
                outputs = vit_model(pixel_values=img_tensor)
                query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
            
            # Calculer similarité
            similarities = cosine_similarity([query_embedding], vit_embeddings)[0]
            
            # Top 10
            top_indices = np.argsort(similarities)[::-1][:10]
            
        st.success(f"**{len(top_indices)} produits les plus similaires trouvés**")
        st.markdown("---")
        
        # Afficher résultats
        for rank, idx in enumerate(top_indices, 1):
            row = df.iloc[idx]
            score = similarities[idx]
            
            col_img, col_info = st.columns([1, 3])
            
            with col_img:
                img_path = f"data/subset_fashion_dataset/images/{row['id']}.jpg"
                try:
                    st.image(img_path, use_container_width=True)
                except:
                    st.write("Image non disponible")
            
            with col_info:
                st.markdown(f"### {rank}. {row['nom']}")
                st.caption(f"Score de similarité : **{score:.2f}**")
                
                col_meta = st.columns(3)
                with col_meta[0]:
                    st.metric("Catégorie", row['categorie'])
                with col_meta[1]:
                    st.metric("Prix", f"{row['prix']:.2f} FCFA")
                with col_meta[2]:
                    st.metric("Matière", row['matiere'])
            
            st.divider()
