import torch
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F

class GeneratorResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_projection=False):
        super().__init__()
        self.use_projection = use_projection
        self.dense = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.LeakyReLU(0.2)
        
        if self.use_projection:
            self.skip = nn.Linear(in_dim, out_dim)
            
    def forward(self, x):
        identity = self.skip(x) if self.use_projection else x
        out = self.act(self.norm(self.dense(x)))
        return out + identity

class GANQueryExpanderGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = GeneratorResidualBlock(364, 256, use_projection=True)
        self.block2 = GeneratorResidualBlock(256, 256, use_projection=False)
        
        # Output stage mapping to LSI space
        self.out_dense1 = nn.Linear(256, 150)
        self.out_norm = nn.LayerNorm(150)
        self.out_act = nn.LeakyReLU(0.2)
        
        self.out_dense2 = nn.Linear(150, 150)
        self.tanh = nn.Tanh()
        
    def forward(self, query_vec, noise, prf_ctx):
        # Concatenate [q || z || prf] -> 364-dim
        x = torch.cat([query_vec, noise, prf_ctx], dim=-1)
        x = self.block1(x)
        x = self.block2(x)
        
        x = self.out_act(self.out_norm(self.out_dense1(x)))
        # Final expanded query vector
        return self.tanh(self.out_dense2(x))

class QueryExpander:
    def __init__(self, use_stemming: bool, use_stopwords: bool, model_dir="../models/pretrained_qegans"):
        """
        Loads the pre-computed canonical space and GAN generator for the specific configuration.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prefix = f"{model_dir}/stem_{use_stemming}_stop_{use_stopwords}"
        
        # Load Canonical Space (Vectorizers + SVD)
        with open(f"{prefix}_space.pkl", "rb") as f:
            space = pickle.load(f)
            self.vectorizer = space['vectorizer']
            self.svd = space['svd']
            self.doc_vectors = space['doc_vectors']
            self.vocab_vectors = space['vocab_vectors']
            self.vocab_terms = self.vectorizer.get_feature_names_out()
            
        # Load Generator
        self.generator = GANQueryExpanderGenerator().to(self.device)
        self.generator.load_state_dict(torch.load(f"{prefix}_model_best.pt", map_location=self.device))
        self.generator.load_state_dict(torch.load(f"{prefix}_model_ema.pt", map_location=self.device))
        self.generator.eval()

    def expand(self, preprocessed_query_str: str, top_k: int = 5, return_all: bool = False):
        """
        Takes the user's query, projects it into the canonical LSI space, 
        generates the expansion vector, and decodes it back to terms.
        """
        # 1. Project query into canonical LSI space
        q_tfidf = self.vectorizer.transform([preprocessed_query_str])
        q_lsi = self.svd.transform(q_tfidf)
        q_norm = q_lsi / (np.linalg.norm(q_lsi) + 1e-8)
        
        # 2. Get PRF Context (Using canonical doc vectors)
        sims = np.dot(self.doc_vectors, q_norm[0])
        top_5_idx = np.argsort(sims)[::-1][:5]
        prf_ctx = np.mean(self.doc_vectors[top_5_idx], axis=0)
        prf_norm = prf_ctx / (np.linalg.norm(prf_ctx) + 1e-8)
        
        # 3. Generate Expansion via GAN
        q_tensor = torch.tensor(q_norm[0], dtype=torch.float32).to(self.device)
        prf_tensor = torch.tensor(prf_norm, dtype=torch.float32).to(self.device)
        noise = torch.zeros(64, dtype=torch.float32).to(self.device) # Deterministic at inference
        
        with torch.no_grad():
            # Generator output is already in 256-dim LSI space
            expanded_vec = self.generator(q_tensor, noise, prf_tensor).cpu().numpy()
            
        # 4. Decode via Cosine Similarity in canonical space
        e_norm = expanded_vec / (np.linalg.norm(expanded_vec) + 1e-8)
        
        # Because vocab_vectors are already L2 normalized, dot product == cosine similarity
        scores = np.dot(self.vocab_vectors, e_norm) 

        query_tokens = preprocessed_query_str.split()
        for i, term in enumerate(self.vocab_terms):
            if term in query_tokens:
                scores[i] = -9999.0  # Force original terms to the bottom of the rank
        
        if return_all:
            top_indices = np.argsort(scores)[::-1]
            return [(self.vocab_terms[i], float(scores[i])) for i in top_indices]
        else:
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [(self.vocab_terms[i], float(scores[i])) for i in top_indices]