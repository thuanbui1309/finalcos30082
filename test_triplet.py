import torch
import numpy as np
import json
from src.core.face_verifier import FaceVerifier
verifier = FaceVerifier(model_type='triplet', weights_path='weights/face_metric_learning.pth', device='cpu')
db = json.load(open('face_db/metadata.json'))
me = [x for x in db if 'Bui' in x['name']][0]
my_id = me['face_id']
my_emb = np.load(f"face_db/embeddings/{my_id}_triplet.npy")

print("My Triplet Embedding (first 10):", my_emb[:10])

# check other IDs
others = [x for x in db if 'Bui' not in x['name']][:2]
for o in others:
    o_emb = np.load(f"face_db/embeddings/{o['face_id']}_triplet.npy")
    sim = verifier.compare(my_emb, o_emb, 'cosine')
    print(f"Cosine with {o['name']} = {sim:.4f}")

    euc = verifier.compare(my_emb, o_emb, 'euclidean')
    print(f"Euclidean with {o['name']} = {euc:.4f}")
