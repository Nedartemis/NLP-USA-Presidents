from gensim.models import Word2Vec
from scipy.stats import spearmanr

# Charger le modèle Word2Vec pré-entraîné
model = Word2Vec.load("../word2vecBest.model")

# Charger l'ensemble de données WordSim-353
word_pairs = []
with open("wordsim353.csv", "r") as f:
    for line in f.readlines()[1:]:  # Ignorer l'en-tête
        word1, word2, similarity = line.strip().split(",")
        word_pairs.append((word1, word2, float(similarity)))

# Calculer la similarité des mots avec le modèle Word2Vec
model_similarities = []
human_similarities = []
for word1, word2, human_similarity in word_pairs:
    try:
        model_similarity = model.wv.similarity(word1, word2)
        model_similarities.append(model_similarity)
        human_similarities.append(human_similarity)
    except KeyError:
        pass  # Ignorer les paires de mots absents dans le vocabulaire du modèle

# Calculer le coefficient de corrélation de Spearman
spearman_correlation, _ = spearmanr(model_similarities, human_similarities)
print("Coefficient de corrélation de Spearman:", spearman_correlation)
