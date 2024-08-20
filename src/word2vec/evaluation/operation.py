from gensim.models import Word2Vec

# Charger le modèle Word2Vec pré-entraîné
model = Word2Vec.load("word2vecBest.model")

# Effectuer l'opération d'analogie de mots
result = model.wv.most_similar(positive=["obama"], negative=["trump"], topn=1)

# Afficher le résultat
print("Le mot le plus similaire à 'Obama - trump' est :", result[0][0])
