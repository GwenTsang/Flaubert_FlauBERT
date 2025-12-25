# Projets similaires :

https://github.com/Lidan0241/python_tp

https://github.com/Dinarque/INALCO_Inalco_M2_langage_de_scripts_2024_eleves/blob/main/Corrections/TP1_de_Flaubert_a_FlauBERT_correction_vf.ipynb (corrections et codes plus simples, des comparaisons seraient intéressantes).

# Détails spécifiques sur les fichiers

[k_most_similar_phrases.py](https://github.com/GwenTsang/Flaubert_FlauBERT/blob/main/k%20most%20similar%20phrases.py) permet de renvoyer les $k$ phrases les plus similaires à une requête textuelle donnée (`query`) à partir d’un ensemble pré-encodé de phrases (`sentences`) représentées par leurs vecteurs d’embedding (`emb_matrix_normed`).

La phrase `query` est transformée en vecteur avec `embed_sentence(query)`.
Puis on divise le vecteur par sa norme euclidienne : `q = q / np.linalg.norm(q)`.

Cela permet de faire ensuite **un calcul des similarités cosinus** par un produit matriciel ( `sims = emb_matrix_normed @ q` ).

`sims[i] ` correspond au score de similarité cosinus entre query et la phrase `i` . Ce score est un nombre flottant dans l'intervalle réel $[-1,+1]$.
1 signifie que les vecteurs ont la même direction, 0 signifie qu'ils sont orthogonaux et -1 signifie qu'ils sont opposés.

# Voir aussi

https://github.com/Dinarque/Flaubert
