La phrase `query` est transformée en vecteur avec `embed_sentence(query)`.
Puis on divise le vecteur par sa norme euclidienne : `q = q / np.linalg.norm(q)`.

Cela permet de faire ensuite **un calcul des similarités cosinus** par un produit matriciel ( `sims = emb_matrix_normed @ q` ).

`sims[i] ` correspond au score de similarité entre query et la phrase `i`.
