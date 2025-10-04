#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Génère un diff HTML entre raw.txt et clean.txt en surlignant en rouge
les segments présents dans raw.txt mais absents de clean.txt.

- Encodage: force/détecte UTF-8 au mieux, avec normalisation Unicode NFC.
- Granularité par défaut: mots (avec espaces qui suivent), en préservant les sauts de ligne.
- Fuzzy matching optionnel (--fuzzy) avec seuil (--fuzzy-threshold), 
  utilise python-Levenshtein si disponible, sinon difflib.
- Sortie: diff.html avec l'intégralité de raw.txt. Segments supprimés
  enveloppés dans <span class="removed" style="color:red;">…</span>.

Usage:
  python diff_gutenberg.py raw.txt clean.txt [-o diff.html]
  [--granularity word|char|line] [--fuzzy] [--fuzzy-threshold 0.7]
  [--autojunk]  # active l'heuristique accel. de difflib (par défaut désactivée)

Astuce Google Docs:
  L'utilisation d'un <span style="color:red;"> assure l'affichage en rouge.
"""

import argparse
import difflib
import html as htmlmod
import re
import sys
import unicodedata

# Dépendance optionnelle pour un fuzzy matching plus robuste
try:
    import Levenshtein  # pip install python-Levenshtein
except Exception:
    Levenshtein = None


def decode_best(data: bytes) -> str:
    """Décode en UTF-8 si possible, sinon tente cp1252 puis latin-1 (avec remplacement).
    Retourne une str normalisée NFC.
    """
    for enc in ("utf-8-sig", "utf-8"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            text = None
    if text is None:
        for enc in ("cp1252", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                text = None
        if text is None:
            # Dernier recours
            text = data.decode("utf-8", errors="replace")
    # Normalisation NFC pour éviter les divergences d'accents composés vs décomposés
    return unicodedata.normalize("NFC", text)


def read_text(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    return decode_best(data)


def tokenize_word_level(text: str):
    """Tokenisation 'mots' pour diff:
    - On conserve les sauts de ligne comme tokens séparés.
    - Les 'mots' sont des séquences non-blancs + espaces qui les suivent (hors newline).
      => si un mot disparaît, l'espace qui le suit disparaît aussi.
    - Les séquences d'espaces non précédées par un mot sont conservées telles quelles.
    """
    tokens = []
    # Séparer les sauts de ligne
    parts = re.split(r'(\r\n|\r|\n)', text)
    # Unité: tout bloc non-espace + ses espaces (hors newline) OU des espaces seuls (hors newline)
    unit = re.compile(r'[^\s]+(?:[ \t]+)?|[ \t]+')
    for part in parts:
        if part in ("\r\n", "\r", "\n"):
            tokens.append(part)
        elif part:
            for m in unit.finditer(part):
                tokens.append(m.group(0))
    return tokens


def tokenize_line_level(text: str):
    """Granularité ligne: garde les fins de ligne."""
    return text.splitlines(keepends=True)


def tokenize_char_level(text: str):
    """Granularité caractère."""
    return list(text)


def similarity(a: str, b: str) -> float:
    """Score de similarité entre deux tokens (str).
    On compare sur une forme simplifiée (strip + casefold).
    Utilise Levenshtein.ratio si dispo, sinon difflib.SequenceMatcher.ratio.
    """
    a_norm = a.strip().casefold()
    b_norm = b.strip().casefold()
    if not a_norm and not b_norm:
        return 1.0
    if Levenshtein is not None:
        return Levenshtein.ratio(a_norm, b_norm)
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


def refine_replace(a_tokens, i1, i2, b_tokens, j1, j2, removed_mask, threshold=0.7):
    """Dans un bloc 'replace', tente d'identifier des tokens de a (~raw) suffisamment
    proches de tokens de b (~clean) pour NE PAS les marquer comme supprimés.
    Stratégie gloutonne: pour chaque token a, on cherche le meilleur b restant.
    """
    used_b = set()
    for i in range(i1, i2):
        a_tok = a_tokens[i]
        a_cmp = a_tok.strip()
        # On ne tente pas de 'matcher' des newlines ou du vide contre du texte
        if not a_cmp:
            # si b possède aussi au moins un vide, on essaie de l'ignorer
            best_j = None
            best_s = 0.0
            for j in range(j1, j2):
                if j in used_b:
                    continue
                if not b_tokens[j].strip():
                    best_j = j
                    best_s = 1.0
                    break
            if best_s >= threshold and best_j is not None:
                used_b.add(best_j)
                # Considéré comme 'matché', donc pas supprimé
                continue
            # sinon, on laisse la décision ci-dessous (sera supprimé)
        # Cherche le b le plus proche
        best_j = None
        best_s = 0.0
        for j in range(j1, j2):
            if j in used_b:
                continue
            s = similarity(a_tok, b_tokens[j])
            if s > best_s:
                best_s = s
                best_j = j
        if best_j is not None and best_s >= threshold:
            used_b.add(best_j)
            # on le considère conservé
        else:
            removed_mask[i] = True


def compute_removed_mask(a_tokens, b_tokens, fuzzy=False, threshold=0.7, autojunk=False):
    """Retourne un masque booléen de longueur len(a_tokens) indiquant
    quels tokens de 'a' (raw) sont absents de 'b' (clean).
    """
    sm = difflib.SequenceMatcher(a=a_tokens, b=b_tokens, autojunk=autojunk)
    removed = [False] * len(a_tokens)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "delete":
            for i in range(i1, i2):
                removed[i] = True
        elif tag == "replace":
            if fuzzy:
                refine_replace(a_tokens, i1, i2, b_tokens, j1, j2, removed, threshold=threshold)
            else:
                for i in range(i1, i2):
                    removed[i] = True
        elif tag == "insert":
            # insertions dans b: n'impacte pas a (raw) -> rien à marquer
            continue
    return removed


def tokens_to_html(a_tokens, removed_mask):
    """Assemble le HTML final à partir des tokens et du masque de suppression.
    Conserve 100% du texte de raw; les parties supprimées sont colorées en rouge.
    """
    out = []
    i = 0
    n = len(a_tokens)
    esc = htmlmod.escape

    while i < n:
        if removed_mask[i]:
            j = i + 1
            while j < n and removed_mask[j]:
                j += 1
            segment = ''.join(a_tokens[i:j])
            out.append(f'<span class="removed" style="color: red;">{esc(segment)}</span>')
            i = j
        else:
            out.append(esc(a_tokens[i]))
            i += 1

    return ''.join(out)


def build_html_document(inner_html: str, title: str = "Diff raw vs clean") -> str:
    """Construit un document HTML minimal et valide."""
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{htmlmod.escape(title)}</title>
<style>
  body {{ margin: 1.5rem; font-family: "Noto Serif", "Times New Roman", serif; line-height: 1.5; }}
  .legend {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; color: #444; margin-bottom: 1rem; }}
  pre.text {{ white-space: pre-wrap; word-wrap: break-word; }}
  .removed {{ color: red; }}
</style>
</head>
<body>
  <div class="legend">
    Segments présents dans raw.txt mais absents de clean.txt affichés en <span style="color:red;">rouge</span>.
  </div>
  <pre class="text">
{inner_html}
  </pre>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Génère un diff HTML surlignant en rouge ce qui a été retiré de raw.txt.")
    parser.add_argument("raw", help="Chemin vers raw.txt (Gutenberg, avec entête anglais)")
    parser.add_argument("clean", help="Chemin vers clean.txt (version nettoyée)")
    parser.add_argument("-o", "--output", default="diff.html", help="Fichier de sortie HTML (def: diff.html)")
    parser.add_argument("--granularity", choices=["word", "char", "line"], default="word",
                        help="Niveau de granularité du diff (def: word)")
    parser.add_argument("--fuzzy", action="store_true", help="Active la détection floue (similarité)")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.7, help="Seuil de similarité pour --fuzzy (def: 0.7)")
    parser.add_argument("--autojunk", action="store_true",
                        help="Active l'heuristique autojunk de difflib (accélère mais peut réduire la précision)")

    args = parser.parse_args()

    raw_text = read_text(args.raw)
    clean_text = read_text(args.clean)

    # Tokenisation selon la granularité
    if args.granularity == "char":
        a_tokens = tokenize_char_level(raw_text)
        b_tokens = tokenize_char_level(clean_text)
    elif args.granularity == "line":
        a_tokens = tokenize_line_level(raw_text)
        b_tokens = tokenize_line_level(clean_text)
    else:
        a_tokens = tokenize_word_level(raw_text)
        b_tokens = tokenize_word_level(clean_text)

    removed_mask = compute_removed_mask(
        a_tokens, b_tokens,
        fuzzy=args.fuzzy,
        threshold=args.fuzzy_threshold,
        autojunk=args.autojunk
    )

    inner_html = tokens_to_html(a_tokens, removed_mask)
    html_doc = build_html_document(inner_html)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        f.write(html_doc)

    print(f"OK: '{args.output}' généré.")


if __name__ == "__main__":
    main()