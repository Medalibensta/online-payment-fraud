"""
Build and execute notebooks/02_fraud_detection.ipynb.

The notebook tells the project story: EDA runs live (fast), while the heavy
model comparisons re-use the metrics CSVs and figures produced by the src/
modules, so the notebook stays quick to execute (<2 min) and the single source
of truth for modelling stays in src/.

Run:
    python notebooks/build_notebook.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NB_PATH = HERE / "02_fraud_detection.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md("""
# Détection de fraude sur paiements en ligne

**Angle décisionnel** — support aux équipes risque : bloquer les transactions
suspectes **sans pénaliser les clients légitimes**, et choisir le seuil de
décision à partir d'un **coût métier** explicite plutôt que d'une convention
statistique.

**Données** — dataset ULB *Credit Card Fraud Detection* (Dal Pozzolo et al.) :
284 807 transactions réelles de cartes européennes (sept. 2013), dont **492
fraudes (0,173 %)**. Les features V1–V28 sont des composantes PCA anonymisées ;
seule `Amount` est brute. Téléchargé de façon reproductible depuis OpenML
(`data_id=1597`), sans compte.

**Plan du notebook**
1. Chargement & déséquilibre extrême
2. EDA — montants, séparabilité des composantes
3. Stratégies de rééquilibrage (comparaison contrôlée)
4. Supervisé vs non-supervisé
5. Seuil de décision coût-sensible
6. Interprétabilité SHAP
7. Conclusions & limites

> La logique lourde vit dans `src/` (un module par étape) ; ce notebook raconte
> l'histoire et affiche les résultats produits par ces modules.
"""),
    code("""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from data import load_raw, TARGET_COL

sns.set_theme(style="whitegrid")
RNG = np.random.default_rng(42)

df = load_raw()
print(f"{len(df):,} transactions, {df[TARGET_COL].sum()} fraudes "
      f"({df[TARGET_COL].mean()*100:.3f} %)")
df.head()
"""),
    md("""
## 1. Un déséquilibre extrême — et pourquoi l'accuracy est piégeuse

577 transactions légitimes pour **une** fraude : un modèle qui répond
« jamais fraude » atteint 99,83 % d'accuracy en ne détectant rien. Toute
l'évaluation de ce projet repose donc sur le couple **précision / rappel**
(PR-AUC en tête), jamais sur l'accuracy.
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts = df[TARGET_COL].value_counts()
axes[0].bar(["légitime", "fraude"], counts.values, color=["#4c72b0", "#c44e52"])
axes[0].set_yscale("log")
axes[0].set_title("Classes (échelle log)")
for i, v in enumerate(counts.values):
    axes[0].text(i, v, f"{v:,}", ha="center", va="bottom")

sns.kdeplot(data=df.assign(LogAmount=np.log1p(df.Amount)), x="LogAmount",
            hue=TARGET_COL, common_norm=False, fill=True, ax=axes[1])
axes[1].set_title("Distribution du montant (log) par classe")
plt.tight_layout()
"""),
    md("""
Les fraudes ont une distribution de montants plus étalée vers le haut, mais le
recouvrement est fort : **le montant seul ne suffit pas**, ce sont les
composantes PCA qui portent le signal.
"""),
    code("""
# Séparabilité des composantes les plus discriminantes (identifiées par SHAP §6)
top_v = ["V14", "V4", "V12", "V10"]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=False)
for ax, v in zip(axes, top_v):
    sns.kdeplot(data=df, x=v, hue=TARGET_COL, common_norm=False,
                fill=True, ax=ax, legend=(v == top_v[0]))
    ax.set_title(v)
plt.suptitle("Composantes PCA les plus séparatrices", y=1.03)
plt.tight_layout()
"""),
    md("""
## 2. Le rééquilibrage change le *seuil*, pas le *classement*

Protocole contrôlé : **même modèle** (régression logistique), on ne fait varier
que la stratégie de gestion du déséquilibre — aucune (baseline), pondération de
classe, sous-échantillonnage, SMOTE, SMOTE+Tomek. Le rééchantillonnage est
appliqué **dans un pipeline imblearn**, donc uniquement sur l'entraînement,
jamais sur le test.
"""),
    code("""
resamp = pd.read_csv(ROOT / "reports" / "resampling_comparison.csv")
display(resamp.round(3))
display(Image(ROOT / "reports" / "figures" / "resampling_pr_curves.png",
              width=640))
"""),
    md("""
**Lecture** — le PR-AUC (qualité du *classement* des transactions) bouge à
peine (~0,69–0,72) : le rééquilibrage ne crée pas d'information. En revanche,
au seuil fixe de 0,5, il déplace radicalement le compromis : la baseline est
précise mais rate 37 % des fraudes ; les variantes rééquilibrées rappellent
~89 % des fraudes au prix de ~1 700–2 500 fausses alertes. **Le vrai levier est
donc le choix du seuil** — traité en §4 avec un coût métier.
"""),
    md("""
## 3. Supervisé vs non-supervisé

Trois modèles supervisés (régression logistique pondérée, forêt aléatoire,
XGBoost avec `scale_pos_weight`) et un détecteur d'anomalies **non supervisé**
(Isolation Forest, entraîné sans étiquettes) évalués sur le même test.
"""),
    code("""
models = pd.read_csv(ROOT / "reports" / "model_comparison.csv")
display(models.round(3))
display(Image(ROOT / "reports" / "figures" / "model_comparison.png", width=900))
"""),
    md("""
**Lecture** — XGBoost domine (PR-AUC **0,86**, précision 89 % / rappel 81 % au
seuil 0,5). L'Isolation Forest est loin derrière en précision (PR-AUC 0,17)
mais son ROC-AUC de 0,95 **sans aucune étiquette** en fait un filet de sécurité
pertinent contre les fraudes *nouvelles*, absentes de l'historique labellisé —
complément, pas concurrent.
"""),
    md("""
## 4. Le seuil de décision comme problème de coût métier

Modèle de coût simple et explicite :

| Événement | Coût |
|---|---|
| Fraude manquée (FN) | montant remboursé de la transaction (min. 50 €) |
| Transaction signalée (TP ou FP) | 5 € de revue analyste |

On balaie le seuil et on cherche le minimum du coût total sur le test.
"""),
    code("""
sweep = pd.read_csv(ROOT / "reports" / "cost_threshold_sweep.csv")
best = sweep.loc[sweep.total_cost.idxmin()]
naive = sweep.iloc[(sweep.threshold - 0.5).abs().idxmin()]
print(f"seuil optimal : {best.threshold:.3f}  ->  {best.total_cost:,.0f} EUR "
      f"({best.n_missed:.0f} fraudes manquées, {best.n_flagged:.0f} signalées)")
print(f"seuil naïf 0.5 : {naive.total_cost:,.0f} EUR "
      f"({naive.n_missed:.0f} manquées, {naive.n_flagged:.0f} signalées)")
display(Image(ROOT / "reports" / "figures" / "cost_vs_threshold.png",
              width=640))
"""),
    md("""
**Lecture** — sans modèle, la fraude du test coûterait ≈ 16 600 €. Le modèle au
seuil naïf ramène ce coût à ≈ 5 200 € (–69 %) ; le seuil optimal (≈ 0,13, bien
en dessous de 0,5) gagne encore ~3 % en acceptant quelques alertes de plus. La
forme plate de la courbe autour de l'optimum est une bonne nouvelle
opérationnelle : le choix exact du seuil est **robuste**.
"""),
    md("""
## 5. Interprétabilité — expliquer chaque alerte (SHAP)

Une équipe risque n'agit pas sur un score opaque. TreeSHAP fournit des
attributions exactes par transaction pour XGBoost.
"""),
    code("""
display(Image(ROOT / "reports" / "figures" / "shap_beeswarm.png", width=620))
display(Image(ROOT / "reports" / "figures" / "shap_waterfall_fraud.png",
              width=620))
"""),
    md("""
**Lecture** — quatre composantes (V14, V4, V12, V10) portent l'essentiel du
signal : des valeurs basses de V14/V12/V10 et hautes de V4 tirent le score vers
la fraude. Le *waterfall* montre la décomposition d'une alerte réelle — la vue
« pourquoi cette transaction ? » d'un analyste. Les features étant des
composantes PCA anonymisées, l'interprétation reste structurelle ; sur données
brutes, on lirait directement commerçant, pays, heure, etc.
"""),
    md("""
## 6. Conclusions & limites

**Conclusions**
- XGBoost + pondération de classe : **PR-AUC 0,86** sur 0,17 % de positifs.
- Le rééquilibrage (SMOTE & co) ne change pas le classement, seulement le
  compromis au seuil fixe — le levier décisionnel est le **seuil**, optimisé
  ici par un coût métier (–69 % de coût vs absence de modèle).
- Le non-supervisé ne remplace pas le supervisé mais couvre les fraudes
  nouvelles ; SHAP rend chaque alerte auditable.

**Limites**
- Deux jours de données d'une seule banque : aucune garantie de généralisation
  temporelle ou géographique ; un déploiement réel exige un réentraînement
  continu (dérive des patterns de fraude).
- Features PCA anonymisées → interprétabilité sémantique limitée.
- Coûts métier illustratifs (5 €/revue, plancher 50 €) ; la courbe de coût
  permet de rejouer l'analyse avec les vrais chiffres d'une équipe risque.
- Le split est aléatoire stratifié ; un split temporel strict (train passé /
  test futur) serait plus dur et plus réaliste.
"""),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = cells
    print(f"[notebook] executing {len(cells)} cells...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(HERE)}})
    client.execute()
    nbf.write(nb, NB_PATH)
    print(f"[notebook] written -> {NB_PATH}")


if __name__ == "__main__":
    main()
