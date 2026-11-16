Give Me Some Credit
Projet de machine learning sur le dataset Kaggle Give Me Some Credit.
Le but est de prédire si une personne va avoir un retard de paiement grave (90 jours ou plus) dans les 2 ans. C'est une classification binaire avec des classes très déséquilibrées : environ 6,7 % de défauts seulement.
Tout est dans le notebook GiveMeSomeCredit.ipynb.
Données
J'utilise cs-training.csv (150 000 lignes). Je n'utilise pas cs-test.csv pour évaluer, parce qu'il n'a pas la colonne cible. À la place, j'ai mis de côté 20 % de cs-training.csv comme jeu de test, avec une découpe stratifiée (random_state=42). Les choix de modèles se font uniquement sur les 80 % restants, avec de la validation croisée à 5 plis.
Ce que j'ai fait
Exploration : valeurs manquantes (revenu mensuel et nombre de personnes à charge), valeurs bizarres (codes 96 et 98 dans les retards, âges inférieurs à 18 ans, revenus à 0 ou 1, ratio de dette très grand), histogrammes, taux de défaut par tranche d'âge, corrélations.
Nettoyage : une seule fonction qui met les valeurs aberrantes en valeurs manquantes. Les valeurs manquantes sont ensuite remplacées par la médiane.
Baselines : un modèle qui prédit toujours « pas de défaut » et une régression logistique.
Comparaison de modèles : régression logistique, forêt aléatoire, gradient boosting, puis XGBoost.
Réglage du XGBoost avec une recherche sur grille.
Choix du seuil de classification.
Test final sur les 20 % mis de côté.
Résultats
Validation croisée sur le train :
Modèle	Accuracy	AUC	Average precision
Toujours « pas de défaut »	0,933	0,500	0,067
Régression logistique	0,936	0,814	0,352
Forêt aléatoire	0,936	0,864	0,396
Gradient boosting	0,937	0,863	0,394
L'accuracy n'est pas très utile ici : le modèle qui ne prédit jamais de défaut a déjà 93 %. Je regarde plutôt l'AUC et l'average precision.
Avec le seuil par défaut (0,5), le modèle rate beaucoup de défauts. J'ai donc choisi un seuil de 0,20 avec un graphique précision / rappel / F1. Résultat sur le test avec le XGBoost optimisé :
Seuil	Rappel	Précision	F1
0,5	0,186	0,643	0,288
0,20	0,519	0,412	0,459
L'average precision sur le test est de 0,428.
Lancer le notebook
Il faut numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost et jupyter. Les fichiers CSV doivent être dans le même dossier que le notebook. Ensuite on exécute les cellules dans l'ordre.
