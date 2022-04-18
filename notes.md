
## Ressources

https://www.kaggle.com/datatattle/covid-19-nlp-text-classification

https://medium.com/@kocur4d/hyper-parameter-tuning-with-pipelines-5310aff069d6

https://www.kaggle.com/sreejiths0/efficient-tweet-preprocessing

https://www.kaggle.com/code/ludovicocuoghi/twitter-sentiment-analysis-with-bert-roberta -> plein de pistes !! 


citer en sources : le mooc, le preprocessing de tweets sur kaggle

- Version de scikit_learn >= 0.23.2
- nltk --> import nltk / nltk.download('wordnet')




### A FAIRE

- **- Les Hashtags sont mals supprimés. ex : as #coronavirus-fearing shoppers ==> as fearing shoppers** HONORINE --> DONE
- tester aussi des hyperparametres de tfidfs ?  -> min and max_df ? ELIOTT --> DONE structure but need training/testing (meaning a LOT of time)
- **Change from 5 to 3 classes ? Or compare ?**  --> comparaison classements en terme d'accuracy pour le train et pour le test HONORINE --> DONE
- Faire un script pour les modèles ELIOTT --> DONE for RNN, no need for more for the others
- requirement.txt pour les imports HONORINE --> DONE
- One hot encoding sur y ?? ELIOTT --> DONE (pour le RNN ok, pour les autres modèles c'est pas possible en fait)





### A FIGNOLER

- piste : tokeniser avec spacy ou tensorflow (lemmatizing et fancy) ELIOTT
- Prétraitement des données ---> voir au dessus HONORINE --> DONE
- ensembling ? -> pk pas meilleur que tous les autres, en particulier SVC ? ELIOTT  --> tester le rnn dedans ? 
- Lemmatisation ? (Semble inaproprié mais peut être mieux avec un aute tokenizer ?) ELIOTT  --> DONE et pas mieux
- Add some visualisation HONORINE --> DONE
- Entrainer sur tout le train (si pas de grid search) ELIOTT
- Un modèle "à la main" avec des LSTM, GRU basé sur les meilleurs de kaggle ,  GRU , LSTM ,  Word to vec ? ELIOTT
- meilleure mesure de performance que accuracy --> classification report ? --> DONE
- Ameliorer le requirement.txt





### A CHERCHER 

- Vader --> se renseigner sur ça pour l'ouverture ELIOTT
- Se renseigner sur l'attribution originelle des notes. Qu'est ce qui fait qu'un tweet est positif ou très positif ? !! -> Vérifier la cohérance des notes données par les noteurs HONORINE --> DONE
    Reponse : les tweets ont été labelés manuellement par l'auteur de la base de données, ainsi il est possible qu'il y ait des erreurs dans l'affectation des notes 
- Add EDA phase (fancy stuff) HONORINE --> = visualisation




### FAIT

- tf idf 
- latin !! (encoding)
- On n'a que 9% de données de test ? Si pas suffisant réduire le train pour avoir plus de test ? 
- spacy & nltk -> pk spacy -> cf tableau comparatif MOOC
- supprimer les @ et les liens
- utiliser la norme 2 ou plus ? pour mettre en avant les "grosses" erreurs
- équilibrage de données SMOTE ?
- preprocessing 
- Sauvergarder les poids dans un fichier json et pas devoir tout refaire à chaque fois
- Faire 5 ou 6 modèles "naifs" basés sur le cours 
- Optuna --> Faire cross validation en faisant la moyenne sur k fit (5 par default)






### Dire dans le rapport que : 
- -> lutilisation de SMOTE n'est pas pertinante ici -> pas de gain en performances mais ajout significatif de temps de fit. (*1.3)
- les pipelines sont toutes indiquées. En effet si on vectorise PUIS on applique le modele classique alors c'est environ 2 à 3 fois plus long.  
- Mettre des titres sur nos graphiques, ***vraiment*** !!!
- k fold sur les données ??? -> très long de tout faire tourner k fois 
- enlever les points du texte car ils sont innutiles mais fréquents. Cela va rendre les chiffres moins  lisible car multipliés par 100 mais c'est ok .




### Liste des modèles : 

- SGDClassifier   GOOD
- LinearSVC   GOOD
- Random forests   BAD
- Logistic Regression  GOOD (but long)
- Perceptron LESS GOOD  (but fast)
- VotingClassifier ? -> pondéré ?  GOOD (take the best one)


# PLAN 

## Intro    HONORINE

    - Contexte
    - Sujet + problématique
    - annonce du plan

## Part 0 : Visualisation : partie indépendante ou l'intégrer au fur et à mesure ? HONORINE

## Part 1 : Preprocessing   

    - Ce qui a été fait et a fonctionné :
        - bla bla    HONORINE
        - bla bli    HONORINE
    - Ce qui a été fait mais n'a pas donné de bons résultats:
        - Lemmatisation (SpaCy and NLTK)      ELIOTT
        - Équilibrage de données avec SMOTE   ELIOTT


## Part 2 : Modélisation

    1) Choix des modèles :
        Présentatin ou Rappel du fonctionnement de chaque modèle + éventuellement résultats attendus
        a) Les modèles sklearn 
            i) Présenter rapidement TFIDF     ELIOTT
            ii) Présenter les Pipelines (vérifier le fait qu'elles sont bcp plus rapide --> NON)    ELIOTT
        b) Le/les "Voting Classifier" (à décaler en "c" si on inclu le RNN dans le "Voting Classifier")    ELIOTT
        c) Le RNN    ELIOTT
    
    2) Recherche d'hyperparamètres : 
        a) GridSearchCV pour les modèles sklearn    ELIOTT
        b) Optuna pour le RNN    ELIOTT


## Part 3 : Analyse des résultats 

    1) Comparer l'efficacité des différents modèles (accuracy et/ou f1-score):
        a) 3 classes VS 5 classes    HONORINE
        b) ...    HONORINE
    2) 5 modèles retenus et pourquoi ?    HONORINE 
        a) Se baser sur les performances (accuracy et/ou f1-score et/ou autre)
        b) Se baser sur le temps d'entrainement --> on exclu probablement Gradient Boosting
        c) Se baser sur la similarité entre modèles : 
            Ne garder que LR ou PER --> comparer les perfs teporelles et pratique et prendre le compromis qui nous semble le meilleur

## Conclusion 

    - Bilan (ce qu'on a appris, ce qu'il faut retenir ...)    HONORINE
    - Limites (qualité du dataset, limite de nos machines pour les perfs ....)    HONORINE
    - Ouverture (Utilisation de Transformers ? Vader ? Word2Vec/Glove ? )    ELIOTT

## Bibliographie

- Les codes kaggle et autres dont on s'est inspiré
- Les articles qu'on a lu pour se documenter, notament pour l'ouverture
- La documentation de certaines library si elles sont "originales" (Optuna, sklearn.Pipeline, tweet-preprocessing, ...)




### TODO

- bibliographie plus exhaustive
- mettre une cellule de markdown pour expliquer les tests perso, et en parler vite fait dans le rapport dans les limites ?
- readme pour le git ?


-  presentation du dataset + exemple de tweet 
-  renommer la partie 'ce qui a fonctionné' 
-  parler de tfidf dans la partie SMOTE car on ne peut pas faire smote directement sur du texte
-  parler de pk il y a 2 voting , surtout le 2 parler des eprfs de GB et RF qui sont moins bonnes.
-  reformuler "c'estt le meme principe que wor2vec"
-  mettre les graphs du RNN dans le rapport





    

