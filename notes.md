
## Ressources

https://www.kaggle.com/datatattle/covid-19-nlp-text-classification

https://medium.com/@kocur4d/hyper-parameter-tuning-with-pipelines-5310aff069d6

https://www.kaggle.com/sreejiths0/efficient-tweet-preprocessing

https://www.kaggle.com/code/ludovicocuoghi/twitter-sentiment-analysis-with-bert-roberta -> plein de pistes !! 


citer en sources : le mooc, le preprocessing de tweets sur kaggle

- Version de scikit_learn >= 0.23.2




### A FAIRE

- **- Les Hashtags sont mals supprimés. ex : as #coronavirus-fearing shoppers ==> as fearing shoppers** HONORINE --> DONE
- tester aussi des hyperparametres de tfidfs ?  -> min and max_df ? ELIOTT
- **Change from 5 to 3 classes ? Or compare ?**  --> comparaison classements en terme d'accuracy pour le train et pour le test HONORINE --> DONE
- Faire un script pour les modèles ELIOTT
- requirement.txt pour les imports HONORINE --> DONE
- One hot encoding sur y ?? ELIOTT





### A FIGNOLER

- piste : tokeniser avec spacy ou tensorflow (lemmatizing et fancy) ELIOTT
- Prétraitement des données ---> voir au dessus HONORINE --> DONE
- ensembling ? -> pk pas meilleur que tous les autres, en particulier SVC ? ELIOTT
- Lemmatisation ? (Semble inaproprié mais peut être mieux avec un aute tokenizer ?) ELIOTT
- Add some visualisation HONORINE --> DONE
- Entrainer sur tout le train (si pas de grid search) ELIOTT
- Un modèle "à la main" avec des LSTM, GRU basé sur les meilleurs de kaggle ,  GRU , LSTM ,  Word to vec ? ELIOTT
- meilleure mesure de performance que accuracy --> classification report ?





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





### Dire dans le rapport que : 
- -> lutilisation de SMOTE n'est pas pertinante ici -> pas de gain en performances mais ajout significatif de temps de fit. (*1.3)
- les pipelines sont toutes indiquées. En effet si on vectorise PUIS on applique le modele classique alors c'est environ 2 à 3 fois plus long.  
- Mettre des titres sur nos graphiques, ***vraiment*** !!!
- k fold sur les données ??? -> très long de tout faire tourner k fois 




### Liste des modèles : 

- SGDClassifier   GOOD
- LinearSVC   GOOD
- Random forests   BAD
- Logistic Regression  GOOD (but long)
- Perceptron LESS GOOD  (but fast)
- VotingClassifier ? -> pondéré ?  GOOD (take the best one)




