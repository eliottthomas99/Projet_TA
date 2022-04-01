https://www.kaggle.com/datatattle/covid-19-nlp-text-classification

https://medium.com/@kocur4d/hyper-parameter-tuning-with-pipelines-5310aff069d6

- tf idf
- Vader
- latin !! (encoding)
- On n'a que 9% de données de test ? Si pas suffisant réduire le train pour avoir plus de test ? 
- Prétraitement des données
- Réfléchir à des modèles
- min and max_df
- spacy & nltk 
- supprimer les @ 
- supprimer les liens ?
- utiliser la norme 2 ou plus ? pour mettre en avant les "grosses" erreurs
- augmentation de données ?
- citer en sources : le mooc, le preprocessing de tweets sur kaggle
- ensembling ?
- https://www.kaggle.com/sreejiths0/efficient-tweet-preprocessing
- changer la fonction d'erreur pour prendre en compte que les classes plus proches sont plus proches  -> peut être simplement mettre des chiffres 
- Se renseigner sur l'attribution originelle des notes. Qu'est ce qui fait qu'un tweet est positif ou très positif ? 
- Version de scikit_learn >= 0.23.2


**- Les Hashtags sont mals supprimés. ex : as #coronavirus-fearing shoppers ==> as fearing shoppers**
piste : tokeniser avec spacy

- Vérifier la cohérance des notes données par les noteurs


- Lemmatisation ?


trucs de luca


Dire dans le rapport que : 
- -> lutilisation de SMOTE n'est pas pertinante ici -> pas de gain en performances mais ajout significatif de temps de fit. (*1.3)
- les pipelines sont toutes indiquées. En effet si on vectorise PUIS on applique le modele classique alors c'est environ 2 à 3 fois plus long.  


tester aussi des hyperparametres de tfidfs ? 



- **Change from 5 to 3 classes ? Or compare ?**  --> compare methods efficiency for each nuber of class

- preprocessing 
- Add EDA phase (fancy stuff)
- Add some visualisation
- Mettre des titres sur nos graphiques, ***vraiment*** !!!


Liste des modèles : 

- Perceptron BAD  (but fast)
- SGDClassifier   GOOD
- LinearSVC   GOOD


- Random forests   BAD
- Logistic Regression  GOOD (but long)

- VotingClassifier ? -> pondéré ?  GOOD (take the best one)




- Entrainer sur tout le train (apres grid search) et test sur données de test  
- Faire un script pour les modèles 


- Sauvergarder les poids dans un fichier json et pas devoir tout refaire à chaque fois DONE

Faire 5 ou 6 modèles "naifs" basés sur le cours 


Un modèle "à la main" avec des LSTM, GRU basé sur les meilleurs de kaggle

- GRU ?
- LSTM ?
- Word to vec ? 





k fold sur les données ??? -> très long de tout faire tourner k fois 


On semble pouvoir préciser l'ensemble de validation lors du fit d'un modèle

One hot encoding sur y ??


SMOTE very good -> preciser dans le rapport que c'est des données synthétiques


TODO : 

- Visualisation
- fancy model + lemmatisation



