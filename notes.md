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
- Les Hashtags sont mals supprimés. ex : as #coronavirus-fearing shoppers ==> as fearing shoppers
- Vérifier la cohérance des notes données par les noteurs


- Lemmatisation ?
- GRU ?
- LSTM ?
- Word to vec ? 


- Change from 5 to 3 classes ? Or compare ?


- Add EDA phase (fancy stuff)
- Add some visualisation
- Mettre des titres sur nos graphiques, ***vraiment*** !!!


Liste des modèles : 

- Perceptron
- SGDClassifier
- LinearSVC


- Random forests
- Logistic Regression

- VotingClassifier ? -> pondéré ?


Faire 5 ou 6 modèles "naifs" basés sur le cours 


Un modèle "à la main" avec des LSTM, GRU basé sur les meilleurs de kaggle

