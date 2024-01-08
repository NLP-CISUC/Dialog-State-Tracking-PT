# data - MultiWOZ-PT

Portuguese Dialogue Corpus Adapted from MultiWOZ 2.2 Dataset 

The creation of the MultiWOZ-PT dataset was based on the manual adaptation and translation of the test dialogues present in the English MultiWOZ dataset. These dialogues include five services, namely:
+ Attractions
+ Hotels
+ Restaurants
+ Taxis
+ Trains
  
The translation involved converting the sentences uttered by the User and System into Portuguese. 

The adaptation part encompassed adjusting the five Cambridge services present in the test dialogues to align with the existing services in Coimbra. These adapted services can be found in the created database(DataBase - Services), which contains the following files: "attractionsCoimbra_db.json", "hotelsCoimbra_db.json", "restaurantsCoimbra_db.json", and "trainsCoimbra_db.json".

Versions

dialogues_001.json(12/07/2023) -> The First version of the dataset contains 512 test dialogues, 1003 services, and 3240 intents. The dialogues were translated over the period from February to July.

dialogues_002.json (3/10/2023) -> The second version of the dataset contains 488 test dialogues that have been added. It has 6226 intentions. The dialogues were translated from August to October.

# Scripts

In the 'Scripts' folder, there are two Models used for applying Dialogue State Tracking (DST) on Portuguese dialogues that have been translated/adapted.

This folder contains two subfolders named after the QA models used: 'QA-Model-BERT-base' and 'QA-Model-T5-base'.

Each model is organized into two further subfolders. In one, the models have access to the intent directly ('Gold_Intent'), while in the other, they utilize an intent classifier to determine the intent in each user utterance ('Intent_Classifier').

The file names 'QA_BERT/T5.py' indicate that these QA models do not employ post-processing methods. 

In contrast, 'QA_BERT/T5_Lev.py' denotes that both models use the Levenshtein (Lev) method for post-processing, and 'QA_BERT/T5_STS.py' signifies the use of the Semantic Textual Similarity (STS) method for post-processing.

In the 'Scripts' folder, you'll find the 'intents_classifier.py' script, designed to train an intent recognition model for dialogues using the MultiWOZ-PT dataset. Two language models, BERTimbau-base (based on BERT), and Albertina-PTPT (based on DeBERTa), were fine-tuned using the transformers library and Hugging Face. Both models were trained with a batch size of 32, a learning rate of 1e−5, and for 5 epochs. The model's performance is evaluated on the test set, considering metrics such as precision, recall, F1-score, and accuracy.
