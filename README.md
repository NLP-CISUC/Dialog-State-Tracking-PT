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

+ QA Models:

In the 'Scripts' folder, there are two Question-Answering (QA) Models designed for Dialogue State Tracking (DST) in Portuguese. These models are intended for use with dialogues formatted in MultiWOZ-PT, similar to those in the 'data' folder. A QA model requires two inputs: a question and a context. In our case, the questions were specifically crafted by us, considering the domains and respective slots in MultiWOZ-PT. The context for the QA models is provided by the user's utterances. Given a question and a context, the model generates an answer, which is then used to populate specific slots.

This folder contains two subfolders named after the QA models used: 'QA-Model-BERT-base' and 'QA-Model-T5-base'.

Each model is organized into two further subfolders. In one, the models have access to the annotated intent ('Gold_Intent'), while in the other, they utilize an intent classifier to determine the intent in each user utterance ('Intent_Classifier').

The file names 'QA_BERT/T5.py' indicate that these QA models do not employ post-processing methods. 

In contrast, 'QA_BERT/T5_Lev.py' denotes that both models use the Levenshtein (Lev) method for post-processing, and 'QA_BERT/T5_STS.py' indicates the use of the Semantic Textual Similarity (STS) method for post-processing.

+ Intent Classifier:

In the 'Scripts' folder, you'll find the 'intents_classifier.py' script, designed to train an intent recognition model for dialogues using the MultiWOZ-PT dataset. Two language models, BERTimbau-base (based on BERT), and Albertina-PTPT (based on DeBERTa), were fine-tuned using the transformers library and Hugging Face. Both models were trained with a batch size of 32, a learning rate of 1e−5, and for 5 epochs. The model's performance is evaluated on the test set, considering metrics such as precision, recall, F1-score, and accuracy.

+ Auxiliary Files:

questionsv2.json - Contains the questions used in the QA models, tailored for each domain.

get_questions.py - Retrieves all the questions made within the respective domains and stores them in corresponding lists.

schemaPT.json - This file includes the categorical slots present in the MultiWOZ-PT dataset, along with their possible fillers.

get_slots_en.py - Used for translating a categorical slot from English to Portuguese when the slot is filled in English.


