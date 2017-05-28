import numpy  as np
import pandas as pd
import gensim
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)

TRAIN_PATH = "data/train.csv"
SUBMISSION_PATH = "submission/"
TEST_PATH = "data/test.csv"

x1 = []
x2 = []
for q1 in train['question1'].values.astype("str"):
    q1_count = 0.001
    q1_vec = np.zeros((300,),dtype="float64")
    for word in q1.split(" "):
        try:
             q1_vec = np.add(q1_vec,model.word_vec(word))
        except KeyError as e:
            continue
        q1_count += 1
    q1_vec = np.divide(q1_vec,q1_count)
    x1.append(q1_vec)
    
for q2 in train['question2'].values.astype("str"):
    q2_count = 0.001
    q2_vec = np.zeros((300,),dtype="float64")
    for word in q2.split(" "):
        try:
             q2_vec = np.add(q2_vec,model.word_vec(word))
        except KeyError as e:
            continue
        q2_count += 1
    q2_vec = np.divide(q2_vec,q2_count)
    x2.append(q2_vec)

Word2Vec = np.concatenate((x1,x2),axis=1)
y = train['is_duplicate']
print(len(Word2Vec))


X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)
X_test = np.nan_to_num(X_test)
y_test = np.nan_to_num(y_test)
X_train, X_test, y_train, y_test = train_test_split(Word2Vec, y, test_size=0.2, random_state=0)        
clf = MLPClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test) 

eval_data = pd.read_csv(TEST_PATH)
eval_data.head()

x1 = []
x2 = []
for q1 in eval_data['question1'].values.astype("str"):
    q1_count = 0.001
    q1_vec = np.zeros((300,),dtype="float64")
    for word in q1.split(" "):
        try:
             q1_vec = np.add(q1_vec,model.word_vec(word))
        except KeyError as e:
            continue
        q1_count += 1
    q1_vec = np.divide(q1_vec,q1_count)
    x1.append(q1_vec)
    
for q2 in eval_data['question2'].values.astype("str"):
    q2_count = 0.001
    q2_vec = np.zeros((300,),dtype="float64")
    for word in q2.split(" "):
        try:
             q2_vec = np.add(q2_vec,model.word_vec(word))
        except KeyError as e:
            continue
        q2_count += 1
    q2_vec = np.divide(q2_vec,q2_count)
    x2.append(q2_vec)
Word2Vec_test = np.concatenate((x1,x2),axis=1)
eval_data_y = clf.predict_proba(Word2Vec_test)


submission = pd.DataFrame({'test_id':eval_data["test_id"], 'is_duplicate':eval_data_y[:,0]})
submission.to_csv(SUBMISSION_PATH+"baseline_submission1"+'.csv', index=False)