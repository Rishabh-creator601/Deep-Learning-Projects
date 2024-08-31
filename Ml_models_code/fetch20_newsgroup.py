from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer 
from sklearn.naive_bayes import MultinomialNB

categories= {'comp.graphics':'Graphics','rec.autos':'Auto',
             'rec.motorcycles':'MotorCycle','rec.sport.baseball':'Baseball'
             ,'rec.sport.hockey':'Hockey',
             'sci.space':'Space',
             'talk.religion.misc': 'Religion'}

train = fetch_20newsgroups(subset='train',categories=categories.keys(),shuffle=True,random_state=5)
test = fetch_20newsgroups(subset='test',categories=categories.keys(),shuffle=True,random_state=5)

x_test = test.data
y_test = test.target

x = train.data 
y = train.target

cv = CountVectorizer()
train_cv= cv.fit_transform(x)

tfidf = TfidfTransformer()
train_tf = tfidf.fit_transform(train_cv)

model = MultinomialNB()
model.fit(train_tf,y)

def initialize(data,details=False,passing=3):
  p1 = cv.transform(data)
  p2 = tfidf.transform(p1)
  p3 = model.predict(p2)
  s1 = p1.shape 
  s2 = p2.shape 
  s3 = p3.shape
  shapes = [s1,s2,s3]



  if details == True:  
    return shapes,p3
  if passing==1:
    return p1
  if passing==2:
    return p2
  if passing==3:
    return p3
  if details==False :
    return p3
     
  

print(model.score(initialize(x_test,passing=2),y_test))

predictions = initialize(x_test,passing=3)

for sent , cat in zip(x_test[:5],predictions[:5]):
  print(sent)
  print('{',categories[train.target_names[cat]],'}')
  

