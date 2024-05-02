from sklearn.model_selection import StratifiedKFold
X_train = []
y_train = []
indexes = []

#Load Training Dataset
f = open('occupancy.csv', 'r')
lines = f.read().split('\n')
no_attributes = len(lines[0].split(",")) - 1
for i in range(len(lines)):
  splitted = lines[i].split(",")
  if len(splitted) > 1:
    X_train.append([float(i) for i in splitted[:no_attributes]])
    y_train.append(splitted[no_attributes])
  
# generate 2 class dataset
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# enumerate the splits and summarize the distributions
for _, test_ix in kfold.split(X_train, y_train):
  indexes.append(test_ix)

f = open('occupancy_10fold.csv', 'a')
for i in range(len(indexes)):
  f.write(f'fold{i+1}'+'\n')
  for index in indexes[i]:
    f.write(lines[index]+'\n')
  f.write('\n')
f.close()
