import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, recall_score
from prettytable import PrettyTable
from sklearn.linear_model import LogisticRegression

probs = np.linspace(0, 1, 100)
entropy = -probs * np.log2(probs) - (1 - probs) * np.log2(1 - probs)
gini_impurity = 2 * probs * (1 - probs)
plt.figure(figsize=(8, 6))
plt.plot(probs, entropy, label='Entropy', linewidth=3, linestyle='--')
plt.plot(probs, gini_impurity, label='Gini', linewidth=3, linestyle="--")
plt.xlabel('Probability')
plt.ylabel('Magnitude')
plt.title('Entropy versus Gini Index')
plt.legend()
plt.grid()
plt.show()
#
z = np.linspace(-5, 5, 100)
sigmoid = 1 / (1 + np.exp(-z))
cross_entropy = -np.log(sigmoid)
cross_entropy2 = -np.log(1-sigmoid)
plt.figure(figsize=(8, 6))
plt.plot(sigmoid, cross_entropy, linewidth=3, label='$\int(w)$ if y=0')
plt.plot(sigmoid, cross_entropy2, linewidth=3, linestyle = '--', label='$\int(w)$ if y=1')
plt.xlabel(r'$\sigma(z)$')
plt.ylabel('$\int(w)$')
plt.title('Log-Loss function')
plt.grid()
plt.legend()
plt.show()

df=sns.load_dataset('titanic')
print(df.columns)
print(df.head().to_string())
numerical_features = ['age','sibsp','parch','fare']
X = df[numerical_features]
X.fillna(X.mean(),inplace=True)
y = df['survived']
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=5805, stratify=y)
clf = DecisionTreeClassifier(random_state=5805)
clf.fit(X_train, y_train)
y_train_predicted = clf.predict(X_train)
y_test_predicted = clf.predict(X_test)
print(f'Train accuracy {round(accuracy_score(y_train, y_train_predicted),2)}')
print(f'Test accuracy {round(accuracy_score(y_test, y_test_predicted),2)}')
print(clf.feature_importances_)
tree.plot_tree(clf)
plt.show()
print(X_train.columns)

clf=DecisionTreeClassifier(random_state=5805)
tuned_parameters = [{'max_depth':[1,2,3,4,5,6,7],
'min_samples_split': [2,5,8,10],
'min_samples_leaf':[1,2,4,5,6,7,8],
'max_features':[1,2,3,4],
'splitter':['best','random'],
'criterion':['gini','entropy','log_loss']}]
grid_search = GridSearchCV(clf, tuned_parameters, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)
best_classifier = grid_search.best_estimator_
y_train_pred = best_classifier.predict(X_train)
y_test_pred = best_classifier.predict(X_test)
pre_acc = accuracy_score(y_test, y_test_pred)
print(f'Train accuracy of pre pruned tree {round(accuracy_score(y_train, y_train_pred),2)}')
print(f'Test accuracy of pre pruned tree {round(accuracy_score(y_test, y_test_pred),2)}')
conf_matrix_pre = confusion_matrix(y_test, y_test_pred)
y_prob_pre = best_classifier.predict_proba(X_test)[:, 1]  # Probability of the positive class
roc_auc_pre = roc_auc_score(y_test, y_prob_pre)
recall_pre = recall_score(y_test, y_test_pred)
tree.plot_tree(best_classifier)
plt.show()


path = best_classifier.cost_complexity_pruning_path(X_train,y_train)
alphas = path['ccp_alphas']
# print(alphas)

accuracy_train, accuracy_test = [],[]
for i in alphas:
    clf = DecisionTreeClassifier(random_state=5805, ccp_alpha=i)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    accuracy_train.append(accuracy_score(y_train, y_train_pred))
    accuracy_test.append(accuracy_score(y_test, y_test_pred))
fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(alphas, accuracy_train, marker="o", label="train",
drawstyle="steps-post")
ax.plot(alphas, accuracy_test, marker="o", label="test",
drawstyle="steps-post")
ax.legend()
plt.grid()
plt.tight_layout()
plt.show()

clf = DecisionTreeClassifier(random_state=5805, ccp_alpha=0.006)
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)
post_acc = accuracy_score(y_test, y_test_pred)
conf_matrix_post = confusion_matrix(y_test, y_test_pred)
y_prob_post = clf.predict_proba(X_test)[:, 1]  # Probability of the positive class
roc_auc_post = roc_auc_score(y_test, y_prob_post)
recall_post = recall_score(y_test, y_test_pred)
print(f'Train accuracy of post pruned tree {round(accuracy_score(y_train, y_train_pred),2)}')
print(f'Test accuracy of post pruned tree {round(accuracy_score(y_test, y_test_pred),2)}')
tree.plot_tree(clf)
plt.show()

pt=PrettyTable()
pt.field_names = ['Classifier', 'Accuracy', 'Confusion matrix', 'recall', 'AUC']
pt.add_row(['Pre pruned tree', round(pre_acc,2), conf_matrix_pre.round(2), round(recall_pre,2), round(roc_auc_pre,2)])
pt.add_row(['Post pruned tree', round(post_acc,2), conf_matrix_post.round(2), round(recall_post,2), round(roc_auc_post,2)])
print(pt)

# Calculate ROC curve
fpr_pre, tpr_pre, thresholds_pre = roc_curve(y_test, y_prob_pre)
fpr_post, tpr_post, thresholds_post = roc_curve(y_test, y_prob_post)
# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_pre, tpr_pre, linewidth=2, label="ROC curve of Pre pruned tree (area = {:.2f})".format(roc_auc_pre))
plt.plot(fpr_post, tpr_post, linewidth=2, label="ROC curve of Post pruned tree (area = {:.2f})".format(roc_auc_post))
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve ')
plt.legend(loc="lower right")
plt.show()

log_model = LogisticRegression(random_state=5805)
log_model.fit(X_train,y_train)
y_train_pred = log_model.predict(X_train)
y_test_pred = log_model.predict(X_test)
accuracy_test_log = accuracy_score(y_test, y_test_pred)
accuracy_train_log = accuracy_score(y_train, y_train_pred)
print(f'Accuracy of logistic regression on train set {round(accuracy_train_log,2)}')
print(f'Accuracy of logistic regression on test set {round(accuracy_test_log,2)}')
conf_matrix_log = confusion_matrix(y_test, y_test_pred)
y_prob_log = log_model.predict_proba(X_test)[:, 1]  # Probability of the positive class
roc_auc_log = roc_auc_score(y_test, y_prob_log)
recall_log = recall_score(y_test, y_test_pred)
pt.add_row(['Logistic Regression', round(accuracy_test_log,2), conf_matrix_log.round(2), round(recall_log,2), round(roc_auc_log,2)])
print(pt)

fpr_log, tpr_log, thresholds_log = roc_curve(y_test, y_prob_log)
plt.figure(figsize=(8, 6))
plt.plot(fpr_pre, tpr_pre, linewidth=2, label="ROC curve of Pre pruned tree (area = {:.2f})".format(roc_auc_pre))
plt.plot(fpr_post, tpr_post, linewidth=2, label="ROC curve of Post pruned tree (area = {:.2f})".format(roc_auc_post))
plt.plot(fpr_log, tpr_log, linewidth=2, label="ROC curve of Logisitic regressor (area = {:.2f})".format(roc_auc_log))
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve ')
plt.legend(loc="lower right")
plt.show()