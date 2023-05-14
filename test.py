from sklearn.metrics import *
import matplotlib.pyplot as plt

y_true = [1, 1, 0, 0]
y_pred = [.8, .7, .2, .1]

pr, rec, t = precision_recall_curve(y_true, y_pred)

plt.scatter(rec, pr)
plt.ylim([0, 1.05])
plt.savefig("prplot.png")
print(t)