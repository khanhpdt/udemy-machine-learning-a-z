import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def visualize_two_feature_classification(features, labels, classifier, 
                                         xlabel='', ylabel=''):
    # plot classification boundary
    X1, X2 = np.meshgrid(
        np.arange(start=features[:, 0].min() - 1,
                  stop=features[:, 0].max() + 1,
                  step=0.01),
        np.arange(start=features[:, 1].min() - 1,
                  stop=features[:, 1].max() + 1,
                  step=0.01))
    plt.contourf(X1, X2,
                 classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    # plot observations
    for i, label in enumerate(np.unique(labels)):
        plt.scatter(features[labels == label, 0], 
                    features[labels == label, 1],
                    c = ListedColormap(('red', 'green'))(i), 
                    label=label)
    plt.legend()
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.show()