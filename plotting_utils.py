import numpy as np
import matplotlib.pyplot as plt

"""
    Producing a cleaner notebook by abstracting away the plotting.
    This is a custom collection of light wrappers around matplotlib
    for the purpose of the k-nearest neighbor classifier.ipynb
"""

def line(x, **kwargs):
    if "title" in kwargs:
        plt.title(kwargs.pop("title"))
    for key in kwargs:
        plt.plot(x, kwargs[key], label=key)
    plt.legend()

def confusion_matrix(C, k, labels):
    plt.imshow(C, interpolation='nearest')
    plt.title("Confusion Matrix (k={})".format(k))
    tick_marks = range(len(labels))
    plt.yticks(tick_marks, labels, rotation=45)
    plt.ylabel('Groundtruth')
    plt.xticks(tick_marks, labels)
    plt.xlabel('Prediction')
    plt.colorbar()

def scatter(x, y, c=None, xlabel=None, ylabel=None, **kwargs):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(x, y, c=c, **kwargs)

def _generate_plane(X, n):
    MeshH, MeshV = np.meshgrid(np.linspace(X.min(axis=0)[0],
                                           X.max(axis=0)[0], n),
                               np.linspace(X.min(axis=0)[1],                                X.max(axis=0)[1], n))
    # MeshH, MeshV = np.array(MeshH.flat), np.array(MeshV.flat)
    return np.vstack((MeshH.flat, MeshV.flat)).T

def decision_boundary(X, y, clf, xlabel=None, ylabel=None, n=75):
    X_plane = _generate_plane(X, n)
    y_plane = clf.predict(X_plane)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(X_plane[:,0], X_plane[:,1], c=y_plane, marker=".", alpha=.6)
    plt.scatter(X[:,0], X[:,1], c=y, edgecolor="black")
