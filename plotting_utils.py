import numpy as np
import matplotlib.pyplot as plt

"""
Producing a cleaner notebook by abstracting away the plotting.
This is a custom collection of light wrappers around matplotlib
for the purpose of the k-nearest neighbor classifier.ipynb
"""

def line(x, **kwargs):
    """
    Creates a line graph with x as the units on the x-axis and
    y-axis being passed in via keyword-arguments.
    The keyword used is simultaneously the label the line gets
    plotted with.
    Optionally a plot-title can be provided using the 'title'
    keyword.

    Example
    -------
    >>> x = numpy.linspace(0, 10)
    >>> line(x, square=x**2, cube=x**3, title="Example Plot")

    Produces a plot with "Example Plot" as title and x determining
    the x-axis. The lines are a collection of points where the i-th
    point is given by (x[i], square[i]) or (x[i], cube[i]) respectively.
    """
    if "title" in kwargs:
        plt.title(kwargs.pop("title"))
    for key in kwargs:
        plt.plot(x, kwargs[key], label=key)
    plt.legend()
    plt.show()

def confusion_matrix(C, k, labels):
    """
    Plot a confusion matrix 
    """
    plt.imshow(C, interpolation='nearest')
    plt.title("Confusion Matrix (k={})".format(k))
    tick_marks = range(len(labels))
    plt.yticks(tick_marks, labels, rotation=45)
    plt.ylabel('Groundtruth')
    plt.xticks(tick_marks, labels)
    plt.xlabel('Prediction')
    plt.colorbar()
    plt.show()

def scatter(x, y, c=None, xlabel=None, ylabel=None, **kwargs):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(x, y, c=c, **kwargs)
    plt.show()

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
    plt.show()
