import numpy as np
import math

class ANN:
    layers = []
    w = []
    a = []
    z = []

    def __init__(self, layers):
        self.layers = layers

        for i in range(0, len(layers)-1):
            matrix = np.matrix(np.random.random((layers[i], layers[i+1])))
            self.w.append(matrix)

        self.a = [None] * len(layers)
        self.z = [None] * len(layers)

    def setInput(self, x):
        self.a[0] = x

    def computeZ(self, j):
        self.z[j] = self.a[j-1].dot(self.w[j-1])

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def computeA(self, j):
        matrix = np.matrix(self.z[j])
        for r in range(0,len(matrix.A)):
            for c in range(0, len(matrix.A[0])):
                matrix.A[r][c] = self.sigmoid(matrix.A[r][c])
        self.a[j] = matrix

    def compute(self):
        for i in range(0,len(self.layers)-1):
            self.computeZ(i+1)
            self.computeA(i+1)
        return self.a[len(self.layers)-1]



ann = ANN([2,3,1])
ann.setInput(np.matrix([0.5,0.5]))
print(ann.compute())
