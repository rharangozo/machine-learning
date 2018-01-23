import numpy as np
import math

class ANN:
    layers = []
    w = []
    a = []
    z = []
    training = []
    delta = []
    derivates = []

    def __init__(self, layers, trainingSet):
        self.layers = layers

        for i in range(0, len(layers)-1):
            matrix = np.matrix(np.random.random((layers[i], layers[i+1])))
            self.w.append(matrix)

        self.a = [None] * len(layers)
        self.z = [None] * len(layers)
        self.derivates = [None] * len(layers)
        self.delta = [None] * (len(layers) + 1)

        self.training = trainingSet

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

    def sigmoidDet(self, x):
        return math.exp(-x) / (1 + math.exp(-x)) ** 2

    def computeDelta(self, index, y):
        if(index == len(self.layers) - 1):
            matrix = - (y - self.a[len(self.layers)-1])
            print('matrix:')
            print(matrix)
            for r in range(0,len(matrix.A)):
                for c in range(0, len(matrix.A[0])):
                    matrix.A[r][c] = matrix.A[r][c] * self.sigmoidDet(self.z[index].A[r][c])
            print('post matrix:')
            print(matrix)

            self.delta[index] = matrix
            return self.delta[index]
        else:
            matrix = self.delta[index+1] * self.w[index].T
            for r in range(0,len(matrix.A)):
                for c in range(0, len(matrix.A[0])):
                    matrix.A[r][c] = matrix.A[r][c] * self.sigmoidDet(self.z[index].A[r][c])
            self.delta[index] = matrix
            return self.delta[index]

    def computeDeltas(self, y):
        for i in range(1, len(self.layers)):
            self.computeDelta(len(self.layers)-i, y)

    def computeDerivates(self):
        for i in range(0, len(self.layers)-1):
            print('aT:')
            print(self.a[i].T)
            print('delta')
            print(self.delta[i+1])
            self.derivates[i] = self.a[i].T * self.delta[i+1]
            print('deriv')
            print(self.derivates[i])

    def learn(self, scalar):
        for i in range(0, len(self.layers)-1):
            print('deriv')
            print(self.derivates[i])
            print('pre')
            print(self.w[i])
            self.w[i] = self.w[i] + scalar * self.derivates[i]
            print('post')
            print(self.w[i])


# ann = ANN([2,3,2],[])
# ann.setInput(np.matrix([[0.5,0.5],[0.4,0.5]]))
# print(ann.compute())
# print(ann.cost([[0.5,0.5],[0.5,0.5]]))
# print('--')
# print(ann.computeDelta(2, [[0.5,0.5],[0.5,0.5]]))
# print(ann.computeDelta(1, [[0.5,0.5],[0.5,0.5]]))

ann = ANN([1,3,1],[])
ann.setInput(np.matrix([[0.1],[0.8]]))
print(ann.compute())
# print('cost:')
# print(ann.cost([[0.5],[0.5]]))
# print('deltas')
# ann.computeDeltas([[0.8],[0.2]])
# ann.computeDerivates()

for i in range(0,100):
    ann.computeDeltas([[0.8],[0.2]])
    ann.computeDerivates()
    ann.learn(2)
print(ann.compute())

# ann.costFunction([[0.5],[0.5]])
