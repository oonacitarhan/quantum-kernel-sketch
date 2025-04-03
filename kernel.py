"""Classical evaluation of a product-state quantum feature map. No quantum advantage claimed."""
import math

def kernel(x, y):
    if len(x) != len(y) or not x:
        raise ValueError('feature vectors must have equal positive length')
    return math.prod(math.cos((a-b)/2)**2 for a,b in zip(x,y))

class KernelClassifier:
    """Kernel-weighted class score; fit stores labeled reference examples."""
    def fit(self, x, y):
        if not x or len(x) != len(y) or set(y) != {-1,1}:
            raise ValueError('need examples from classes -1 and +1')
        width = len(x[0])
        if not width or any(len(row)!=width for row in x):
            raise ValueError('inconsistent feature dimensions')
        self.examples = [(tuple(row),label) for row,label in zip(x,y)]
        return self

    def score(self, point):
        groups = {label:[] for label in (-1,1)}
        for x,label in self.examples:
            groups[label].append(kernel(point,x))
        return sum(groups[1])/len(groups[1])-sum(groups[-1])/len(groups[-1])

    def predict(self, point):
        return 1 if self.score(point)>=0 else -1

def self_test():
    assert abs(kernel([0,0],[0,0])-1)<1e-12
    assert kernel([0],[math.pi])<1e-12
    a,b = [.2,.7],[1.3,.1]
    assert kernel(a,b)==kernel(b,a)
    clf = KernelClassifier().fit([[0],[.2],[2.8],[3]],[-1,-1,1,1])
    holdout = [([.1],-1),([2.9],1)]
    correct = sum(clf.predict(x)==y for x,y in holdout)
    assert correct == 2
    # The kernel is a squared inner product of normalized product states.
    # Check a finite Gram matrix through explicit feature vectors.
    points = [[0],[.3],[1.1]]
    for x in points:
        for y in points:
            fx = [math.cos(x[0]/2)**2, math.sqrt(2)*math.cos(x[0]/2)*math.sin(x[0]/2), math.sin(x[0]/2)**2]
            fy = [math.cos(y[0]/2)**2, math.sqrt(2)*math.cos(y[0]/2)*math.sin(y[0]/2), math.sin(y[0]/2)**2]
            assert abs(sum(a*b for a,b in zip(fx,fy))-kernel(x,y))<1e-12
    return correct, len(holdout)

if __name__ == '__main__':
    print('Toy holdout correct/total:', self_test())
    print('Self-tests passed; this tiny dataset is not a generalization benchmark.')
