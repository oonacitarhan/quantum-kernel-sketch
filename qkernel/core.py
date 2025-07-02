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

