from qkernel.core import *

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

def test_original_reference_cases():
    self_test()
