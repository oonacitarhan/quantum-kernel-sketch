from qkernel.features import Perceptron,accuracy
def test_training():
    model=Perceptron().fit([[0],[.2],[2.8],[3]],[-1,-1,1,1])
    assert accuracy(model,[[.1],[2.9]],[-1,1])==1
    assert model.mistakes[-1]==0
def test_dimension_rejection():
    try:Perceptron().fit([[0],[1,2]],[-1,1])
    except ValueError:return
    raise AssertionError('ragged data accepted')
def test_before_fit():
    try:Perceptron().predict([0])
    except ValueError:return
    raise AssertionError('untrained model accepted')
def test_epoch_validation():
    try:Perceptron().fit([[0],[3]],[-1,1],0)
    except ValueError:return
    raise AssertionError('zero epochs accepted')
