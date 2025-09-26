import argparse,json
from .features import Perceptron,accuracy
from .core import KernelClassifier
def main():
    parser=argparse.ArgumentParser(description='Train and evaluate a product-state quantum-kernel classifier')
    parser.add_argument('dataset');parser.add_argument('--epochs',type=int,default=20)
    args=parser.parse_args()
    try:
        with open(args.dataset) as f:data=json.load(f)
        x,y=data['train_x'],data['train_y'];tx,ty=data['test_x'],data['test_y']
        model=Perceptron().fit(x,y,args.epochs);baseline=KernelClassifier().fit(x,y)
        print(json.dumps({'test_accuracy':accuracy(model,tx,ty),'baseline_accuracy':accuracy(baseline,tx,ty),'mistakes_per_epoch':model.mistakes,'support_examples':sum(a>0 for a in model.alpha)}))
    except (ValueError,KeyError,OSError,TypeError) as error:parser.error(str(error))
