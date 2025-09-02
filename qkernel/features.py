from .core import kernel,KernelClassifier

class Perceptron:
    def fit(self,x,y,epochs=20):
        if not isinstance(epochs,int) or epochs<1:raise ValueError('epochs must be a positive integer')
        KernelClassifier().fit(x,y)
        self.x=[tuple(row) for row in x];self.y=list(y);self.alpha=[0]*len(x)
        self.mistakes=[]
        gram=[[kernel(a,b) for b in self.x] for a in self.x]
        for _ in range(epochs):
            errors=0
            for i in range(len(x)):
                score=sum(a*yj*gram[j][i] for j,(a,yj) in enumerate(zip(self.alpha,self.y)))
                if y[i]*score<=0:self.alpha[i]+=1;errors+=1
            self.mistakes.append(errors)
            if errors==0:break
        return self
    def score(self,point):
        if not hasattr(self,'alpha'):raise ValueError('fit must be called first')
        return sum(a*y*kernel(x,point) for x,y,a in zip(self.x,self.y,self.alpha))
    def predict(self,point):return 1 if self.score(point)>=0 else -1

def accuracy(model,x,y):
    if not x or len(x)!=len(y):raise ValueError('nonempty paired evaluation data required')
    return sum(model.predict(row)==label for row,label in zip(x,y))/len(y)
