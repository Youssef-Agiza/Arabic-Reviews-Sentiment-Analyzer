import numpy as np
class AdamOptimizer:
    def __init__(self,beta1,beta2,alpha,eps=10e-8):
#         self.params=params
#         self.n_iter=n_iter
        self.beta1=beta1
        self.beta2=beta2
        self.alpha=alpha
        self.eps=eps
        self.ms=[]
        self.vs=[]
        
    def reset_params(self,layers):
        self.ms=[ 
                  [np.zeros_like(layer.W), np.zeros_like(layer.b)] 
                  for layer in layers      
                 ]
        self.vs=self.ms.copy()
        
    def update(self,layers,N):
        for i in range(len(layers)):
            self.ms[i][0]= self.beta1*self.ms[i][0]*(1.0-self.beta1)*layers[i].dW
            self.ms[i][1]= self.beta1*self.ms[i][1]*(1.0-self.beta1)*layers[i].db
            
            self.vs[i][0]= self.beta1*self.ms[i][0]*(1.0-self.beta2)*np.square(layers[i].dW)
            self.vs[i][1]= self.beta1*self.ms[i][1]*(1.0-self.beta2)*np.square(layers[i].db)
            
            
            deltaW = (-1 * self.alpha * self.ms[i][0]) / (np.sqrt(self.vs[i][0] + self.eps))
            deltab = (-1 * self.alpha * self.ms[i][1]) / (np.sqrt(self.vs[i][1] + self.eps))

            layers[i].W +=  deltaW/N
            layers[i].b +=  deltab/N
        
class GradientDescent:
    def __init__(self,alpha):
        self.alpha=alpha
    def reset_params(self,layers):
        pass
    def update(self,layers,N):
        for i in range(len(layers)):
            # layers[i].dW=layers[i].dW/N
            # layers[i].db=layers[i].db/N
            layers[i].W = layers[i].W - self.alpha * (layers[i].dW/N)
            layers[i].b = layers[i].b - self.alpha * (layers[i].db/N)