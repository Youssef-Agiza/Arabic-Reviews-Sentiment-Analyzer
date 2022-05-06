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
                  [np.zeros_like(layer.W,dtype=np.float64), np.zeros_like(layer.b,dtype=np.float64)] 
                  for layer in layers      
                 ]
        self.vs=[ 
                  [np.zeros_like(layer.W,dtype=np.float64), np.zeros_like(layer.b,dtype=np.float64)] 
                  for layer in layers      
                 ]
#         print("vs:",self.vs[0][0])
        
    def update(self,layers,N):
#         print(self.vs[0][0])
        for i in range(len(layers)):
#             print("i:",i)
#             print("vs part1: ",self.vs[i][0])
#             print("beta2:",self.beta2)
            self.ms[i][0]= self.beta1*self.ms[i][0]+(1.0-self.beta1)*layers[i].dW
            self.ms[i][1]= self.beta1*self.ms[i][1]+(1.0-self.beta1)*layers[i].db
            
#             print("before: vs of",i," = ", self.vs[i][0])
            self.vs[i][0]= self.beta2*self.vs[i][0]+(1.0-self.beta2)*np.square(layers[i].dW)
            self.vs[i][1]= self.beta2*self.vs[i][1]+(1.0-self.beta2)*np.square(layers[i].db)
#             print("after: vs of",i," = ", self.vs[i][0])

#             print("vs:",self.vs[i][0])
#             print("eps:", self.eps)
            denDW= np.sqrt((self.vs[i][0] + self.eps))
            denB=(np.sqrt((self.vs[i][1] + self.eps)))
            
            numDW=(-1 * self.alpha * self.ms[i][0])
            numB=(-1 * self.alpha * self.ms[i][1])
                    
            deltaW = np.array(numDW /denDW ,dtype=np.float64)
            deltab = np.array( numB/ denB  ,dtype=np.float64)
        
#             print("deltaW",deltaW)
#             print("deltab",deltab)
            layers[i].W +=  deltaW/N
            layers[i].b +=  deltab/N
#             layers[i].W +=  deltaW/np.sqrt(N)
#             layers[i].b +=  deltab/np.sqrt(N)
        
# class GradientDescent:
#     def __init__(self,alpha):
#         self.alpha=alpha
#     def reset_params(self,layers):
#         pass
#     def update(self,layers,N):
#         for i in range(len(layers)):
#             # layers[i].dW=layers[i].dW/N
#             # layers[i].db=layers[i].db/N
#             layers[i].W = layers[i].W - self.alpha * (layers[i].dW/N)
#             layers[i].b = layers[i].b - self.alpha * (layers[i].db/N)