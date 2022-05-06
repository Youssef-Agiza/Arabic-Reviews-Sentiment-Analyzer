import numpy as np
import pickle


class Layer:
    
    ### activations
    def _relu(self,z):
        return np.maximum(0,z)
    def _diff_relu(self,z):
        dZ=np.array(z,copy=True)
        dZ[dZ<=0]=0
        dZ[dZ>0]=1
        return dZ
    
    def _identity(self,z):
        return z
    
    def _identity_diff(self,z):
        return np.ones_like(z)
    
    def _sigmoid(self,z):
        return (1/(1+np.exp(-1*z)))

    def _diff_sigmoid(self,z):
        return self._sigmoid(z)*(1-self._sigmoid(z))
    
    def _softmax(self,z):
        expZ= np.exp(z-np.max(z))
        return expZ/expZ.sum(axis=0, keepdims=True)
    def _diff_softmax(self,z):
        pass

    
    ###########

    def __init__(self,n_input,n_output, activation="identity",name=None):
        self.n_output= n_output
        self.n_input= n_input
        self.name= name
        self.reset_params()
        
        if activation == "identity":
            self.activation = self._identity
            self.diff_act= self._identity_diff
        
        elif activation == "sigmoid":
            self.activation = self._sigmoid
            self.diff_act= self._diff_sigmoid
            
        elif activation == "softmax":
            self.activation=self._softmax
            self.diff_act=self._diff_softmax
        elif activation =="relu":
            self.activation=self._relu
            self.diff_act=self._diff_relu
        
            
        
    def reset_params(self): 
        self.W= np.random.randn(self.n_output,self.n_input)*np.sqrt(2/self.n_input)
        self.b= np.random.randn(self.n_output,1)*np.sqrt(2/self.n_input)

        self.dW= np.zeros_like(self.W)
        self.db= np.zeros_like(self.b)
        
        self.Z= None
        self.Ai = None

    def print_shapes(self):
        print("W: ",self.W.shape)
        print("b: ",self.b.shape)
    
    def forward(self,Ai): #data dim 

        z =  np.add((self.W @ Ai),self.b)
        A = self.activation(z)

        
        self.Z = z
        self.Ai = Ai
        return A
    
    
    def backward(self,inp):
        
#         print("input shape: ",end='')
#         print(inp.shape)
       
        act_diff = self.diff_act(self.Z)
#         print("act_diff shape: ",end='')
#         print(act_diff.shape)
        
        tmp = inp * act_diff
#         print("tmp shape: ",end='')
#         print(tmp.shape)
        
        bet = tmp @ self.Ai.T # vector of 1s
#         print("bet shape: ",end='')
#         print(bet.shape)
        
        
        e = np.ones((self.Ai.shape[1],1))
        db = tmp @ e
#         print("db shape: ",end='')
#         print(db.shape)
        self.dW = (self.dW + bet)
#         print("dw:",self.dW.shape,"\nlen:",len(self.dW))
        self.db = self.db + db
        
        
        return self.W.T @ tmp
    
    def print_weights(self):
        print("\n###################")
        if(self.name):
            print("name: ",self.name)
        print("dW: ",self.dW, "W: ",self.W)
    
    def zeroing_delta(self):
        self.dW= np.zeros_like(self.W)
        self.db= np.zeros_like(self.b)


class NN:
    
    ########
    ## losses
    def _MSE(self,y,yhat):
        a=np.square(yhat-y)
        a=np.sum(a)
        b= 1/(2*y.shape[1])
        return a*b

    ## diff losses
    def _diff_MSE(self,y,yhat,X):
        return (yhat-y)
    
    def _binary_cross_entropy(self,y,yhat):
        arr= -(y*np.log(yhat)+(1-y)*np.log(1-yhat))
        return arr.mean()
        
    def _diff_binary_cross_entropy(self,y,yhat,X):
        dl_dyhat= -(y/(yhat) - (1-y)/(1-yhat))
        return dl_dyhat
 
    
    #########
    
    def __init__(self,optimizer=None,loss="binary_cross"):
        self.layers = []
        self.optimizer=optimizer
        self.loss_name=loss
        self.initialize_loss()
    
   
    def initialize_loss(self): 
        if(self.loss_name=="binary_cross"):
            self.loss=self._binary_cross_entropy
            self.loss_diff=self._diff_binary_cross_entropy
        elif self.loss_name=="MSE":
            self.loss=self._MSE
            self.loss_diff=self._diff_MSE
        
    def reset_layers(self):
        for layer in self.layers:
            layer.reset_params()
    
 
    
    def forward(self,x_train):
        a=x_train
        for layer in self.layers:
            a = layer.forward(a)
        return a
    
    def backward(self,input):
        gd = input
        for layer in self.layers[::-1]:
            gd = layer.backward(gd)
            
    def add_layer(self,n_input,n_output, activation="identity",name=None):
        self.layers.append(Layer(n_input,n_output, activation=activation,name=name))
    
    def batch(self,x,y,batch_size):
        x= x.copy()
        y=y.copy()
        reminder= x.shape[0] % batch_size


        for i in range(0,x.shape[0],batch_size):
            yield (x[i:i+batch_size],y[i:i+batch_size])
        
        if reminder !=0:
            yield (x[x.shape[0]-reminder:],y[x.shape[0]-reminder:] )
    
    def fit(self, x_train,y_train,validation_data=None,batch_size=32, epochs=5): #data dim is MxN .. M no of examples.. N no of dimension
        
        M = x_train.shape[0]

        no_of_batches= np.ceil(M/batch_size)
        if(validation_data):
            x_valid=validation_data[0]
            y_valid=validation_data[1]
        
        
        for i in range(epochs):
            
            print("Epoche {}/{}".format(i+1,epochs))
            self.optimizer.reset_params(self.layers)
            batches=self.batch(x_train,y_train,batch_size)
            losses=[]
            j=0
            for cur_x,cur_y in batches:
                
                cur_x=cur_x.T
                cur_y=cur_y.T
                
                y_hat= self.forward(cur_x)

                dl_dyhat = self.loss_diff(cur_y,y_hat,self.layers[-1].Ai)
                loss=self.loss(cur_y,y_hat)
                
                losses.append(loss)

                self.backward(dl_dyhat)
                
                if batch_size==1:
                    N= M
                else:
                    N=cur_x.shape[-1]
                
                self.optimizer.update(self.layers,N)

                # zeroing deltas
                for layer in self.layers:
                    layer.zeroing_delta()
                j+=1
                
            if validation_data:
                y_hat_val = self.forward(x_valid.T)
                loss_val= self.loss(y_valid.T,y_hat_val)
                print("val_loss: {}....".format(loss_val),end=" ")
                ######
                #calc metrics
            avg_loss= np.array(losses).mean()
            if(avg_loss<0.05):
                print("Stopping early because loss converged to a small number")
                print("losses avg=",avg_loss)
                break
            else: print("losses avg=",avg_loss)

                

        print("Finished....") 
            
            
        

    
    def predict(self,x_test): #data dim is NxD .. N no of examples.. D no of dimension
        print("x_test:", x_test.shape)
        y_hat= self.forward(x_test.T).T
        print("yhat: ",y_hat)
#         return y_hat
        print(y_hat)
        y_hat[y_hat>0.5]=1
        y_hat[y_hat<=0.5]=0
        return y_hat
                    
    def print_weights(self):
        for i in range(len(self.layers)):
            print("layer i= ",i,end=" ")
            self.layers[i].print_weights()
    def print_shapes(self):
        for layer in self.layers:
            layer.print_shapes()
    
    def save_model(self,path):
        model=[self.layers,self.optimizer,self.loss]

        file=open(path,"wb")
        print("dumped model: ",model)

        pickle.dump(model,file)

        file.close()

    def load_model(self,path):
        file=open(path,"rb")
        print("File: ",file)
        model=pickle.load(file)

        file.close()
        print("loaded model: ",model)
        
        self.layers,self.optimizer,self.loss=model
        self.initialize_loss()

    