import math



import  numpy as np


def passFunc(x):
    return  x

def passDerivative(x):
    return 1
class Neuron:
    def __init__(self):
        self.actFunc=passFunc
        self.derivFunc=passDerivative
        self.frontNeighbours=[]
        self.backNeighbours=[]
        self.weights=np.array([])
        self.derivWeights=np.array([])
        self.derivNeurons = np.array([])
        self.bias=0
        self.input=1
        self.output=0
        self.derivOutput=0
    def initializeConnected(self):
        length=len(self.backNeighbours)

        self.derivWeights=np.zeros(length)
        self.derivNeurons=np.zeros(length)
    def updateInput(self):
        if len(self.backNeighbours)==0:
            return
        neuronsVector=np.array([neuron.output for neuron in self.backNeighbours])
        self.input=np.dot(self.weights,neuronsVector)+self.bias

    def updateOutput(self):
        self.output=self.actFunc(self.input)
    def updateDerivOutput(self):
        if(len(self.backNeighbours)==0):
            return
        self.derivOutput=self.derivFunc(self.input)
    def updateDerivWeights(self):

        if (len(self.backNeighbours) == 0):
            return

        for i in range(len(self.backNeighbours)):
            value=self.backNeighbours[i].output
            self.derivWeights[i]=self.derivOutput*value


    def updateDerivNeurons(self):
        if (len(self.backNeighbours) == 0):
            return

        for i in range(len(self.backNeighbours)):
            value=self.weights[i]
            self.derivNeurons[i]=value*self.derivOutput



    def update(self):
        self.updateInput()
        self.updateDerivOutput()
        self.updateDerivNeurons()
        self.updateDerivWeights()
        self.updateOutput()
    def connect(self,neuron,randomWeight=True):
        self.backNeighbours.append(neuron)
        neuron.frontNeighbours.append(self)
        if randomWeight==True:
            self.weights=np.append(self.weights,np.random.uniform(-1,1))
        else:
            self.weights=np.append(self.weights,1)
class Layer:
    def __init__(self,neuronCount,func=passFunc,derivFunc=passDerivative):
        self.count=neuronCount
        self.neurons=self.initializeNeurons(count=neuronCount,actFunc=func,derivFunc=derivFunc)
        self.outputVector=[]
        self.derivNeuronsVectors=np.array([])
        self.derivWeightsVectors=np.array([])
        self.derivBiasesVector=np.array([])
    def takeInput(self,inputVector):

        assert len(inputVector)==self.count , "WRONG INPUT DIMENSIONS @Layer.takeInput"

        for i in range(self.count):
            self.neurons[i].input = inputVector[i]

    def initializeNeurons(self,count,actFunc,derivFunc):
        neurons = []
        for i in range(count):
            neuron = Neuron()
            neuron.actFunc=actFunc
            neuron.derivFunc=derivFunc
            neurons.append(neuron)
        return neurons
    def initializeConnected(self):
        for i in range(self.count):
            self.neurons[i].initializeConnected()
    def connect(self,prevLayer):
        for i in range(self.count):
            for j in range(prevLayer.count):
                self.neurons[i].connect(prevLayer.neurons[j],randomWeight=True)
        self.initializeConnected()

    def updateOutputVector(self):
        self.outputVector = np.array([neuron.output for neuron in self.neurons])
    def updateDerivNeuronsVectors(self):
        self.derivNeuronsVectors=np.array([neuron.derivNeurons for neuron in self.neurons])
    def updateDerivWeightsVectors(self):
        self.derivWeightsVectors=np.array([neuron.derivWeights for neuron in self.neurons])
    def updateDerivBiasesVector(self):
        self.derivBiasesVector=np.array([neuron.derivOutput for neuron in self.neurons])
    def update(self):
        for i in range(self.count):
            self.neurons[i].update()
        self.updateOutputVector()
        self.updateDerivNeuronsVectors()
        self.updateDerivWeightsVectors()
        self.updateDerivBiasesVector()

class NeuralNetwork:
    def __init__(self,*layers):
        self.layerCount=len(layers)
        self.Layers=layers
        self.neuronCount = 0
        self.parameterCount = 0
        self.learningRate=0.01
        self.velocityMomentum=0.9
        self.momentMomentum=0.9
        self.initializeLayers()
        self.getInfo()
        self.gradientMatrix = self.initializeGradientMatrix()
        self.matrixIndex=self.parameterCount-1
        self.X_data=[]
        self.Y_data=[]
        self.lossGradient=[]
        self.velocityVector=[]
        self.momentVector=[]
        self.batchSize=100
        self.MSELoss=0
        self.epsilon=0.000000001
    def initializeLayers(self):
        for i in range(1 , self.layerCount , 1):
            self.Layers[i].connect(self.Layers[i-1])
            self.Layers[i].initializeConnected()
    def getInfo(self):
        for i in range(self.layerCount):
            self.neuronCount += self.Layers[i].count
            for j in range(self.Layers[i].count):
                if i != 0:
                    self.parameterCount += len(self.Layers[i].neurons[j].weights) + 1  # 1 for bias

    def propagateForward(self,inputVector):
        self.Layers[0].takeInput(inputVector)
        for i in range(self.layerCount):
            self.Layers[i].update()
    def initializeGradientMatrix(self):
        count=self.Layers[self.layerCount-1].count
        matrix=[]
        for i in range(count):
            vector=[9 for _ in range(self.parameterCount)]
            matrix.append(vector)

        return matrix
    def extractParamters(self):
        parameters = np.zeros(self.parameterCount)
        idx = self.parameterCount-1
        for i in range(self.layerCount-1,0,-1):
            for j in range(len(self.Layers[i].neurons)):
                for k in range(len(self.Layers[i].neurons[j].weights)):
                    parameters[idx]=self.Layers[i].neurons[j].weights[k]
                    idx -= 1
                parameters[idx]=self.Layers[i].neurons[j].bias
                idx -= 1

        return parameters

    def updateParamters(self,newParametersVector):
        assert len(newParametersVector) == self.parameterCount
        idx = self.parameterCount - 1
        for i in range(self.layerCount - 1, 0, -1):
            for j in range(len(self.Layers[i].neurons)):
                for k in range(len(self.Layers[i].neurons[j].weights)):
                    self.Layers[i].neurons[j].weights[k]=newParametersVector[idx]
                    idx -= 1
                self.Layers[i].neurons[j].bias = newParametersVector[idx]
                idx -= 1
    def lastLayerGradient(self):
        layerIdx = self.layerCount - 1
        neuronsPerLayer = self.Layers[layerIdx].count
        weightsPerNeuron = len(self.Layers[layerIdx].neurons[0].weights)
        parametersToProcess = neuronsPerLayer * (weightsPerNeuron + 1)  # 1 for bias
        parametersProcessed = 0
        counter = 0
        while (parametersProcessed < parametersToProcess):

            for i in range(len(self.Layers[layerIdx].neurons)):
                lowerBound = (weightsPerNeuron + 1) * i
                upperBound = lowerBound + weightsPerNeuron + 1
                if parametersProcessed >= lowerBound and parametersProcessed < upperBound:
                    if counter >= weightsPerNeuron:
                        self.gradientMatrix[i][self.matrixIndex] = self.Layers[layerIdx].neurons[i].derivOutput
                        counter = 0
                    else:
                        self.gradientMatrix[i][self.matrixIndex] = self.Layers[layerIdx].neurons[i].derivWeights[counter]
                        counter += 1
                else:
                    self.gradientMatrix[i][self.matrixIndex] = 0

            self.matrixIndex -= 1
            parametersProcessed += 1
    def firstDeepLayerGradient(self):
        counterCopy=0
        for i , neuron in enumerate(self.Layers[-1].neurons):
            counter = 0
            for j , neighbourNeuron in enumerate(neuron.backNeighbours):
                for k , weightDeriv in enumerate(neighbourNeuron.derivWeights):
                    self.gradientMatrix[i][self.matrixIndex-counter] = weightDeriv*neuron.derivNeurons[j]
                    counter+= 1
                    if k == len(neighbourNeuron.derivWeights)-1:
                        self.gradientMatrix[i][self.matrixIndex-counter]=neighbourNeuron.derivOutput
                        counter += 1
            counterCopy=counter

        self.matrixIndex -= counterCopy

    def secondDeepLayerGradient(self):
        layerIdx=self.layerCount-3
        counterCopy=0
        for i,neuron in enumerate(self.Layers[-1].neurons):
            counter=0
            for j,targetNeuron in enumerate(self.Layers[layerIdx].neurons):
                intermediateNeuronsDerivVector=np.array([neighbourNeuron.derivNeurons[j] for neighbourNeuron in targetNeuron.frontNeighbours])
                intermediateProduct=np.dot(intermediateNeuronsDerivVector,neuron.derivNeurons)
                for k , derivWeight in enumerate(targetNeuron.derivWeights):
                    self.gradientMatrix[i][self.matrixIndex-counter]=derivWeight*intermediateProduct
                    counter+=1
                    if k==len(targetNeuron.derivWeights)-1:
                        self.gradientMatrix[i][self.matrixIndex-counter]=targetNeuron.derivOutput*intermediateProduct
                        counter+=1
            counterCopy = counter
        self.matrixIndex-=counterCopy
    def superDeepLayersGradient(self):
        interLayerIndex_1=self.layerCount-2
        interLayerIndex_2=self.layerCount-3
        m1=np.array([neuron.derivNeurons for neuron in self.Layers[interLayerIndex_1].neurons])
        m1=np.transpose(m1)
        m2=np.array([neuron.derivNeurons for neuron in self.Layers[interLayerIndex_2].neurons])
        m2=np.transpose(m2)
        intermediateJacobian=np.matmul(m2,m1)
        targetLayerIndex=self.layerCount-4
        while targetLayerIndex>0 :
            if targetLayerIndex!=self.layerCount-4:
                #first update the intermediate jacobian...
                frontNeighboursDerivMatrix=np.array([neuron.derivNeurons for neuron in self.Layers[targetLayerIndex+1].neurons])
                frontNeighboursDerivMatrix=np.transpose(frontNeighboursDerivMatrix)
                intermediateJacobian=np.matmul(frontNeighboursDerivMatrix,intermediateJacobian)
            counterCopy=0
            for i,neuron in enumerate(self.Layers[-1].neurons):
                counter=0
                for j , targetNeuron in enumerate(self.Layers[targetLayerIndex].neurons):
                    targetRowVector=intermediateJacobian[j]
                    intermediateProduct=np.dot(targetRowVector,neuron.derivNeurons)
                    for k,weightDeriv in enumerate(targetNeuron.derivWeights):
                        self.gradientMatrix[i][self.matrixIndex-counter]=weightDeriv*intermediateProduct
                        counter+=1
                        if k==len(targetNeuron.derivWeights)-1:
                            self.gradientMatrix[i][self.matrixIndex - counter] = targetNeuron.derivOutput * intermediateProduct
                            counter+=1
                counterCopy=counter
            self.matrixIndex-=counterCopy
            targetLayerIndex-=1
    def updateGradientMatrix(self):
        if self.layerCount>=2:
            self.lastLayerGradient()
        if self.layerCount>=3:
            self.firstDeepLayerGradient()
        if self.layerCount>=4:
            self.secondDeepLayerGradient()
        if self.layerCount>=5:
            self.superDeepLayersGradient()
        self.matrixIndex=self.parameterCount-1
    def feedData(self,x_data,y_data):
        assert isinstance(x_data,list) ,"WRONG INPUT FORMAT @feedData"
        assert isinstance(y_data,list) , "WRONG OUTPUT FORMAT @feedData"

        if not isinstance(x_data[0],list):
            self.X_data=[x_data]
        else:
            self.X_data=x_data
        if not isinstance(y_data[0],list):
            self.Y_data=[y_data]
        else:
            self.Y_data=y_data


        assert len(self.X_data[0]) == self.Layers[0].count , "WRONG INPUT SIZE @feedData"
        assert len(self.Y_data[0]) == self.Layers[-1].count , "WRONG OUTPUT SIZE @feedData"

    def updateLossGradient(self):
        self.lossGradient = np.zeros(self.parameterCount)
        for i in range(len(self.X_data)):
            self.propagateForward(self.X_data[i])
            self.updateGradientMatrix()
            subtractionVector=np.subtract(self.Layers[-1].outputVector,self.Y_data[i])
            multiplicationVector=np.matmul(np.transpose(self.gradientMatrix),subtractionVector)
            self.lossGradient=self.lossGradient+multiplicationVector
    def updateBatchLossGradient(self,start,end):
        self.lossGradient = np.zeros(self.parameterCount)
        for i in range(start,end+1 ,1):
            #print(f"i value : {i}")
            self.propagateForward(self.X_data[i])
            self.updateGradientMatrix()
            subtractionVector = np.subtract(self.Layers[-1].outputVector, self.Y_data[i])
            multiplicationVector = np.matmul(np.transpose(self.gradientMatrix), subtractionVector)
            self.lossGradient = self.lossGradient + multiplicationVector
        self.lossGradient=np.divide(self.lossGradient,len(self.X_data))

    def updateLoss(self):
        loss=0
        for i in range(len(self.X_data)):
            self.propagateForward(self.X_data[i])
            subtractionVector = np.subtract(self.Layers[-1].outputVector, self.Y_data[i])
            poweredVector = subtractionVector*subtractionVector
            loss += poweredVector.sum()
        self.MSELoss=loss
        self.MSELoss/=len(self.X_data)
    def updateBatchLoss(self,start,end):
        loss = 0
        for i in range(start,end):
            self.propagateForward(self.X_data[i])
            subtractionVector = np.subtract(self.Layers[-1].outputVector, self.Y_data[i])
            poweredVector = subtractionVector * subtractionVector
            loss += poweredVector.sum()
        self.MSELoss = loss
    def simpleTrain(self,iterations):
        parameters=self.extractParamters()
        for i in range(iterations):
            self.updateLossGradient()
            parameters=parameters-np.multiply(self.learningRate,self.lossGradient)
            self.updateParamters(parameters)
            self.updateLoss()
            print(f"Loss:{self.MSELoss}")
    def SGDTrain(self,iterations):
        parameters = self.extractParamters()
        for i in range(iterations):
            starter=(i*self.batchSize)%len(self.X_data)
            ender=(i*self.batchSize+self.batchSize-1)%len(self.X_data)
            self.updateBatchLossGradient(start=starter,end=ender)
            parameters = parameters - np.multiply(self.learningRate, self.lossGradient)
            self.updateParamters(parameters)
            self.viewLoss()

    def SGDNesterovTrain(self,iterations):
        parameters = self.extractParamters()
        predictiveJumpParameters=parameters.copy()
        self.velocityVector=np.zeros(len(parameters))

        for i in range(iterations):
            starter = (i * self.batchSize) % len(self.X_data)
            ender = (i * self.batchSize + self.batchSize - 1) % len(self.X_data)
            predictiveJumpParameters=np.add(predictiveJumpParameters,self.velocityVector)
            self.updateParamters(predictiveJumpParameters)
            self.updateBatchLossGradient(starter,ender)
            self.velocityVector = np.multiply(self.velocityVector, self.velocityMomentum) - np.multiply(self.lossGradient, self.learningRate)
            parameters = np.add(self.velocityVector,parameters)
            self.updateParamters(parameters)
            self.updateLoss()
            print(self.MSELoss)



    def updateSquaredGradientVelocityVector(self, squaredLossGradient):
        self.velocityVector= np.multiply(self.velocityVector, self.velocityMomentum) + np.multiply(squaredLossGradient, 1 - self.velocityMomentum)
    def updateAdaGradVelocityVector(self,squaredLossGradient):
        self.velocityVector+=np.multiply(self.lossGradient,self.lossGradient)
    def updateRMSWeight(self,parameters):
        for i in range(len(parameters)):
            parameters[i]=parameters[i]-self.learningRate*self.lossGradient[i]/math.sqrt(self.velocityVector[i]+self.epsilon)
    def updateMomentVector(self):
        self.momentVector=np.multiply(self.momentMomentum,self.momentVector)+np.multiply(self.lossGradient,1-self.momentMomentum)

    def updateADAMParamters(self,parameters,mHat,vHat):
        for i in range(len(parameters)):
            parameters[i]=parameters[i]-self.learningRate*mHat[i]/(np.sqrt(vHat[i])+self.epsilon)
    def RMSPropTrain(self,iterations):
        parameters = self.extractParamters()
        self.velocityVector = np.zeros(len(parameters))
        for i in range(iterations):
            self.updateLossGradient()
            squaredGradient=np.multiply(self.lossGradient,self.lossGradient)
            self.updateSquaredGradientVelocityVector(squaredGradient)
            self.updateRMSWeight(parameters)
            self.updateParamters(parameters)
            self.updateLoss()
            print(f"Iteration:{i+1} Loss:{self.MSELoss}")
    def AdaGradTrain(self,iterations):
        parameters = self.extractParamters()
        self.velocityVector = np.zeros(len(parameters))
        for i in range(iterations):
            self.updateLossGradient()
            squaredGradient=np.multiply(self.lossGradient,self.lossGradient)

            self.updateAdaGradVelocityVector(squaredGradient)

            self.updateRMSWeight(parameters)
            self.updateParamters(parameters)

            self.updateLoss()

            print(f"Iteration:{i+1} Loss:{self.MSELoss}")
    def SGD_AdaGradTrain(self,start,end):
        parameters = self.extractParamters()
        self.velocityVector = np.zeros(len(parameters))
        for i in range(start,end):
            self.updateLossGradient()
            squaredGradient=np.multiply(self.lossGradient,self.lossGradient)
            self.updateAdaGradVelocityVector(squaredGradient)
            self.updateRMSWeight(parameters)
            self.updateParamters(parameters)
            self.updateLoss()
            print(f"Iteration:{i+1} Loss:{self.MSELoss}")
    def ADAMTrain(self,iterations,viewIndex=25):
        parameters = self.extractParamters()
        self.velocityVector = np.zeros(len(parameters))
        self.momentVector=np.zeros(len(parameters))
        for i in range(iterations):
            k = iterations+1
            self.updateLossGradient()
            squaredGradient = np.multiply(self.lossGradient, self.lossGradient)
            self.updateSquaredGradientVelocityVector(squaredGradient)
            self.updateMomentVector()
            mHat=np.divide(self.momentVector,1-np.power(self.momentMomentum,k))
            vHat=np.divide(self.velocityVector,1-np.power(self.velocityMomentum,k))
            self.updateADAMParamters(parameters,mHat=mHat,vHat=vHat)
            self.updateParamters(parameters)
            if (i+1)%viewIndex==0:
                self.updateLoss()
                print(f"Iteration:{i+1} Loss:{self.MSELoss}")

    def fineTunedADAMTrain(self, iterations):
        parameters = self.extractParamters()
        
        self.velocityVector = np.zeros(len(parameters))
        self.momentVector = np.zeros(len(parameters))
        for i in range(iterations):
            k = iterations + 1
            self.updateLossGradient()
            squaredGradient = np.multiply(self.lossGradient, self.lossGradient)
            self.updateSquaredGradientVelocityVector(squaredGradient)
            self.updateMomentVector()
            mHat = np.divide(self.momentVector, 1 - np.power(self.momentMomentum, k))
            vHat = np.divide(self.velocityVector, 1 - np.power(self.velocityMomentum, k))
            self.updateADAMParamters(parameters, mHat=self.momentVector, vHat=self.velocityVector)
            self.updateParamters(parameters)
            self.updateLoss()
            print(f"Loss: {self.MSELoss}")



    def HybridTrain(self,iterations,divisions,decayingFactor,fineTune=True):
        originalRate=self.learningRate
        partition=int(iterations/divisions)
        for i in range(divisions):
            self.ADAMTrain(partition)
            self.learningRate/=decayingFactor
        if fineTune==True:
            self.AdaGradTrain(partition)
        self.learningRate = originalRate
def max(x,y):
    if x > y:
        return x
    else:
        return y
def relu(x):
    return max(0,x)
def reluDeriv(x):
    if x>=0:
        return 1
    else:
        return 0
def leakyRelu(x):
    return max(0.01*x,x)
def leakyReluDeriv(x):
    if x>=0:
        return  1
    else :
        return 0.01

def normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalized_arr




x_data=[]
y_data=[]
i=0
while(i<np.pi/2):
    x_data.append([i])
    y_data.append([np.sin(i)])
    i+=0.1
net = NeuralNetwork(Layer(1),
                    Layer(100,leakyRelu,leakyReluDeriv),

                    Layer(1)
                    )
net.feedData(x_data=x_data,y_data=y_data)
net.learningRate=0.01
net.velocityMomentum=0.9
net.momentMomentum=0.9
#net.HybridTrain(5000,1,2,fineTune=False)
net.fineTunedADAMTrain(5000)
net.propagateForward([np.pi/4])
print(net.Layers[-1].outputVector[0])

