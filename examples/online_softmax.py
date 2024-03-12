import numpy as np
import copy as cp


def simplesoftmax(S_in):
    S=cp.copy(S_in)    
    # S=softmax(S)
    rowmaxS=np.max(S,axis=1)
    S=S-np.tile(rowmaxS, (np.shape(S)[0],1)).T
    S=np.exp(S)
    rowsumS=np.sum(S,axis=1)
    S=S/(np.tile(rowsumS, (np.shape(S)[0],1)).T)
    
    return(S)


def onlinesoftmax(S_in,Br=5,Bc=4):
    save_shape=S_in.shape
        
    N,x=S_in.shape

    #check for too large block sizes
    Br=min(Br,N)
    Bc=min(Bc,N)
    #get number of row/column blocks
    Tr=N//Br
    if(Tr*Br!=N):
        Tr+=1
    Tc=N//Bc
    if(Tc*Bc!=N):
        Tc+=1
        
    # outputs
    # Initialize om HBM
    O=np.zeros((N,N))
    l=np.zeros(N)
    m=np.zeros(N)-np.Infinity
    
    
    #switch to tensors
    # dimensions (Tr, Br, Tc, Bc, d)
    Otensor=np.reshape(O,(Tr,Br,Tc,Bc))
    mtensor=np.reshape(m,(Tr,Br))
    ltensor=np.reshape(l,(Tr,Br))
    del(m,l)
    
    QKtensor=np.reshape(S_in,(Tr,Br,Tc,Bc))

    
    for i in range(Tr):
            
        for j in range(Tc):
            Si=QKtensor[i,:,j,:]

            mi_old=cp.copy(mtensor[i,:])
                
            rowmaxS=np.max(Si,axis=-1)

            mtensor[i,:]=np.maximum(mtensor[i,:],rowmaxS)

            Si=Si-np.expand_dims(mtensor[i,:], axis=-1)

            Si=np.exp(Si)

            expmidiff=np.exp(mi_old-mtensor[i,:])
            ltensor[i,:]*=expmidiff

            ltensor[i,:]+= np.sum(Si,axis=-1)

            Otensor[i,:,:,:]*=np.expand_dims(expmidiff, axis=(-2,-1))

            Otensor[i,:,j,:]=Si

        Otensor[i,:,:,:]/=np.expand_dims(ltensor[i,:], axis=(-2,-1))

    O=np.reshape(Otensor,(N,N))
            
    return(O,ltensor)


shape1=(128,16)
Q=np.random.random(shape1)
K=np.random.random(shape1)
V=np.identity(shape1[0])

Stmp=Q.dot(K.T)

Osimple=simplesoftmax(Stmp)
Oflash,llash=onlinesoftmax(Stmp,Br=8,Bc=4)

print("difference=",np.linalg.norm(Osimple-Oflash))





