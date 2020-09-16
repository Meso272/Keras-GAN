import numpy as np 
import os 


def load_CLDHGH_orig(path,size=32,startnum=0,endnum=50,scale=True):
    height=1800
    width=3600
    picts=[]
    for i in range(startnum,endnum):
        s=str(i)
        if i<10:
            s="0"+s
        filename="CLDHGH_%s.dat" % s
        filepath=os.path.join(path,filename)
        array=np.fromfile(filepath,dtype=np.float32).reshape((height,width))
        #print(array)
        for x in range(0,height,size):
            if x+size>height:
                continue
            for y in range(0,width,size):
                if y+size>width:
                    continue
                #print(array[x:x+size,y:y+size])
                picts.append(array[x:x+size,y:y+size])
    picts=np.array(picts)
    if scale:
        picts=(picts-0.5)*2
    return picts

def load_CLDHGH_decomp(path,size=32,startnum=0,endnum=50,scale=True):
    height=1800
    width=3600
    picts=[]
    for i in range(startnum,endnum):
        s=str(i)
        if i<10:
            s="0"+s
        filename="CLDHGH_%s.dat.sz.out" % s
        filepath=os.path.join(path,filename)
        array=np.fromfile(filepath,dtype=np.float32).reshape((height,width))
        for x in range(0,height,size):
            if x+size>height:
                continue
            for y in range(0,width,size):
                if y+size>width:
                    continue
                picts.append(array[x:x+size,y:y+size])
    picts=np.array(picts)
    if scale:
        picts=(picts-0.5)*2
    return picts


#path1="/home/jliu447/lossycompression/multisnapshot-data-cleaned/CLDHGH/"
#path2="/home/jliu447/lossycompression/multisnapshot-data-cleaned/CLDHGH_SZ/"

#a1=load_CLDHGH_orig(path1)
#a2=load_CLDHGH_decomp(path2)

#print(a1.shape)
#print(a2.shape)