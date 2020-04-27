from imports import *
csv1=pd.read_csv("train_aug_64.csv")
emotion=csv1['Emotion']
values=[]
for i in emotion:
    if i=='angry':
        values.append(0)
    if i=='happy':
        values.append(1)
    if i=='sad':
        values.append(2)
    if i=='surprised':
        values.append(3)
    if i=='Unknown':
        values.append(4)
print("Angry",values.count(0))
print("Happy",values.count(1))
print("Sad",values.count(2))
print("Surprised",values.count(3))
print("Unknown",values.count(4))
# exit(0)
indices_unk = [i for i, x in enumerate(values) if x == 4]
shuffle(indices_unk)
indices_unk=indices_unk[:1300]
values_unk=[values[i] for i in range(len(values)) if i in indices_unk]
values=[values[i] for i in range(len(values)) if i not in indices_unk]
file=h5py.File("data_train_aug_64.h5",'r')
data=file.get('images')
# print(data.shape)
data=np.delete(data,indices_unk,axis=0)
indices_sur = [i for i, x in enumerate(values) if x == 3]
shuffle(indices_sur)
indices_sur=indices_sur[:500]
values_sur=[values[i] for i in range(len(values)) if i in indices_sur]
values=[values[i] for i in range(len(values)) if i not in indices_sur]
data=np.delete(data,indices_sur,axis=0)
N=data.shape[0]
image_size=64
# print(data.shape)
# exit(0)
h5_file=h5py.File('data_balanced_aug_64.h5','w')
data = h5_file.create_dataset('images', data=data, shape=(N,3,image_size,image_size))
h5_file.close()
values=np.array(values)
outputs=[]
for i in values:
    if i==0:
        outputs.append('angry')
    if i==1:
        outputs.append('happy')
    if i==2:
        outputs.append('sad')
    if i==3:
        outputs.append('surprised')
    if i==4:
        outputs.append('Unknown')
print("Angry",outputs.count('angry'))
print("Happy",outputs.count('happy'))
print("Sad",outputs.count('sad'))
print("Surprised",outputs.count('surprised'))
print("Unknown",outputs.count('Unknown'))
outputs=np.array(outputs)
pd.DataFrame(outputs,columns=['Emotion']).to_csv("train_balanced_aug_64.csv")
