from imports import *
from model import *
from utils import CaptionDataset
data_folder = ''
data_name='data_balanced_aug'
batch_size=16
workers=0
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
dataset=CaptionDataset(data_folder, data_name, 'train_balanced_aug.csv', 'TRAIN', transform=transforms.Compose([normalize]))

train_loader = torch.utils.data.DataLoader(dataset,
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

print(iter(train_loader).next())
device="cuda"
# model = ConvNet()
# model=Encoder()
model=ResNet()
# model= Ensemble()
learning_rate=1e-5
num_epochs=10
# Loss and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model=model.to(device)

total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if epoch%2==0 and epoch!=0 and epoch!=2 and epoch!=4:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate/10
                # print(param_group['lr'])
        # Run the forward pass
        outputs = model(images.to(device))
        #print(outputs[:4])
        # print("lala",labels)
        labels=labels.reshape((labels.shape[0]))
        #print("lala",labels)
        loss = criterion(outputs.to(device), labels.to(device))
        loss_list.append(loss.item())
        # if epoch==num_epochs-1 and i==np.ceil(2328/batch_size):
        #     break
        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        predicted = torch.softmax(outputs,dim=1)
        _,predicted=torch.max(predicted, 1)

        correct = (predicted == labels.to(device)).sum().item()
        acc_list.append(correct / total)
        f1=f1_score(labels.cpu().numpy(),predicted.cpu().numpy(),average='weighted')

        if (i + 1) % 10 == 0:
            # print(predicted)
            # print(labels)
            # print((predicted == labels.to(device)).shape)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}, F1: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100, 100*f1))
            #print(correct,total)
# print(outputs)
#### Test ####

data=h5py.File('data_test.h5','r')
images=data['images']
test_loader=torch.utils.data.DataLoader(images,batch_size=batch_size, num_workers=0)
# a=iter(test_loader).next()[4]
# print(a.max(axis=0))
# exit(0)
final=[]
for i in test_loader:
    i=i/255.
    # print(i)
    outputs=model(i.to(device))
    # print(outputs)
    outputs=torch.softmax(outputs,dim=1).argmax(dim=1)
    final.extend(outputs.tolist())
outputs=final
print(outputs)
values=[]
for i in outputs:
    #print(i)
    if i==0:
        values.append('angry')
    if i==1:
        values.append('happy')
    if i==2:
        values.append('sad')
    if i==3:
        values.append('surprised')
    if i==4:
        values.append('Unknown')
values=np.array(values)
pd.DataFrame(values).to_csv("result_submit.csv")

