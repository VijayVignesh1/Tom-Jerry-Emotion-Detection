from imports import *
class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, data_folder, data_name, label_file, data_type='None', transform=None):
            """
            :param data_folder: folder where data files are stored
            :param data_name: base name of processed datasets
            :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
            :param transform: image transform pipeline
            """

            # Open hdf5 file where images are stored
            add=''
            if data_type=='VAL':
                add='_VAL' 
            self.h = h5py.File(os.path.join(data_folder, data_name + add + '.h5'), 'r')
            self.imgs = self.h['images']
            # print(self.imgs.shape)
            self.csv=pd.read_csv(label_file)
            emotion=self.csv['Emotion']
            print(emotion.shape)
            values=[]
            for i in emotion:
                #print(i)
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
            self.values=np.array(values)
            #print(values)
            ### open and read label_file
            #print(self.imgs[4232])
            #self.imgs=torch.Tensor(self.imgs)
            #self.imgs=self.imgs.resize((21223,1,256,256))
            #self.split = data_type
            # # Captions per image
            # self.value = self.h['Y_train']
            # self.value=np.array(self.value)
            # self.value=np.array([np.where(r==1)[0][0] for r in self.value])
            # # PyTorch transformation pipeline for the image (normalizing, etc.)
            self.transform = transform

            # Total number of datapoints
            self.dataset_size = len(self.values)
            #print(self.dataset_size)
            #print(self.imgs.shape[0])
    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i] / 255.)
        value = torch.LongTensor([self.values[i]])
        #img=img.unsqueeze(0)
        if self.transform is not None:
            img = self.transform(img)
        # print(value.shape)
        return [img,value]    #### unsqueeze later and remove list brackets
    def __len__(self):
        return self.dataset_size