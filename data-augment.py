from imports import *
file_name="data_generate_64.h5"
output_filename="data_train_aug_64.h5"
label_output_file="train_aug_64.csv"
image_shape=64

file=h5py.File(file_name,'r')
data=file.get('images')
csv1=pd.read_csv('Dataset/Train.csv')
emotion=csv1['Emotion']
emotions=[]


# Creating a translation matrix
# translation_matrix = np.float32([ [1,0,20], [0,1,0] ])

# Image translation
# img_translation = cv2.warpAffine(img, translation_matrix, (num_cols,num_rows))


images=torch.zeros((1,image_shape,image_shape,3)).float()
# print(len(data))
my_list=list(range(len(data)))
print(images.shape)

for i in tqdm(my_list):
    # print(data[i].shape)
    tempz=data[i].transpose(1,2,0)
    # print(tempz.shape)
    # cv2.imshow('',tempz/255.)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    emotions.append(emotion[i])
    temp=data[i].reshape((1,image_shape,image_shape,3))
    temp=torch.Tensor(temp)
    images=torch.cat((images,temp),0)
    #### Rotation at 15,30,45,60,75 degrees ####
    for j in range(15,90,15):
        rotated = rotate(tempz, angle=j, mode = 'wrap')
        rotated=rotated.reshape((1,image_shape,image_shape,3))
        # print(rotated.shape)
        # ro=rotated.reshape((224,224,3))
        # cv2.imshow('',ro/255.)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        rotated=torch.Tensor(rotated)
        images=torch.cat((images,rotated),0)
        emotions.append(emotion[i])
    for j in range(10,40,10):
        translation_matrix = np.float32([ [1,0,j], [0,1,0] ])
        img_translation = cv2.warpAffine(tempz, translation_matrix, (image_shape,image_shape))
        # print(img_translation.shape)
        # cv2.imshow('',img_translation/255.)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img_translation=img_translation.reshape((1,image_shape,image_shape,3))
        img_translation=torch.Tensor(img_translation)
        images=torch.cat((images,img_translation),0)
        emotions.append(emotion[i])
        # data[i]=data[i].transpose(2,0,1)
    for j in range(10,40,10):
        temp1=data[i].transpose(1,2,0)
        translation_matrix = np.float32([ [1,0,0], [0,1,j] ])
        img_translation = cv2.warpAffine(tempz, translation_matrix, (image_shape,image_shape))
        # cv2.imshow('',img_translation/255.)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img_translation=img_translation.reshape((1,image_shape,image_shape,3))
        img_translation=torch.Tensor(img_translation)
        images=torch.cat((images,img_translation),0)
        emotions.append(emotion[i])    
        temp1=data[i].transpose(2,0,1)
    #### Shift Transform ####
    transform = AffineTransform(translation=(25,25))
    wrapShift = warp(tempz,transform,mode='wrap')
    # cv2.imshow('',wrapShift/255.)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    wrapShift=wrapShift.reshape((1,image_shape,image_shape,3))
    wrapShift=torch.Tensor(wrapShift)
    images=torch.cat((images,wrapShift),0)
    emotions.append(emotion[i])

    #### Flip left-right and up-down ####
    flipLR = np.fliplr(tempz).copy()
    # cv2.imshow('',flipLR/255.)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    flipLR=flipLR.reshape((1,image_shape,image_shape,3))
    flipLR=torch.Tensor(flipLR)
    images=torch.cat((images,flipLR),0)
    emotions.append(emotion[i])

    flipUD = np.flipud(tempz).copy()
    # cv2.imshow('',flipUD/255.)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    flipUD=flipUD.reshape((1,image_shape,image_shape,3))
    flipUD=torch.Tensor(flipUD)
    images=torch.cat((images,flipUD),0)
    emotions.append(emotion[i])

    #### Random Noise ####
    sigma=0.155
    noisyRandom = random_noise(tempz/255.,var=sigma**2)
    noisyRandom*=255.
    # cv2.imshow('',noisyRandom/255.)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    noisyRandom=noisyRandom.reshape((1,image_shape,image_shape,3))
    noisyRandom=torch.Tensor(noisyRandom)
    images=torch.cat((images,noisyRandom),0)
    emotions.append(emotion[i])

    #### Gaussian Blurring ####
    blurred = gaussian(tempz,sigma=1,multichannel=True)
    # cv2.imshow('',blurred/255.)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    blurred=blurred.reshape((1,image_shape,image_shape,3))
    blurred=torch.Tensor(blurred)
    images=torch.cat((images,blurred),0)
    emotions.append(emotion[i])

print(np.array(emotions).shape[0])
print(images[1:].shape)
assert np.array(emotions).shape[0]==images[1:].shape[0]

images=images[1:].permute(0,3,1,2)
# print(np.array(emotions).shape)
emotions=np.array(emotions)
pd.DataFrame(emotions,columns=['Emotion']).to_csv(label_output_file)
# print(emotions)
# print(images.shape)
N=images.shape[0]
h5_file=h5py.File(output_filename)
data = h5_file.create_dataset('images', data=images, shape=(N,3,image_shape,image_shape))
h5_file.close()
