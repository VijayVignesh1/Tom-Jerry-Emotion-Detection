from imports import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="train", help="Train or Test")
opt = parser.parse_args()
if opt.data=="train":
    image_size=224
    cap=cv2.VideoCapture('Dataset/TrainTomandJerry.mp4')
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #cap.set(cv2.CAP_PROP_FPS,1.0)
    #print(cap.get(cv2.CAP_PROP_FPS))
    #print(length)
    #exit(0)
    images=torch.zeros((1,image_size,image_size,3)).float()
    frame_rate=0.857
    frameRate = cap.get(5)
    print(frameRate)
    #exit(0)
    prev=0
    while(cap.isOpened()):
        frameId = cap.get(1)
        time_elapsed=time.time()-prev
        ret, frame= cap.read()
        #print(ret)
        if ret==False:
            break
            print("False")
        frame_show=frame
        frame=cv2.resize(frame,(image_size,image_size))
        frame=frame.reshape((1,image_size,image_size,3))
        #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame=torch.Tensor(frame)
        #if time_elapsed>1./frame_rate:
        if (frameId % math.floor(frameRate) == 0):
            prev=time.time()
            images=torch.cat((images,frame),0)
            #cv2.imshow('frame',frame_show)
            #cv2.waitKey(0)
    cap.release()
    #cv2.destroyAllWindows()

    images=images[1:].permute(0,3,1,2)

    #print(images[0])
    N=images.shape[0]
    print(images[1:].shape)
    h5_file=h5py.File('data_generate_224.h5')
    data = h5_file.create_dataset('images', data=images, shape=(N,3,image_size,image_size))
    h5_file.close()
else:
    cap=cv2.VideoCapture('Dataset/TestTomandJerry.mp4')
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #cap.set(cv2.CAP_PROP_FPS,1.0)
    #print(cap.get(cv2.CAP_PROP_FPS))
    #print(length)
    #exit(0)
    images=torch.zeros((1,224,224,3)).float()
    frameRate = 29.97
    print(frameRate)
    #exit(0)
    prev=0
    while(cap.isOpened()):
        frameId = cap.get(1)
        time_elapsed=time.time()-prev
        ret, frame= cap.read()
        #print(ret)
        if ret==False:
            break
            print("False")
        frame_show=frame
        frame=cv2.resize(frame,(224,224))
        frame=frame.reshape((1,224,224,3))
        #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame=torch.Tensor(frame)
        #if time_elapsed>1./frame_rate:
        if (frameId % math.floor(frameRate) == 0):
            prev=time.time()
            images=torch.cat((images,frame),0)
            # cv2.imshow('frame',frame_show)
            # cv2.waitKey(0)
    cap.release()
    #cv2.destroyAllWindows()
    print(images[1:].shape)
    images=images[1:].permute(0,3,1,2)
    N=images.shape[0]
    image_size=224
    exit(0)
    h5_file=h5py.File('data_test.h5')
    data = h5_file.create_dataset('images', data=images, shape=(N,3,image_size,image_size))
    h5_file.close()