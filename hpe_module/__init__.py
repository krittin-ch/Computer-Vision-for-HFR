from .utils import load_snapshot
from .utils.camera_normalize import drawAxis
from .network.network import Network
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import torch
import time

def scale_bbox(bbox, scale):
    w = max(bbox[2], bbox[3]) * scale
    x= max(bbox[0] + bbox[2]/2 - w/2,0)
    y= max(bbox[1] + bbox[3]/2 - w/2,0)
    return np.asarray([x,y,w,w],np.int64)

def genAxis(input_img_path, output_img_path):
    # Setup
    # face_cascade = cv2.CascadeClassifier('hpe_module/lbpcascade_frontalface_improved.xml')
    face_cascade = cv2.CascadeClassifier('hpe_module/haarcascade_frontalface_alt.xml')
    pose_estimator = Network(bin_train=False)
    load_snapshot(pose_estimator,"hpe_module/models/model-b66.pkl")
    pose_estimator = pose_estimator.eval()

    transform_test = transforms.Compose([transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    
    # Import image
    frame = cv2.imread(input_img_path)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_img, 1.2)

    # Find bbox of faces in a frame
    face_images = []
    face_tensors = []
    for i, bbox in enumerate(faces):
        x,y, w,h = scale_bbox(bbox,1.5)
        frame = cv2.rectangle(frame,(x,y), (x+w, y+h),color=(0,0,255),thickness=2)
        face_img = frame[y:y+h,x:x+w]
        face_images.append(face_img)
        pil_img = Image.fromarray(cv2.cvtColor(cv2.resize(face_img,(224,224)), cv2.COLOR_BGR2RGB)) # downsample image
        face_tensors.append(transform_test(pil_img)[None])

    # generate row, pitch, yaw axis
    if len(face_tensors)>0:
        with torch.no_grad():
            start = time.time()
            face_tensors = torch.cat(face_tensors,dim=0)
            roll, yaw, pitch = pose_estimator(face_tensors)
            print("inference time: %.3f ms/face"%((time.time()-start)/len(roll)*1000))
            for img, r,y,p in zip(face_images, roll,yaw,pitch):
                headpose = [r,y,p]
                drawAxis(img, headpose,size=50)

    cv2.imwrite(output_img_path, frame)

def getAxis(frame):
    # Setup
    # face_cascade = cv2.CascadeClassifier('hpe_module/lbpcascade_frontalface_improved.xml')
    face_cascade = cv2.CascadeClassifier('hpe_module/haarcascade_frontalface_alt.xml')
    pose_estimator = Network(bin_train=False)
    load_snapshot(pose_estimator,"hpe_module/models/model-b66.pkl")
    pose_estimator = pose_estimator.eval()

    transform_test = transforms.Compose([transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    
    # Import image
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_img, 1.2)

    # Find bbox of faces in a frame
    face_images = []
    face_tensors = []
    for i, bbox in enumerate(faces):
        x,y, w,h = scale_bbox(bbox,1.5)
        frame = cv2.rectangle(frame,(x,y), (x+w, y+h),color=(0,0,255),thickness=2)
        face_img = frame[y:y+h,x:x+w]
        face_images.append(face_img)
        pil_img = Image.fromarray(cv2.cvtColor(cv2.resize(face_img,(224,224)), cv2.COLOR_BGR2RGB)) # downsample image
        face_tensors.append(transform_test(pil_img)[None])
    
    # generate row, pitch, yaw axis
    if len(face_tensors)>0:
        with torch.no_grad():
            start = time.time()
            face_tensors = torch.cat(face_tensors,dim=0)
            head_pose = pose_estimator(face_tensors)

            return head_pose
        
    return None