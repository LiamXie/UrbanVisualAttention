from genericpath import exists
import os

import cv2
import numpy as np
from PIL import Image


from util.projection import projection_methods
#args
pre_folder='D:/00Dataset/THUE0001_360/'
stored_path='D:/00Dataset/THUE0001_360/prediction'

face_key={}
face_key[0]='F'
face_key[1]='R'
face_key[2]='B'
face_key[3]='L'
face_key[4]='U'
face_key[5]='D'
face_key[6]='ERP'

def fusion_cmp(img_set):
    out=projection_methods.c2e(img_set,h=320, w=640, mode='bilinear', cube_format='dict')
    return 255*out/out.max()

def get_map(in_path,video_num,img_num,face_key):
    #args
    beta_F=0.8
    beta_LR=0.1
    beta_U=0.01
    beta_D=0.001
    img_set={}
    for k in face_key:
        v=face_key[k]
        if v =='ERP':
            img_path=in_path+'/THUE0001_'+v+'/prediction/'+str(video_num).zfill(4)+'/'+str(img_num).zfill(4)+'.jpg'
            a = np.array((Image.open(img_path)).convert("L"))
            if len(a.shape) == 2 :
                a = a[..., None]
            img_set[v] = np.array(a)
        elif v =='F':
            img_path=in_path+'/'+'/THUE0001_'+v+'/prediction/'+str(video_num).zfill(4)+'/'+str(img_num).zfill(4)+'.jpg'
            a = np.array((Image.open(img_path)).convert("L"))*beta_F
            if len(a.shape) == 2 :
                a = a[..., None]
            img_set[v] = np.array(a)
        elif v == 'L':
            img_path=in_path+'/THUE0001_LR/prediction/'+str(video_num).zfill(4)+'/'+str(img_num).zfill(4)+'.jpg'
            a = np.array((Image.open(img_path)).convert("L"))*beta_LR
            if len(a.shape) == 2 :
                a = a[..., None]
            img_set[v] = np.array(a)
        elif v == 'R':
            img_path=in_path+'/THUE0001_LR/prediction/'+str(video_num+550).zfill(4)+'/'+str(img_num).zfill(4)+'.jpg'
            a = np.array((Image.open(img_path)).convert("L"))*beta_LR
            if len(a.shape) == 2 :
                a = a[..., None]
            img_set[v] = np.array(a)
        elif v=='U' :
            img_path=in_path+'/THUE0001_'+v+'/prediction/'+str(video_num).zfill(4)+'/'+str(img_num).zfill(4)+'.jpg'
            a = np.array((Image.open(img_path)).convert("L"))*beta_U
            if len(a.shape) == 2 :
                a = a[..., None]
            img_set[v] = np.array(a)
        elif v=='D' :
            img_path=in_path+'/THUE0001_'+v+'/prediction/'+str(video_num).zfill(4)+'/'+str(img_num).zfill(4)+'.jpg'
            a = np.array((Image.open(img_path)).convert("L"))*beta_D
            if len(a.shape) == 2 :
                a = a[..., None]
            img_set[v] = np.array(a)
        else:
            a=np.zeros(img_set['F'].shape)
            if len(a.shape) == 2 :
                a = a[..., None]
            img_set[v] = np.array(a)
    return img_set

def fusion(video_num,img_num,stored_path,save=True):
    #args
    
    img_set=get_map(in_path=pre_folder,video_num=video_num,img_num=img_num,face_key=face_key)
    cmp_out=fusion_cmp(img_set=img_set)
    for beta in (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9):
        video_stored_path=stored_path+'/'+str(beta)+'/'+str(video_num).zfill(4)
        if not exists(video_stored_path):
            if not exists(stored_path+'/'+str(beta)):
                os.mkdir(stored_path+'/'+str(beta))
            os.mkdir(video_stored_path)
        path=video_stored_path+'/'+str(img_num).zfill(4)+'.jpg'

        final=beta*cmp_out+(1-beta)*img_set['ERP']
        final=final/final.max()

        if save:
            cv2.imwrite(path,(255*final).astype(np.uint8))

    # video_stored_path=stored_path+'/dot/'+str(video_num).zfill(4)
    # if not exists(video_stored_path):
    #     if not exists(stored_path+'/dot'):
    #         os.mkdir(stored_path+'/dot')
    #     os.mkdir(video_stored_path)
    # path=video_stored_path+'/'+str(img_num).zfill(4)+'.jpg'

    # final=cmp_out*img_set['ERP']
    # final=final/final.max()

    # if save:
    #     cv2.imwrite(path,(255*final).astype(np.uint8))

def main():
    #args
    for video_num in range (386,551):
        count=len(os.listdir(pre_folder+'/THUE0001_ERP/annotation/'+str(video_num).zfill(4)+"/images"))
        print(video_num,count)
        for img_num in range(1,count//80*80,5):
            fusion(video_num=video_num,img_num=img_num,stored_path=stored_path)

if __name__ == "__main__":
    main()