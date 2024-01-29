import cv2
import numpy as np
import os

def normalize(img):
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    return img

#read frame and saliency map, convert to color map and add
def heat_single(salicency_path=str,org_path=str,output_path=str):
    salicency_img=cv2.imread(salicency_path)
    salicency_img = normalize(salicency_img)
    salicency_img = salicency_img * 255
    salicency_img=np.uint8(salicency_img)

    org_img=cv2.imread(org_path)
    org_img=cv2.resize(org_img,(salicency_img.shape[1],salicency_img.shape[0]))
    heat_img = cv2.applyColorMap(salicency_img, cv2.COLORMAP_JET)
    #heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    img_add = cv2.addWeighted(org_img, 0.7, heat_img, 0.3, 0)
    cv2.imwrite(output_path,img_add)

def generation_heatmap():
    for video_num in range(386,551):
        sal_folder="D:/00Dataset/THUE0001_360/prediction/0.5/"+str(video_num).zfill(4)+"/"
        fra_folder="D:/00Dataset/THUE0001_360/THUE0001_ERP/annotation/"+str(video_num).zfill(4)+"/images/"
        out_folder="D:/00Dataset/temp/comparision_360/pred/"+str(video_num).zfill(4)+"/"
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        for i in range(1,241,5):
            heat_single(salicency_path=sal_folder+str(i).zfill(4)+".jpg",org_path=fra_folder+str(i).zfill(4)+".jpg",output_path=out_folder+str(i).zfill(4)+".jpg")

        sal_folder="D:/00Dataset/THUE0001_360/THUE0001_ERP/annotation/"+str(video_num).zfill(4)+"/maps/"
        out_folder="D:/00Dataset/temp/comparision_360/gt/"+str(video_num).zfill(4)+"/"
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        for i in range(1,241,5):
            heat_single(salicency_path=sal_folder+str(i).zfill(4)+".jpg",org_path=fra_folder+str(i).zfill(4)+".jpg",output_path=out_folder+str(i).zfill(4)+".jpg")

def generation_comparison():
    # read random 3 frames from random video
    # upper row: ground truth
    # lower row: prediction
    # white space between frames
    for i in range(1):
        video_num=np.random.randint(386,551)
        ground_truth_folder="D:/00Dataset/temp/comparision_360/gt/"+str(video_num).zfill(4)+"/"
        prediction_folder="D:/00Dataset/temp/comparision_360/pred/"+str(video_num).zfill(4)+"/"
        out_folder="D:/00Dataset/temp/comparision/comparison_both/"
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        for j in range(3):
            img_num=np.random.randint(i*16,i*16+16)*5+1
            gt_img=cv2.imread(ground_truth_folder+str(img_num).zfill(4)+".jpg")
            pred_img=cv2.imread(prediction_folder+str(img_num).zfill(4)+".jpg")
            if j==0:
                gt_row=gt_img[:,50:590,:]
                pred_row=pred_img[:,50:590,:]
            else:
                gt_row=np.hstack((gt_row,gt_img[:,50:590,:]))
                pred_row=np.hstack((pred_row,pred_img[:,50:590,:]))
        if i==0:
            gt=gt_row
            pred=pred_row
        else:
            gt=np.vstack((gt,gt_row))
            pred=np.vstack((pred,pred_row))
    gt=cv2.resize(gt,(gt.shape[1]//2,gt.shape[0]//2))
    pred=cv2.resize(pred,(pred.shape[1]//2,pred.shape[0]//2))
    # add white space between rows
    gt=np.vstack((gt,np.ones((5,gt.shape[1],3))*255))
    out_img=np.vstack((gt,pred))
    cv2.imwrite(out_folder+str(np.random.randint(1,1000,1)[0]).zfill(4)+".jpg",out_img)

if __name__=="__main__":
    # generation_heatmap()
    generation_comparison()