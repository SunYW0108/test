import os
from PIL import Image
import numpy as np
from skimage import measure
import cv2
import pycocotools.mask as mask_util
import math

def is_sameobject(mask1,mask2,object_label):
    x1=np.where(mask1 == 1)[0][0]
    y1=np.where(mask1 == 1)[1][0]
    x2 = np.where(mask2 == 1)[0][0]
    y2 = np.where(mask2 == 1)[1][0]
    if object_label[x1,y1]==object_label[x2,y2]:
        return True
    return False

def Object_Connected_Component(masks,filepath):
    masks=list(masks)
    object_label = Image.open(filepath)
    object_label = np.array(object_label)
    del_list=[]
    for i in range(len(masks)-1):
        if np.where(masks[i] == 1)[0].size == 0:
            continue
        for j in range(i+1,len(masks)):
            if np.where(masks[j] == 1)[0].size == 0:
                continue
            if is_sameobject(masks[i],masks[j],object_label):
                masks[i]=masks[i]+masks[j]
                masks[j]=np.zeros(masks[j].shape)
                del_list.append(j)

    masks_new=[]
    for idx in range(len(masks)):
        if idx not in del_list:
            masks_new.append(masks[idx])
    return masks_new

def ParallelLinesApprox(contour):
    slope_list=[]
    for i in range(contour.shape[0]):
        pt1=contour[i,0]
        i2=i+1
        if i2>contour.shape[0]-1:
            i2=i2-contour.shape[0]
        pt2=contour[i2,0]
        slope=(pt1[1]-pt2[1])/(pt1[0]-pt2[0])
        slope_list.append(slope)
        # 5度对应斜率为0.0875
    thred_slope = math.tan(5.0 / 180.0 * math.pi)
    for i in range(len(slope_list)):
        s1=slope_list[i]
        i2 = i + 2
        if i2 > len(slope_list) - 1:
            i2 = i2 - len(slope_list)
        s2 = slope_list[i2]
        if s1-s2<thred_slope:
            #i+1和i2构成中间直线，将其平移至最大端点处



    return

def main():
    # ROOT_DIR='/home/sun/facades_datasets/3.etrims/'
    # MASK_NAME='basel_000051_mv0'
    # MASK_PATH=os.path.join(ROOT_DIR,'annotations','%s.png' % MASK_NAME)
    MASK_PATH='./test6.png'
    mask=Image.open(MASK_PATH)
    mask = np.array(mask)
    # P模式索引图像使用单个数字
    # mask = (mask == 8).astype('uint8')*255 #数字代表某一个要素
    # RGB变布尔
    height = mask.shape[0]
    width = mask.shape[1]
    labels = np.zeros([height, width])
    for i in range(height):
        for j in range(width):
            r = mask[i, j, 0]
            g = mask[i, j, 1]
            b = mask[i, j, 2]
            if mask[i, j, 0] == 255 and mask[i, j, 1] == 255 and mask[i, j, 2] == 255:
                labels[i, j] = 255
    mask = labels.astype('uint8')

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_pt_list = np.concatenate(contours,axis=0)
    contour = cv2.convexHull(contour_pt_list)
    contour_length = cv2.arcLength(contour, True)
    cof = 1
    while cof > 0:
        epsilon = cof * contour_length
        gt_poly = cv2.approxPolyDP(contour, epsilon, True)
        edge_num = len(gt_poly)
        if edge_num >= 4: break
        cof = cof - 0.05
    # if edge_num < 3: continue
    if edge_num>=4:
        gt_poly=ParallelLinesApprox(gt_poly)




    mask2 = np.zeros(mask.shape)
    cv2.fillPoly(mask2, [gt_poly], 255)
    cv2.imshow('raw', mask)
    cv2.imshow('contour', mask2)
    cv2.waitKey()








    #此处开始*********************************************
    mask = measure.label(mask, connectivity=2)
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[1:]
    masks = (mask == obj_ids[:, None, None]).astype('uint8')   *255
    # masks = Object_Connected_Component(masks,os.path.join(ROOT_DIR,'annotations-object',"%s.png" % MASK_NAME))
    #输出是一个list
    num_objs = len(masks)
    for i in range(num_objs):
        binary_mask_encoded = mask_util.encode(np.asfortranarray(masks[i].astype(np.uint8)))

        mask_area = mask_util.area(binary_mask_encoded)
        if mask_area < 1:
            continue

        bounding_box = mask_util.toBbox(binary_mask_encoded)
        # 判断包围盒是否贴近图像边缘
        img_size=masks[i].shape
        if bounding_box[0] == 0 or bounding_box[1] == 0: continue
        if bounding_box[0] + bounding_box[2] > img_size[1] - 1 \
                or bounding_box[1] + bounding_box[3] > img_size[0] - 1:
            continue

        contour,hierarchy=cv2.findContours(masks[i],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        mask_length=cv2.arcLength(contour[0], True)

        contour[0]=cv2.convexHull(contour[0])

        cof=1
        while True:
            epsilon=cof*mask_length
            poly=cv2.approxPolyDP(contour[0],epsilon ,True)
            edge_num=len(poly)
            if edge_num >= 4: break
            cof=cof/2

        print (edge_num)

        poly_length=cv2.arcLength(poly, True)
        poly_mask=np.zeros(masks[i].shape)
        cv2.fillPoly(poly_mask, [poly], 255)
        poly_encoded=mask_util.encode(np.asfortranarray(poly_mask.astype(np.uint8)))
        poly_area = mask_util.area(poly_encoded)

        poly_mask_IOU = mask_util.iou([binary_mask_encoded], [poly_encoded],[0])

        #loss=(1-poly_mask_IOU)**2+(mask_area/poly_area-1)**2+(mask_length/poly_length-1)**2

        cv2.imshow('mask',masks[i])
        cv2.imshow('contour',poly_mask)
        cv2.waitKey()








if __name__ == "__main__":
    main()