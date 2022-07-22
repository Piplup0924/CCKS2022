import os
from tqdm.auto import tqdm
import json
import cv2

type2id = dict()

for i in range(-100, 101):
    type2id[str(i)] = i + 100

for up in [chr(i) for i in range(65, 91)]:
    type2id[up] = ord(up) - 65 + 201

for low in [chr(i) for i in range(97, 123)]:
    type2id[low] = ord(low) - 97 + 227

types = type2id.keys()


def saveCOCOJson(Annotations,saveJson):
    json_str = json.dumps(Annotations, indent=4)
    with open(saveJson, 'w') as json_file:
        json_file.write(json_str)

def data_xywh2coco_xywh(yolo_xywh: list, imgW: int, imgH: int):
    center_x = yolo_xywh[0]
    center_y = yolo_xywh[1]
    w = yolo_xywh[2]
    h = yolo_xywh[3]

    x_min = center_x - w / 2
    y_min = center_y - h / 2

    coco_xywh = [x_min, y_min, w, h]

    return coco_xywh

def load_data(classNames: list, imagesPath, labelsPath):
    Annotations = {"images": [], "annotations": [], "categories": []}

    imgNameList = []
    for root, _, files in os.walk(imagesPath):
        imgNameList = files
    labelNameList = []
    for root, _, files in os.walk(labelsPath):
        labelNameList = files

    imgIdIdx = 0
    imgId2Name = {}
    imgName2Id = {}
    name2ImgName = {}
    for imgName in imgNameList:
        imgId2Name[imgIdIdx] = imgName
        imgName2Id[imgName] = imgIdIdx
        name, extension = os.path.splitext(imgName)
        name2ImgName[name] = imgName
        imgIdIdx += 1

    cateId2Name = {}
    for i, cateName in enumerate(classNames):
        cateId2Name[i + 1] = cateName

    annId = 0
    for txtName in tqdm(labelNameList):
        imgName = name2ImgName[txtName[:-5]]
        imgH, imgW, _ = cv2.imread(imagesPath + imgName).shape

        # write image info
        fjson = json.load(open(os.path.join(labelsPath, txtName)))
        Annotations["images"].append(
            {
                "file_name": imgName,
                "height": imgH,
                "width": imgW,
                "id": imgName2Id[imgName],
                "structure": fjson["structure"],
            })
        c = fjson["nodes"]
        for node in c:
            defect_type = str(node["value"])
            cateId = type2id[defect_type]
            b_box = node['xy']
            center_x = (b_box[2] + b_box[0]) / 2
            center_y = (b_box[1] + b_box[3]) / 2
            w = b_box[2] - b_box[0]
            h = b_box[3] - b_box[1]
            cateId = int(cateId)
            center_x = float(center_x)
            center_y = float(center_y)
            w = float(w) * 0.8
            h = float(h) * 0.8
            # w = float(w)
            # h = float(h)

            coco_xywh = data_xywh2coco_xywh([center_x, center_y, w, h], imgW, imgH)

            # write ann info
            Annotations["annotations"].append(
                {
                    "segmentation": [[]],
                    "iscrowd": 0,  # 0 or 1
                    "area": coco_xywh[2] * coco_xywh[3],  # float or double
                    "image_id": imgName2Id[imgName],  # int
                    "bbox": coco_xywh,  # list[float], [x,y,w,h]
                    "category_id": cateId + 1,  # data class start from 0, coco start from 1
                    "id": annId  # int
                }
            )
            annId += 1

    # write categories
    for cateId in cateId2Name:
        cateName = cateId2Name[cateId]
        Annotations["categories"].append({"id": cateId, "name": cateName})

    return Annotations


if __name__ == '__main__':

    img_path = "./dataset1/image/"
    label_path = "./dataset1/label/"
    save_path = "./dataset1/bbox_coco_0.8.json"
    jsonInfo = load_data(types, img_path, label_path)
    saveCOCOJson(jsonInfo, save_path)


    


