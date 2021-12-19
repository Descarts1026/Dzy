import os
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import vgg
from draw_box_utils import draw_box
import glob
classes = ["Weevil", "Leafhopper", "Grubs", "Mole_cricket", "Elateroidea",
"Adult_aphid", "potosia_brevitarsis", "Blister_beetle", "Pieris_canidia",
"Golden_needle_worm", "Miridae"]
def create_model(num_classes):
    vgg_feature = vgg(model_name="vgg16", weights_path="./pretrained/backbone/vgg16.pth").features
    backbone = torch.nn.Sequential(*list(vgg_feature._modules.values())[:-1])  # 删除features中最后一个Maxpool层
    backbone.out_channels = 512

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                                    output_size=[7, 7],  # roi_pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes + 1,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

with open("./train_data/valdata.txt", "r") as f:
    lines = f.readlines()
import  time
detect_time = open("accuracy/time.txt", "w")

def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=11)#n

    # load train weights
    train_weights = "./save_weights/vgg-model-100.pth"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)

    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}
    for file in lines :
        file = file.split(" ")[0]
        img_name = file.split("/")[-1]
        # load image
        original_img = Image.open(file)

        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))
            detect_time.write(img_name + " " + str(t_end - t_start) + "\n")

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            with open("accuracy/detections/" + img_name + ".txt", "w") as f:
                for data in zip(predict_classes,predict_scores,predict_boxes):
                    f.write(" ".join([str(classes[data[0]-1]),str(data[1]),
                                      str(data[2][0]),str(data[2][1]),str(data[2][2]),str(data[2][3])]))
                    f.write("\n")

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")




if __name__ == '__main__':
    main()

