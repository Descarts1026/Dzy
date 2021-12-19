├── backbone: 特征提取网络，可以根据自己的要求选择
├── network_files: Faster R-CNN网络（包括Fast R-CNN以及RPN等模块）
├── train_utils: 训练验证相关模块（包括cocotools）
├── my_dataset.py: 自定义dataset用于读取VOC数据集 
├── train_resnet.py: 以resnet50 做为backbone进行训练 
├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
├── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
└── pascal_voc_classes.json: pascal_voc标签文件

  1、修改train_data/data_config中的路径 
  3、修改train_resnet.py 中的num-classes  为自己的类别个数 
  4、修改pascal_voc_classes.json 为自己的类别
  5、运行 train_resnet.py
  