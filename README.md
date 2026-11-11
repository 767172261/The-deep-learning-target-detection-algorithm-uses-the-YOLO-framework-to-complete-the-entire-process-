深度学习目标检测算法使用 YOLO 框架完成从训练工业仪表数据集到性能评估的全流程
@[toc]
以下文字及代码仅供参考。

工业仪表数据集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/65a6a8d6f26d45d9890a8d9b1be52f78.png)

包含指针式仪表和数字式仪表数据集
926张，xml和txt格式标签，yolo训练
![\](https://i-blog.csdnimg.cn/direct/ea35c7881fd243ba930dfc573d2a22b9.png)
1标签：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7d08aa7e9c6445aca1fbfa1bbb8b0924.png)好工业仪表数据集（包含指针式仪表和数字式仪表），使用 YOLO 框架完成从数据准备到性能评估的全流程代码实现。
：

---

### **1. 数据准备**

#### 1.1 数据集结构
假设数据集目录如下：
```
/dataset
    /images
        img1.jpg
        img2.jpg
        ...
    /labels_xml
        img1.xml
        img2.xml
        ...
    /labels_txt
        img1.txt
        img2.txt
        ...
```

#### 1.2 标签格式转换 (XML 转 TXT)
YOLO 使用 `.txt` 格式的标签文件，每个文件包含目标的类别和归一化坐标。

创建 `convert_xml_to_yolo.py`：
```python
import os
import xml.etree.ElementTree as ET

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_file, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    out_file = open(os.path.join(output_dir, os.path.splitext(os.path.basename(xml_file))[0] + '.txt'), 'w')

    for obj in root.findall('object'):
        cls = obj.find('name').text
        if cls not in ['pointer', 'digital']:  # 假设类别为 pointer 和 digital
            continue
        cls_id = 0 if cls == 'pointer' else 1
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(f"{cls_id} {' '.join(map(str, bb))}\n")
    out_file.close()

if __name__ == "__main__":
    xml_dir = 'dataset/labels_xml'
    txt_dir = 'dataset/labels_txt'
    os.makedirs(txt_dir, exist_ok=True)

    for xml_file in os.listdir(xml_dir):
        convert_annotation(os.path.join(xml_dir, xml_file), txt_dir)
```

---

### **2. 数据划分**

将数据集划分为训练集、验证集和测试集。

创建 `split_dataset.py`：
```python
import os
import random
from shutil import copyfile

def split_dataset(image_dir, label_dir, train_ratio=0.8, val_ratio=0.1):
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(images)

    total_images = len(images)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)

    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    datasets = [('train', train_images), ('val', val_images), ('test', test_images)]

    for dataset_name, image_list in datasets:
        os.makedirs(os.path.join(image_dir, dataset_name), exist_ok=True)
        os.makedirs(os.path.join(label_dir, dataset_name), exist_ok=True)

        for img in image_list:
            img_path = os.path.join(image_dir, img)
            label_path = os.path.join(label_dir, img.replace('.jpg', '.txt').replace('.png', '.txt'))

            if os.path.exists(img_path) and os.path.exists(label_path):
                copyfile(img_path, os.path.join(image_dir, dataset_name, img))
                copyfile(label_path, os.path.join(label_dir, dataset_name, img.replace('.jpg', '.txt').replace('.png', '.txt')))

if __name__ == "__main__":
    image_dir = 'dataset/images'
    label_dir = 'dataset/labels_txt'
    split_dataset(image_dir, label_dir)
```

---

### **3. 环境搭建**

#### 3.1 创建虚拟环境
```bash
conda create -n industrial_instrument python=3.9
conda activate industrial_instrument
```

#### 3.2 安装依赖
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install ultralytics opencv-python pandas matplotlib
```

---

### **4. 数据配置**

创建 `data.yaml` 文件：
```yaml
train: ../dataset/images/train
val: ../dataset/images/val
test: ../dataset/images/test

nc: 2  # 类别数：指针式仪表和数字式仪表
names: ['pointer', 'digital']
```

---

### **5. 模型训练**

#### 5.1 启动训练
```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640 batch=16 name=instrument_detection
```

#### 5.2 配置超参数
可以在训练命令中指定超参数，例如：
```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640 batch=16 lr0=0.01 momentum=0.9 weight_decay=0.0005 name=instrument_detection
```

---

### **6. 模型推理**

#### 6.1 单张图片推理
```python
from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO("runs/detect/train/weights/best.pt")

# 推理单张图片
img_path = "dataset/images/test/img1.jpg"
results = model(img_path)

# 可视化结果
for r in results:
    im_array = r.plot()
    cv2.imshow("Result", im_array)
    cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 6.2 批量推理
```python
import os
from ultralytics import YOLO

# 加载模型
model = YOLO("runs/detect/train/weights/best.pt")

# 推理批量图片
test_dir = "dataset/images/test"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    results = model(img_path)
    for r in results:
        im_array = r.plot()
        cv2.imwrite(os.path.join(output_dir, img_name), im_array)
```

---

### **7. 性能评估**

#### 7.1 验证模型性能
```bash
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml
```

#### 7.2 绘制 PR 曲线和混淆矩阵
YOLO 自动生成评估报告，路径为 `runs/detect/train/`。

---

jgck .仅供参考。
