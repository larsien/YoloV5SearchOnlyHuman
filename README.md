# YoloV5SearchOnlyHuman
https://velog.io/@larsien/YoloV5-%EC%82%AC%EB%9E%8C%EB%A7%8C-%EC%B0%BE%EC%95%84%EB%B3%B4%EC%9E%90

![image](https://user-images.githubusercontent.com/4662566/190848847-e8d0a5cf-2cc9-454e-b31c-d16d36e0a538.png)

0. 목표
* 웹캠으로 사람을 모니터링하는 시스템 만들때

* YoloV5 이용해 사람을 추적할 AI를 만들어보자

나는 YoloV5로 사람 인식하기를 진행하고, 그 외는 다른 분이 하기로 했다.

# 1. YoloV5?
> You only look once (YOLO) is a state-of-the-art, real-time object detection system

실시간 객체 인식 시스템이다.
기본 적으로 80개의 이미지를 구분할 수 있다.

![image](https://user-images.githubusercontent.com/4662566/190848842-b15a1ca2-f08c-47ac-a7d7-6031dfc01101.png)

80개의 Label 중 사람만 찾도록 바꿔보자

## 1.1 train.py
이미지를 학습시키는 파일.
파일을 살펴보자
~~~
def parse_opt(known=False):
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
~~~ 
-- data : 학습할 데이터 정보가 있는 파일
-- img : 이미지 크기. 디폴트 640
-- batch-size : 한 번에 학습하는 데이터 크기. 디폴트 16. -1은 자동 배치
-- epochs : 전체 데이터를 몇 번 사용해서 1번 학습하는지. 디폴트 300
-- cfg : 위에서 정한 모델 크기. 디폴트 경로는 model.yaml
-- weights : 미리 학습된 모델로 학습(전이학습)할 경우 . 디폴트는 yolov5s.pt
-- name : 학습된 모델의 이름

## 1.2 data.yaml
train: /content/yolov5/train/images
val: /content/yolov5/valid/images
nc: 80
names:['aeroplane', 'apple', 'backpack', 'banana', 'baseball bat', ...]
train : 학습할 이미지 경로
val : 학습 후 훈련된 weight로 정답을 체크해볼 이미지 경로
nc : number of case. 기본적으로 80개의 이미지 인식
names : nc의 이름에 해당하는 class 값들. nc와 갯수가 맞아야 함.

설명을 보면 경로는 train, val의 path는 3가지 형태로 지정 가능

dir: path/to/imgs
file: path/to/imgs.txt
list: [path/to/imgs1, path/to/imgs2, ..]

# 2. 사전 작업
## 2.1 data.ymal 경로 변경
data.yaml 경로를 절대경로로 잘 인식하게 변경하고 person만 인식할 것이므로 다른건 지움
~~~
train: /content/yolov5/train/images
val: /content/yolov5/valid/images
nc: 1
names: ['person']
import yaml 
path = '/content/yolov5/data.yaml'
with open(path, 'r') as f : 
  data = yaml.load(f, Loader=yaml.FullLoader) 
  data['train'] = '/content/yolov5/train/images'
  data['val'] = '/content/yolov5/valid/images'
  data['nc'] = 1 
  data['names'] = ['person']

  with open(path, 'w')  as f: 
    yaml.dump(data, f)
    print(data )
~~~

## 2.2 label 파일들 내용 변경
정답이 기록된 파일. 하나만 살펴보자.

/content/yolov5/train/labels/000000439676_jpg.rf.d47b5404b3cdae2c67b278083f44c303.txt
48 0.32125 0.41796875 0.0825 0.0515625

/content/yolov5/train/labels/000000439676_jpg 이미지는 라벨이 48인 이미지가 있고 위치는 0.32125 0.41796875 0.0825 0.0515625이다.
순서대로 x중심좌표,y중심좌표, width, heigth 이며, 정규화 되어 소숫점으로 표시된다.
![image](https://user-images.githubusercontent.com/4662566/190848838-5b54e162-dba0-49c0-ac49-30da9b5b5a59.png)




사람만 인식하게 바꿔야 하므로 사람에 해당하는 48번만 라벨 파일에서 찾아 저장한다.
파일 하나만 변경되나 테스트해보자.
~~~
from glob import glob 

train_label_list = glob('/content/yolov5/train/labels/*.txt')
for file in train_label_list:
  print(file)
  f = open(file, "rt") 
  person = []
  for line in f:
    if line.startswith("48"):
      person.append(line)
  f.close()
  
  f2 = open(file, 'wt')
  for p in person:
    f2.write(p)
  f2.close()
  f3 = open(file, "r")
  for line in f3:
    print(line)
  f3.close()
  break
~~~

잘 바뀌었다면 전체 파일에 적용
~~~
from glob import glob 
import os 
train_label_list = glob('/content/yolov5/train/labels/*.txt')
for path in train_label_list:
  print(path)
  f = open(path, "rt") 
  person = []
  for line in f:
    if line.startswith("48"):
      person.append(line)
  f.close()
  if len(person) == 0 : # person 이 없으면 삭제한다 
    os.remove(path)
    os.remove(path.replace(".txt", ".jpg").replace("/labels/","/images/"))
  else : 
    f2 = open(path, 'wt')
    for p in person:
      f2.write(p.replace("48","0",1))
    f2.close()
~~~
잘 변경 됬나 샘플 확인

> !cat /content/yolov5/train/labels/000000009465_jpg.rf.3d5a6b94f8c1afdc004f2641cd578912.txt
> 0 0.16484375 0.7621247113163973 0.04375 0.16166281755196305
> 0 0.24921875 0.7829099307159353 0.0296875 0.16628175519630484
> 0 0.2234375 0.7736720554272517 0.0359375 0.17090069284064666
> 0 0.19765625 0.76905311778291 0.03125 0.16628175519630484

## 2.3 학습하기
훈련시키자. 일해라 핫산!

> !python /content/yolov5/train.py --img 640 --batch 16 --epochs 20 --data /content/yolov5/data.yaml --cfg /content/yolov5/models/yolov5s.yaml --weights yolov5s.pt --name yolov5_coco

코랩에서 gpu 써도 한 epoch 마다 30여분 걸린다. 3번 돌아도 왠만큼 잘 뽑혀서 3번 째 weight를 사용.

# 3. detect.py 내용 변경
사람을 찾았을 때, 라벨 없애기
box의 좌표를 찾아 중심 좌표를 로깅
현재 이미지의 사이즈
## 3.1 사람을 찾았을 때, 라벨 없애기
hide_labels라는 파라미터가 있다. 파라미터 입력할 때 True로 줘도 안되서 소스에서 변경함.

hide_labels = True
## 3.2 box의 좌표를 찾아 중심 좌표를 로깅
xyxy 에 좌표가 저장되 있다. 여러명을 찾을 수 있으므로 모든 박스의 좌표를 합해 평균을 찾는다.
~~~
x_positions = []
y_positions = []
for *xyxy, conf, cls in reversed(det): 
  x_positions.append(xyxy[0])
  x_positions.append(xyxy[2])
  y_positions.append(xyxy[1])
  y_positions.append(xyxy[3])
~~~
## 3.3 현재 이미지 사이즈
imgsz 에 저장되 있다.

>imgsz[0] # width 
>imgsz[1] # height

# 4. 결과
## 3.1 기존 결과

![image](https://user-images.githubusercontent.com/4662566/190848804-0740aa84-3ebe-48b3-bd71-151aae447f81.png)

## 3.2 사람만 학습 한 weight 사용한 최종 결과
![image](https://user-images.githubusercontent.com/4662566/190848811-94e6b89f-c960-47f0-a362-4644d595cdec.png)


# 5. 트러블슈팅
학습 중 에러
~~~
Traceback (most recent call last):
File "/content/yolov5/train.py", line 636, in
main(opt)
File "/content/yolov5/train.py", line 533, in main
train(opt.hyp, opt, device, callbacks)
File "/content/yolov5/train.py", line 376, in train
compute_loss=compute_loss)
File "/usr/local/lib/python3.7/dist-packages/torch/autograd/grad_mode.py", line 28, in decorate_context
return func(*args, **kwargs)
File "/content/yolov5/val.py", line 314, in run
maps[c] = ap[i]
IndexError: index 1 is out of bounds for axis 0 with size 1
~~~
ap_class 가 80개로 되있어서 생기는 오류. 어디서 세팅해주는지 모르겠는데, 어차피 클래스는 사람 1개밖에 없으므로 315라인에서 다음과 같이 바꿔준다.

for i , c in enumerate([0]):

# 참고 링크
- 참고 샘플 https://minding-deep-learning.tistory.com/19
- 박스를 어떻게 그리나 예제 https://towardsdatascience.com/how-to-create-real-time-face-detector-ff0e1f81925f
- 좌표 이해 https://stackoverflow.com/questions/52455429/what-does-the-coordinate-output-of-yolo-algorithm-represent
