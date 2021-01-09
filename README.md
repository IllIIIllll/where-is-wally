# where-is-wally
월리를 찾아라
![Preview](preview.png)

# Tree
```
.
├── where-is-wally/
│   ├── data/
│   │   ├── imgs/
│   │   │   ├── bnd_box/           : 학습 이미지에서의 월리 위치 정보
│   │   │   ├── original_imgs/     : 학습 이미지
│   │   │   ├── predictions/       : 예측 결과 이미지
│   │   │   ├── target_imgs/       : 타겟 이미지
│   │   │   ├── imgs.npy           : 모든 학습 이미지 numpy 배열
│   │   │   ├── trgs.npy           : 모든 타겟 이미지 numpy 배열
│   │   │   ├── wally_sub_imgs.npy : 224x224 크기의 모든 학습 서브 이미지 numpy 배열
│   │   │   └── wally_sub_trgs.npy : 224x224 크기의 모든 타겟 서브 이미지 numpy 배열
│   │   ├── create_sumimages.py    : 224x224 크기의 서브 이미지 생성
│   │   ├── generator.py           : 제너레이터
│   │   ├── make_targets.py        : 타겟 이미지 생성
│   │   └── preprocess.py          : 이미지 전처리
│   ├── models/                    : 모델
│   └── utils/
│       ├── params.py              : 상수
│       └── tiramisu.py            : https://arxiv.org/abs/1611.09326
├── predict.py                     : 예측
└── train.py                       : 모델 훈련
```

# Skills
- python 3.7
- Numpy
- TensorFlow
- Pillow
- Beautiful Soup
- imageio

# How to use  
### Set up
타겟 이미지 생성
```python 
$ python make_targets.py
```
전처리
```python 
$ python preprocess.py
```
서브 이미지 생성
```python 
$ python create_subimages.py
```

### Training
```python 
$ python train.py
```
또는 아래 옵션 설정 가능
```
--imgs : 학습 데이터 경로
--trgs : 타겟 데이터 경로
--wally-sub-imgs : 서브 학습 데이터 경로
--wally-sub-trgs : 서브 타겟 데이터 경로
--model : 모델 저장 경로
--tot-bs : batch size
--prop : 월리가 있는 이미지와 없는 이미지 비율
         (0.75라면 3:1의 비율을 갖는다.)
--epochs : epochs
--spe : steps per epoch
```

### Predicting
```
$ python predict.py test.jpg
```
또는 아래 옵션 설정 가능
```
--model : 모델 경로
--output : 결과 저장 경로
--size : 이미지 크기
```
  
예측할 이미지는 공백으로 구분하여 입력 가능
```
example :
$ python predict.py 1.jpg 2.jpg 3.jpg
```
  
결과는 기본적으로 `data/imgs/predictions` 폴더 하위에  
현재 날짜와 시간으로 된 폴더를 생성하여 저장됨

<br>

---
  
<br>

#### Open Source License는 [이곳](NOTICE.md)에서 확인해주시고, 문의사항은 [Issue](https://github.com/IllIIIllll/where-is-wally/issues) 페이지에 남겨주세요.
