# PCB-Anomaly-Detection

This is The 2024 CUK Competition on Data Analytics!

Pixel과 함께하는 제4회 가톨릭대학교 데이터분석 경진대회

## 팀원
## Info
팀장: [박수형] (pshpulip22@catholic.ac.kr) <br>
팀원: [김현이] (kh21234@catholic.ac.kr), [김호범] (hobeom2001@catholic.ac.kr) [윤채원] (codnjs026@catholic.ac.kr)

## 대회 설명
일정: 경진대회 참가팀 및 팀별 계획서 접수: 4월 4일 ~ 4월 18일 18시
팀별 최종 보고서 및 발표자료 제출: 5월 9일 18시
트랙:
(1) 이공 및 자연계열 대상: 제공된 데이터에 대해 규칙 기반 모델부터 딥러닝 모델까지 다양한 인공지능 방법론을 적용하여 주어진 문제를 효과적이고 효율적으로 해결할 수 있는 인공지능 모델 개발
- 데이터: Pixel에서 제공하는 PCB 생산 공정 데이터로 정상 및 불량 PCB 이미지를 포함
- 태스크: 이미지 분석을 통해 불량 PCB 탐지를 목표로 하며, 학습에 사용된 PCB 뿐만아니라 학습에 사용되지 않은 새로운 PCB에 대해서도 안정적으로 동작하는 일반화 모델 개발
- 데이터 배포: 아래 링크를 통해 확인
https://drive.google.com/file/d/1cyzxeFGClEIzNSYSS5EXkzyf6NaYAB2u/view?usp=sharing <br>
※제공되는 데이터는 사전 학습 및 데이터 확인을 위한 샘플 데이터입니다. 데이터는 OK, NG 영상으로 구성되어 있으며, OK는 이상 없는 이미지, NG는 결함이 있는 이미지입니다. 본 데이터를 활용하여 OK와 NG를 binary classification하는 것이 본 대회의 목적입니다."

 
(2) 인문 및 사회계열 대상: 데이터를 수집하고 규칙 기반 모델부터 딥러닝 모델까지 다양한 인공지능 방법론을 활용해 분석하여, 정책 제언이나 시장전략 제안 등 유의미한 분석결과를 도출

시상:
최우수상(500,000원) 트랙별 1팀
우수상(300,000원) 트랙별 2팀
장려상(200,000원) 트랙별 3팀


## :heart: Collaborator:heart:
   
|[<img src="https://avatars.githubusercontent.com/u/115800583?v=4" width = 100>](https://github.com/hye0n22)|[<img src="https://avatars.githubusercontent.com/u/115800583?v=4" width = 100>](https://github.com/Coding-Child)|[<img src="https://avatars.githubusercontent.com/u/115800583?v=4" width = 100>](https://github.com/ycodnjs)|
|-|-|-|
|김현이|박수형|윤채원|




## Requirements
```
pytorch 2.3.0+cu12 => pytorch.org
wandb 0.16.6 => pip install wandb
```

## Quick Start
```
python run.py [-h] [--seed SEED] [-m MODEL_TYPE] [-lr LEARNING_RATE] [-e NUM_EPOCHS] [-b BATCH_SIZE] [-sch SCHEDULER] [-opt OPTIMIZER]\
              [-w WARMUP_STEPS] [-g GAMMA] [-d DROP_PROB] [-s STEP_SIZE]\
              [-trn TRAIN_DATA] [-val VAL_DATA] [-tst TEST_DATA] [-p PRETRAINED] [-mp MODEL_PATH] [-sp SAVE_PATH]
```
```
python run.py -lr 1e-4 -e 250 -b 256 -opt adam -g 0.1 -p False -d 0.1 -m resnext
```
