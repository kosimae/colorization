# colorization
흑백의 이미지에서 color을 추출하는 프로그램

- http://places2.cspython train_placesCNN.py -a resnet18 /xxx/yyy/places365standard_easyformatail.mit.edu/download.html
- 첨부한 링크에서 Place365-Standard dataset (download --> samll images 256*256 validation images (501MB))
- 다운로드 받은 val_256.tar에는 PLACES365_VAL_00000001.jpg부터 PLACES365_VAL_00036500.jpg이 저장되어 있음
- 001부터 300까지 300개의 image을 training set으로 사용하고 301부터 400까지 100개의 image을 validation set으로 사용
- 프레임워크는 Google Colab을 사용

2020년 12월 19일 토요일
- colab에서 cnn 예제 실행
- 201219 폴더에 코드 첨부
- 첨부한 코드는 google colab을 사용하여 실행

2020년 12월 20일 일요일
- Colab에서 Google drive mount
- https://blog.naver.com/wideeyed/221564411127

- Python에서 tensorflow을 사용하여 model 생성
  - xor 연산을 학습하여 checkpoint로 가중치 파일 생성
  - predict 함수를 사용하여 연산 결과 출력
  - 학습한 가중치 파일을 Google drive에 저장 

2020년 12월 22일 화요일
- 12월 19일에 돌린 예제 코드에서 학습한 model에서 predict 실행
- 10000개의 test_images에 대해서 test_labels와 predict_classes을 비교했을 때 정확도는 69% (그렇게 높지 않음)
- 10번 학습했을 때와 30번 학습했을 때 정확도의 차이가 거의 없음
- model의 구조를 다르게 해볼 

앞으로 진행할 사항
  - cnn에서 model 학습 데이터 저장 및 불러오기
  - colorization 이론 정립
