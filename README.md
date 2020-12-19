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


계획
- colab에서 다운로드 받은 파일의 위치 알아보기
- colab에서 학습한 model의 위치 알아보기
- 학습한 model을 바탕으로 predict하는 함수 
