# colorization
흑백의 이미지에서 color을 추출하는 프로그램

- http://places2.cspython train_placesCNN.py -a resnet18 /xxx/yyy/places365standard_easyformatail.mit.edu/download.html
- 첨부한 링크에서 Place365-Standard dataset (download --> samll images 256*256 validation images (501MB))
- 다운로드 받은 val_256.tar에는 PLACES365_VAL_00000001.jpg부터 PLACES365_VAL_00036500.jpg이 저장되어 있음
- 001부터 300까지 300개의 image을 training set으로 사용하고 301부터 400까지 100개의 image을 validation set으로 사용
- 프레임워크는 Google Colab을 사용


