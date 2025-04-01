# 음성과 얼굴 인식을 활용한 아이 감정 파악 및 위험 상황 감지 시스템

## 개요
본 프로젝트는 음성 데이터와 얼굴 인식을 활용하여 아이의 감정을 파악하고, 위험한 상황을 감지하는 시스템입니다. 음성 데이터를 분석하여 감정을 분류하고, OpenCV를 이용하여 얼굴 표정을 분석하는 방식으로 아이의 상태를 종합적으로 판단합니다.

## 주요 기능
1. **음성 데이터 수집 및 감정 분석**
   - 아이의 음성을 녹음하여 `wav` 파일로 저장
   - 음성 데이터를 MFCC 특징 벡터로 변환하여 머신러닝 모델(Random Forest 등)로 감정 분류
   - 감정 상태: 배고픔(hungry), 피곤함(tired), 불편함(discomfort), 기타(etc)
   
2. **얼굴 인식을 통한 감정 분석**
   - OpenCV를 이용하여 실시간 얼굴 인식 수행
   - CNN 모델을 활용하여 얼굴 표정을 감정으로 분류
   - 감정 상태: 행복, 슬픔, 화남, 놀람 등

3. **위험 상황 감지 및 알림 시스템**
   - 음성과 얼굴 인식 결과를 통합하여 위험 상태를 감지
   - MQTT 프로토콜을 사용하여 감지된 정보를 네트워크로 전송
   - 감지된 정보를 서버로 전송하여 실시간 모니터링 가능

## 기술 스택
- **음성 분석:** `Librosa`, `Wave`, `Pandas`, `NumPy`, `scikit-learn`
- **얼굴 인식:** `OpenCV`, `Deep Learning (CNN)`
- **머신러닝 모델:** `RandomForestClassifier`, `LogisticRegression`, `DecisionTreeClassifier`, `SVM`
- **데이터 저장:** `pickle`
- **네트워크 통신:** `MQTT (PubSubClient)`
- **하드웨어:** `Arduino`, `Ethernet 모듈`

## 실행 방법
### 1. 음성 데이터 수집 및 모델 학습
```bash
python record_audio.py  # 음성 녹음
python train_model.py  # 머신러닝 모델 학습 및 저장
```

### 2. 얼굴 인식 실행
```bash
python face_detection/face_recognition.py  # OpenCV를 이용한 얼굴 감정 분석 실행
```

### 3. 실시간 감정 분석 및 위험 감지
```bash
python classify_audio.py  # 실시간 음성 감정 분석 실행
```

### 4. 아두이노를 통한 MQTT 메시지 송수신
- `mqtt_client.ino` 파일을 아두이노 IDE에서 업로드 후 실행

## 기대 효과
- 아이의 감정을 보다 정확하게 파악하여 부모나 보호자가 신속하게 반응할 수 있도록 지원
- 실시간으로 아이의 상태를 모니터링하여 위험 상황을 빠르게 감지
- IoT 기술을 활용하여 스마트한 감정 분석 및 경고 시스템 구축

## 참고 자료
- OpenCV 공식 문서: https://opencv.org/
- Librosa 공식 문서: https://librosa.org/doc/main/index.html
- Scikit-learn 공식 문서: https://scikit-learn.org/stable/

본 프로젝트는 [제 15회 소외된 이웃과 함께하는 창의경진 대회] 출품 목적으로 제작되었습니다.

## 수상

- [창의상] 수상

