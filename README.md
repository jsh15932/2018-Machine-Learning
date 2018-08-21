### 배추 가격 예측 인공지능 소프트웨어
배추 가격을 예측하는 인공지능 소프트웨어입니다.

### 참고 문헌
* [기상청 전국 기온 및 강수량](https://data.kma.go.kr/climate/StatisticsDivision/selectStatisticsDivision.do?pgmNo=158) : 기상청 정보를 토대로 채소 가격에 영향을 미치는 요인을 분석하기 위해 참고하였습니다.
* [월별 배추 가격](https://www.kamis.or.kr/customer/price/retail/period.do?action=monthly&yyyy=2018&period=10&countycode=&itemcategorycode=200&itemcode=211&kindcode=&productrankcode=&convert_kg_yn=N) : 실질적인 국내 월별 배추 가격을 분석하기 위해 참고하였습니다.
* [시계열수치입력 수치예측 모델 레시피](https://tykimos.github.io/2017/09/09/Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe/) : 시계열(Time-Series Analysis) 예측 모델 학습자료로 참고하였습니다.

### 서버 실행 명령어
```
# 플라스크 웹 서버 폴더로 이동합니다.
cd "Flask Web Server"

# 웹 서버를 실행합니다.
python server.py
```