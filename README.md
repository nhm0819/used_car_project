### used_car_project

##### Data : https://www.kaggle.com/austinreese/craigslist-carstrucks-data

#### 중고차 가격 예측 머신러닝 모델링

- SVM
- Random Forest
- XGBoost


###### 결측치 처리 방법
1. 가격과 필요 없는 특성 제거 (이미지 URL 등)
2. 예측 불가능한 특성 제거 (차량 색상 등)
3. 그 외의 결측치 대체법 : IterativeImputer(estimator=RandomForestClassifier())
