# Aruco Marker Localization
- 현재 Kalman filter는 bicycle model이 들어가 있지 않습니다.
- map 사이즈는 600cm x 550cm (가로 x 세로) 입니다.
- Mean Average Filter의 queue는 크기가 5입니다.
- x_global_avg, y_global_avg => aruco marker들을 통해 얻은 x, y 좌표입니다.
- x_global_avg, y_global_avg 값을 확인해보면 outlier들을 볼 수 있으며, outlier 조건을 자유롭게 추가해주시면 됩니다.
