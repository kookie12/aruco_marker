import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from collections import deque
import os

''' '''

class Kalman_filter():

    def __init__(self):
        # 칼만 필터 파라미터 초기화
        self.dt = 1  # 샘플링 시간
        self.A = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]])  # 상태 전이 행렬
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # 측정 행렬
        self.Q = np.eye(4) * 0.1  # 프로세스 노이즈 공분산
        self.R = np.eye(2) * 5    # 측정 노이즈 공분산
        self.x = np.array([[315], [490], [0], [0]])  # 초기 상태
        self.P = np.eye(4)  # 초기 추정 오차 공분산

    def kalman_filter(self, x, P, measurement):
        # 예측 단계
        self.x = self.A @ x
        self.P = self.A @ P @ self.A.T + self.Q
        
        # 측정 업데이트
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = measurement - self.H @ x
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.x, self.P 

    # update 함수에 칼만 필터 업데이트 로직 추가
    def update(self, x_global_avg, y_global_avg): # frame_number, marker_corners, marker_ids, car_scatter
        # global x, P

        # 새로운 측정치에 대한 칼만 필터 업데이트
        measurement = np.array([[x_global_avg], [y_global_avg]])
        x, P = self.kalman_filter(self.x, self.P, measurement)

        # 칼만 필터로 필터링된 위치 정보를 추출하여 차량 위치 업데이트
        x_filtered, y_filtered = x[0, 0], x[1, 0]
        return x_filtered, y_filtered

class moving_average_filter():
    # 최근 5개 값을 유지하는 deque를 생성
    def __init__(self):
        self.x_deque = deque(maxlen=3)
        self.y_deque = deque(maxlen=3)
        
    # 새로운 값을 추가하고 평균을 계산하는 함수
    def update_avg(self, new_x, new_y):
        # 새 값을 deque에 추가
        self.x_deque.append(new_x)
        self.y_deque.append(new_y)
        
        # 평균 계산
        x_avg = np.mean(list(self.x_deque))
        y_avg = np.mean(list(self.y_deque))
        
        return x_avg, y_avg

    # 이상치 판단 기준을 정의(예: Z-Score, IQR 등 사용 가능)
    def is_outlier(self, values, new_value):
        # if len(values) < 5:
        #     # 최소 5개 값이 있어야 이상치를 판단할 수 있음
        #     return False
        threshold = 10  # 예를 들어, Z-Score 기준으로 +/-2 이상이면 이상치로 판단
        mean_val = np.mean(values)
        # std_val = np.std(values)
        # z_score = (new_value - mean_val) / std_val
        
        if abs(mean_val - new_value) > threshold:
            return True
        
        # 갑자기 부호가 반대가 되어버리는 경우를 제거하자
        # if mean_val * new_value < 0:
        #     return False
        
        #return abs(z_score) > threshold

    # 이상치를 검출하고 중간값을 계산하는 함수
    def update_median_if_outlier(self, new_x, new_y):
        
        # x 값 이상치 검출
        if self.is_outlier(self.x_deque, new_x):
            x_median = np.median(list(self.x_deque))
            self.x_deque.append(x_median)  # 이상치 대신 중간값 사용
        else:
            self.x_deque.append(new_x)
        
        # y 값 이상치 검출
        if self.is_outlier(self.y_deque, new_y):
            y_median = np.median(list(self.y_deque))
            self.y_deque.append(y_median)  # 이상치 대신 중간값 사용
        else:
            self.y_deque.append(new_y)
        
        # 중간값 계산
        x_median = np.median(list(self.x_deque))
        y_median = np.median(list(self.y_deque))
        
        return x_median, y_median


def update_scatter(scatter, new_data, color):
    scatter.set_offsets(new_data)
    scatter.set_color(color)
    return scatter

if __name__ == '__main__':


    Video_filename = 'aruco_video_1103.MOV'
    video_output_path_1 = 'aruco_detection_1103_kalman_avg.mp4'  
    # video_output_path_2 = 'Aruco-Localization/aruco_localization.mp4'
    os.makedirs('aruco_localization_1103_kalman_avg', exist_ok=True)

    # save video
    fourcc = cv2.VideoWriter_fourcc(*'H264')

    cap = cv2.VideoCapture(Video_filename)

    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    cameraMatrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    distCoeffs = np.zeros((4,1))

    marker_positions = {
        0: np.array([0, (135/2)*7, 0]),  # 예: ID 1의 마커가 월드 좌표계의 원점에 위치한다고 가정
        1: np.array([0, (135/2)*5, 0]),  # 예: ID 2의 마커가 x축 방향으로 1 단위만큼 이동한 위치에 있다고 가정
        2: np.array([0, (135/2)*3, 0]),   # 예: ID 3의 마커가 y축 방향으로 1 단위만큼 이동한 위치에 있다고 가정
        3: np.array([0, (135/2)*1, 0]),
        4: np.array([int(72.5) + 5, 0, 0]),
        5: np.array([int(147.5) + 2*5, 0, 0]),
        6: np.array([int(222.5) + 3*5, 0, 0]),
        7: np.array([int(297.5) + 4*5, 0, 0])
    }

    # 초기 scatter 객체 생성
    # color: orange, drakorange, deepskyblue, lightskyblue, skyblue

    # 지도 이미지를 로드합니다.
    map_image = plt.imread('map.png')
    map_img = cv2.resize(map_image, (600, 550))

    # 차량 크기
    car_h = 70 # car_img.shape[1] # 700mm
    car_w = 58 # car_img.shape[0] # 580mm
    line_w = 5

    # point들 기록
    st_curve = (550, car_h) # (550, 550 - car_h)
    ed_curve = (315, 550 - (50 + line_w*2))

    parking_1 = (int(72.5) + line_w, 550 - int(482.5))
    parking_2 = (int(147.5) + 2*line_w, 550 - int(482.5))
    parking_3 = (int(222.5) + 3*line_w, 550 - int(482.5))
    parking_4 = (int(297.5) + 4*line_w, 550 - int(482.5))

    fig, ax = plt.subplots()
    # plt.figure(figsize=(12, 6))
    plt.title("Aruco Markers and Estimated Car Position: Kalman + Average Moving Filter")
    # 지도 이미지를 표시합니다.
    ax.imshow(map_img, extent=[0, 600, 0, 550])
    # set size bigger
    fig.set_size_inches(12, 6)

    # ax.imshow(map_img, extent=[0, 600, 0, 550], interpolation='nearest')

    # point 찍기
    ax.add_patch(patches.Circle(st_curve, 10, color='b', fill=True)) # 파란색   
    ax.add_patch(patches.Circle(ed_curve, 10, color='b', fill=True)) # 파란색
    ax.add_patch(patches.Circle(parking_1, 10, color='r', fill=True)) # 빨간색
    ax.add_patch(patches.Circle(parking_2, 10, color='r', fill=True)) # 빨간색
    ax.add_patch(patches.Circle(parking_3, 10, color='r', fill=True)) # 빨간색
    ax.add_patch(patches.Circle(parking_4, 10, color='r', fill=True)) # 빨간색


    car_scatter = ax.scatter([], [], s=100, c='darkorange', label='Estimated Car Position', marker='o')
    markers_scatter_0 = ax.scatter([], [], s=100, c='deepskyblue', label='Aruco Markers id 0', marker='s')
    markers_scatter_1 = ax.scatter([], [], s=100, c='deepskyblue', label='Aruco Markers id 1', marker='s')
    markers_scatter_2 = ax.scatter([], [], s=100, c='deepskyblue', label='Aruco Markers id 2', marker='s')
    markers_scatter_3 = ax.scatter([], [], s=100, c='deepskyblue', label='Aruco Markers id 3', marker='s')
    markers_scatter_4 = ax.scatter([], [], s=100, c='deepskyblue', label='Aruco Markers id 4', marker='s')
    markers_scatter_5 = ax.scatter([], [], s=100, c='deepskyblue', label='Aruco Markers id 5', marker='s')
    markers_scatter_6 = ax.scatter([], [], s=100, c='deepskyblue', label='Aruco Markers id 6', marker='s')
    markers_scatter_7 = ax.scatter([], [], s=100, c='deepskyblue', label='Aruco Markers id 7', marker='s')

    plt.axis([0, 600, 0, 550])

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    fps = cap.get(cv2.CAP_PROP_FPS)  # 원본 비디오의 FPS를 가져옴
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 3)  # 원본 비디오의 너비를 가져옴
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 3)  # 원본 비디오의 높이를 가져옴
    out = cv2.VideoWriter(video_output_path_1, fourcc, fps, (width, height))  # 저장할 비디오 설정 (FPS와 해상도는 원본과 동일)

    Kalman = Kalman_filter()
    MAF = moving_average_filter()

    idx = 0
    
    # plt.figure("Global Map")
    # plt.title("Aruco Markers and Estimated Car Position: ")
    
    x_deque = deque(maxlen=3)
    y_deque = deque(maxlen=3)
    z_deque = deque(maxlen=3)
    
    flag = False
    while True:
        ret, image = cap.read()
        if ret: 
            image = cv2.resize(image, (image.shape[1]//3, image.shape[0]//3))

            corners, ids, _ = aruco.detectMarkers(image, arucoDict, parameters=parameters)

            if len(corners) > 0:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)

                plt.title("Kalman + Average Moving Filter")
                plt.grid(True)
                
                # 아루코 마커의 위치를 플롯
                # for key, value in marker_positions.items():
                #     plt.scatter(value[0], value[1], s=100, label=f'Marker {key}', marker='s')

                # 아루코 마커의 위치를 플롯
                marker_coords = np.array([value[:2] for key, value in marker_positions.items()])
                update_scatter(markers_scatter_0, marker_coords, color='deepskyblue')
                update_scatter(markers_scatter_1, marker_coords, color='deepskyblue')
                update_scatter(markers_scatter_2, marker_coords, color='deepskyblue')
                update_scatter(markers_scatter_3, marker_coords, color='deepskyblue')
                update_scatter(markers_scatter_4, marker_coords, color='deepskyblue')
                update_scatter(markers_scatter_5, marker_coords, color='deepskyblue')
                update_scatter(markers_scatter_6, marker_coords, color='deepskyblue')
                update_scatter(markers_scatter_7, marker_coords, color='deepskyblue')
            
                cam_positions = []
                for i in range(len(ids)): # ids = array([[1], [0], [2]])
                    R, _ = cv2.Rodrigues(rvecs[i])
                    camR = np.transpose(R)
                    T = np.transpose(tvecs[i])
                    camT = np.transpose(np.matmul(-camR, T)) # 월드 좌표계에서의 카메라의 위치를 나타내는 벡터 camT
                    
                    # 카메라 위치 보정
                    if ids[i] in list(marker_positions.keys()):
                        camT += marker_positions[ids[i].item()]
                    
                    cam_positions.append(camT) 

                    # image = aruco.drawDetectedMarkers(image, [corners[i]], [ids[i]], borderColor=(0, 255, 0))
                    image = aruco.drawDetectedMarkers(image, corners, ids, borderColor=(0, 255, 0))
                    image = cv2.drawFrameAxes(image, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.03)


                print("camT: ", camT)
                
                # 자동차의 예상 위치를 플롯 => 3개 마커 고려한거임
                avg_position = np.mean(cam_positions, axis=0)[0][:2]

                x_global_list, y_global_list = [], []
                
                for i, cam_position in enumerate(cam_positions):
                    marker_id = ids[i].item()
                    m_idx = [item[0] for item in ids.tolist()].index(marker_id)
                    # # marker가 다 인식되지 않을 경우
                    # if len(ids) != 3:
                    #     break
                
                    x_marker2car, y_marker2car, z_marker2car = cam_positions[m_idx][0]
                    x_marker, y_marker, z_marker = marker_positions[marker_id]
                
                    if marker_id == 0 or marker_id == 1 or marker_id == 2 or marker_id == 3:  
                        
                        
                        if flag == False: # 위 값들이 튀는게 좀 있다 => filtering 
                            if abs(np.mean(x_deque) - x_marker2car) > 5:
                                x_marker2car = np.mean(x_deque)
                                
                            if abs(np.mean(y_deque) - y_marker2car) > 5:
                                y_marker2car = np.mean(y_deque)
                                
                            if abs(np.mean(z_deque) - z_marker2car) > 5:
                                z_marker2car = np.mean(z_deque)
                        
                            x_deque.append(x_marker2car)
                            y_deque.append(y_marker2car)
                            z_deque.append(z_marker2car)
                        
                        x_global = z_marker2car*10
                        y_global = y_marker + x_marker2car*10
                        
                    # TODO: marker_id가 0-3이 보이다가 4로 넘어갈 때 확 불안정해진다
                    else: # marker_id = 6 -> x=239, y=-4.9, z=27.24
                        
                        flag = True
                        
                        x_global = x_marker + y_marker2car*10
                        y_global = z_marker2car*10
                    
                    x_global_list.append(x_global)
                    y_global_list.append(y_global)
                    
                x_global_avg = np.mean(x_global_list)
                y_global_avg = np.mean(y_global_list)

                # scatter 객체의 데이터와 색상 업데이트
                # update_scatter(car_scatter, [[x_global_avg, y_global_avg]], color='darkorange')
                x_filtered, y_filtered = Kalman.update(x_global_avg, y_global_avg) 
                
                # moving average filter
                # x_avg, y_avg = MAF.update_median_if_outlier(x_filtered, y_filtered)
                x_avg, y_avg = MAF.update_avg(x_filtered, y_filtered)
                
                update_scatter(car_scatter, [[x_avg, y_avg]], color='darkorange')

                plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.5))
                
                plt.pause(0.000000001)
                plt.axis([0, 600, 0, 550])
                # plt.draw()
                
                # set plt more big
        

            cv2.imshow('Video Out', image)
            if cv2.waitKey(1) % 0xFF == ord('q'):
                break

            # save video
            out.write(image)
            
            plt.savefig(f'aruco_localization_1103_kalman_avg/{idx}.png')
            idx += 1
        else:
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    # out_2.release()
