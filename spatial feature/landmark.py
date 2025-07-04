import cv2
import mediapipe as mp
from matplotlib import pyplot as plt
from mediapipe_face import draw_landmarks_on_image,plot_face_blendshapes_bar_graph

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=r"D:\Code\python\Reshow\MediaPipe+VGG-BiLSTM\resource\mediapipe\face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE)

with FaceLandmarker.create_from_options(options) as landmarker:
    img = mp.Image.create_from_file(r"D:\Code\python\Reshow\MediaPipe+VGG-BiLSTM\resource\picture\business-person.png")
    detection_result = landmarker.detect(img)
    # 绘制landmark可视化
    annotated_img = draw_landmarks_on_image(img.numpy_view(), detection_result)
    cv2.imshow("landmark",cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)  # 0表示无限等待，直到用户按下任意键
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
    # 绘制表情系数
    # plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])
    # 输出面部矩阵
    # print(detection_result.facial_transformation_matrixes)


