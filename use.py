from ultralytics import YOLO
import cv2

def main():
    try:
        # 加载训练好的模型
        model = YOLO(r"E:\yolov10\yolov10-main\runs\detect\train11\weights\best.pt")  # 替换为你的模型路径

        # 在图像上进行推理
        results = model.predict(source=r"E:\BackGround\2022329600041-1.jpg",  # 替换为你的图像路径
                                conf=0.25,  # 置信度阈值
                                save=True)  # 是否保存预测结果

        # 打印结果的类型和内容
        print("Type of results:", type(results))
        print("Results:", results)

        # 处理结果
        for result in results:
            if isinstance(result, dict):  # 检查结果是否为字典
                boxes = result.get('boxes', [])  # 使用 get 方法以避免 KeyError
                for box in boxes:
                    # 提取框的属性
                    x1, y1, x2, y2 = box.xyxy[0]  # 框的坐标
                    conf = box.conf[0]  # 置信度
                    cls = box.cls[0]  # 类别
                    print(f"Detected class: {cls}, Confidence: {conf}, Coordinates: {x1, y1, x2, y2}")

                # 如果结果字典中包含保存路径
                if 'path' in result:
                    output_image = cv2.imread(result['path'])
                    cv2.imshow("Predicted Image", output_image)
                    cv2.waitKey(0)  # 等待按键
                    cv2.destroyAllWindows()
            else:
                print("Unexpected result format:", result)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
