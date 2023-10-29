from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import onnxruntime
import json
import base64

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

ort_session = onnxruntime.InferenceSession(
    "facemodel\cpu.onnx", providers=['CPUExecutionProvider'])
list_attr_en = np.array(["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
                         "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
                         "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
                         "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes",
                         "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
                         "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat",
                         "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"])
list_attr_cn = np.array(["不明显的胡子", "拱形眉毛", "有吸引力的", "眼袋", "秃头", "刘海", "大嘴唇", "大鼻子", "黑发", "金发", "模糊的", "棕发", "浓眉", "圆胖", "双下巴", "眼镜", "山羊胡子", "灰白发", "浓妆", "高高的颧骨",
                         "男性", "嘴巴张开", "胡子", "眯眯眼", "没有胡子", "鹅蛋脸", "白皮肤", "尖鼻子", "后退的发际线", "红润脸颊",
                         "鬓角", "微笑", "直发", "卷发", "耳环", "帽子", "口红", "项链", "领带", "年轻"])


def cv2_preprocess(img):
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    img = ((img / 255.0 - mean) / std)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img, dtype=np.float32)
    return img


def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))


def result_inference(input_array):
    ort_inputs = {ort_session.get_inputs()[0].name: input_array}
    ort_outs = ort_session.run(None, ort_inputs)
    possibility = sigmoid_array(ort_outs[0]) > 0.5
    result = list_attr_cn[possibility[0]]
    return result


@csrf_exempt
def face_detection(request):
    if request.method == 'POST':
        # Assuming the image data is sent as 'image' field
        data = json.loads(request.body)

        # Retrieve the image data from the parsed JSON
        image_data_base64 = data.get('image').split(',')[1]

        # Decode Base64 data to binary
        image_data = base64.b64decode(image_data_base64)
        # Convert image_data to a NumPy array (assuming image data is in the form of a NumPy array)

        # Perform face detection using your existing code
        with mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5) as face_detection:
            image = cv2.imdecode(np.frombuffer(
                image_data, np.uint8), cv2.IMREAD_COLOR)
            print(np.frombuffer(
                image_data, np.uint8))
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Process the results and perform attribute analysis
            if results.detections:
                for detection in results.detections:
                    # Process each detected face
                    mp_drawing.draw_detection(image, detection)
                    image_rows, image_cols, _ = image.shape
                    location = detection.location_data.relative_bounding_box
                    start_point = mp_drawing._normalized_to_pixel_coordinates(
                        location.xmin, location.ymin, image_cols, image_rows)
                    end_point = mp_drawing._normalized_to_pixel_coordinates(
                        location.xmin + location.width, location.ymin + location.height, image_cols, image_rows)
                    if start_point is None or end_point is None:
                        break
                    x1, y1 = start_point
                    x2, y2 = end_point
                    img_infer = image[y1 - 70:y2, x1 - 50:x2 + 50].copy()
                    img_infer = cv2_preprocess(img_infer)
                    result = result_inference(img_infer)
                    # Here, you can store or process the attribute analysis result as needed

        result = list(result)
        # Return the analysis result as JSON response
        return JsonResponse({'result': result})
    else:
        return render(request, 'face_detection.html')
