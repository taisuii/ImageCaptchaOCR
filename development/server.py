from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import base64
import torchvision.transforms as transforms

from development import gen_ImageCaptcha, one_hot
from development.model import MyModel

# 加载模型
m = MyModel()
m.load_state_dict(torch.load("../deploy/model/22_0.0007106820558649762.pth"))
m.eval()

# Flask 应用初始化
app = Flask(__name__)

# 预处理步骤（如有必要可以自定义）
data_transform = transforms.Compose([
    transforms.ToTensor(),
])


# OCR 处理函数
def ocr(image_bytes):
    # 将字节数据转化为PIL图像
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # 图像预处理
    imgs = data_transform(image)

    # 模型推理
    predict_outputs = m(imgs.unsqueeze(0))  # 加入批次维度
    predict_outputs = predict_outputs.view(-1, gen_ImageCaptcha.captcha_array.__len__())

    # 转换为可读标签
    predict_labels = one_hot.vectotext(predict_outputs)

    return predict_labels


# 路由处理
@app.route("/runtime/text/invoke", methods=["POST"])
def invoke_ocr():
    # 从请求中提取 JSON
    req_data = request.get_json()

    # 提取项目名和图像（Base64 编码）
    project_name = req_data.get("project_name", "")
    image_base64 = req_data.get("image", "")

    if project_name != "ctc_en5l_240516":
        return jsonify({"error": "Invalid project name"}), 400

    # 解码 Base64 图像
    image_bytes = base64.b64decode(image_base64)

    # 运行 OCR 识别
    try:
        ocr_result = ocr(image_bytes)
        return jsonify({"data": ocr_result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 启动 Flask 应用
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=19199)
