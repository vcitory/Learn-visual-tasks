import cv2
import onnx
import numpy as np
import onnxruntime as rt

def image_process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if (img.shape[0] != 224 or img.shape[1] != 224):
        img = cv2.resize(img, (224, 224))

    img = img.astype(np.float32) / 255.0
    # img = (img - 0.5) / 0.5
    # img[0,:,:] = (img[0,:,:] - 0.485) / 0.229
    # img[1,:,:] = (img[1,:,:] - 0.456) / 0.224
    # img[2,:,:] = (img[2,:,:] - 0.406) / 0.225

    img = img.transpose((2, 0, 1))
    image = img[np.newaxis, :, :, :]
    image = np.array(image, dtype=np.float32)

    return image

def onnx_runtime():
    sess = rt.InferenceSession(r"D:/test.onnx")
    input_name = sess.get_inputs()[0].name
    output = sess.get_outputs()[0].name

    img = cv2.imread(r"D:\006 Test\keypoint/224_keypoint_pfpld_pic_3.jpg")
    imgdata_1 = image_process(img)
    pred_onnx_1 = sess.run(None, {input_name: imgdata_1})[0][0]
    for i in range(5):
        img = cv2.circle(img, (int(pred_onnx_1[i * 2 + 0]), int(pred_onnx_1[i * 2 + 1])), 1, (255, 255, 0), -1)
    cv2.imshow("aaa", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    onnx_runtime()



