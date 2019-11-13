import cv2
import numpy as np
import tensorflow as tf

from captcha_gen import CAPTCHA_LIST




def freeze_graph_test(pb_path, image_path):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 定义输入的张量名称,对应网络结构的输入张量
            # Placeholder:0作为输入图像,Placeholder_2:0作为dropout的参数,测试时值为1
            input_image_tensor = sess.graph.get_tensor_by_name("Placeholder:0")
            input_olaceholder2 = sess.graph.get_tensor_by_name("Placeholder_2:0")

            # 定义输出的张量名称
            output_tensor_name = sess.graph.get_tensor_by_name("output:0")

            # 读取测试图片
            im = read_image(image_path)
            out = sess.run(output_tensor_name, feed_dict={input_image_tensor: im, input_olaceholder2: 1.0})

            vec_idx = out
            text_list = [CAPTCHA_LIST[int(v)] for v in vec_idx[0]]
            return ''.join(text_list)


def read_image(imgpath):
    batch_x = np.zeros([1, 160 * 60])
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (160, 60))
    batch_x[0, :] = img.flatten() / 255.0
    return batch_x


res = freeze_graph_test("model/model.pb", "1563077280950.png")
print(res)
