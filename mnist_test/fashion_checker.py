import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


class FashionChecker:
    def __init__(self):
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def create_model(self) -> []:
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        # 전처리. data processing.
        # 데이터를 불러와서 쪼개는 구문. 개발자는 데이터를 정제해서 쪼개넣으면 이하의 내용은 keras에서 층을 쌓고 학습도 몇줄로 끝남

        # 이미지를 확인하기 위한 구문 이하 5행
        #plt.figure()
        #plt.imshow(train_images[10])
        #plt.colorbar()
        #plt.grid(False) # grid표시없이 이미지만 보임
        #plt.show()

        # train_images = train_images / 255.0   255개의 전체 그림들을 확인
        # test_images = test_images / 255.0

        # modeling : flatten -> relu -> softmax 순으로 층을 쌓음
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'), # 활성화함수 relu
            keras.layers.Dense(10, activation='softmax')
        ])
        # 층층이 쌓은 모델을 compile로 통합
        model.compile(optimizer='adam',
                      loss = 'sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        # 학습
        model.fit(train_images, train_labels, epochs=5)

        # test
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print('테스트 정확도: {}'.format(test_acc))
        # modeling -> learning -> test 가 몇줄만에 끝나므로 개발자는 전처리 과정에 쓰는시간이 큼

