class NumberChecker:
    def __init__(self):
        pass

    def execute(self):
        import numpy as np #답을 숫자화해야함
        import matplotlib.pyplot as plt #답을 도식화해야함
        from tensorflow import keras # 이미 만들어져있는 기능을 묶어놓은 것. 라이브러리
        # tf는 프레임워크. keras는 라이브러리. 프레임워크를 묶으면 라이브러리가 됨
        # keras(라이브러리)를 사용하지 않는다면 tf를 사용해서 원하는 방식대로 코딩해야 함
        mnist = keras.datasets.mnist
        # mailchecker가 더 난이도 낮은 코딩이지만(이미지보다는 텍스트 수준이므로) keras를 사용하므로 코딩이 짧아짐
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        # 데이터를 쪼개는 구문

#        print("훈련 이미지: ", train_images.shape)
#        print("훈련 라벨: ", train_labels.shape)
#        print("테스트 이미지: ", test_images.shape)
#        print("테스트 라벨: ", test_labels.shape)
#        print('\n')

        plt.figure(figsize=(5,5))
        image = train_images[100]
        plt.imshow(image)
        plt.show()







