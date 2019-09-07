class RamPrice:
    def __init__(self):
        pass

    # 혼자 공부하기 위함이므로 def 안에 import 를 넣어도 괜찮음. import를 맨 위에 쓰는것은 협업 시의 약속
    def execute(self):
        import pandas as pd
        import os
        import matplotlib.pyplot as plt
        import mglearn
        import numpy as np

        ram_price = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))
        # mglearn에서 제공되는 예제를 활용. 다운로드 필요없이 온라인상에서 바로 제공됨
        plt.semilogy(ram_price.date, ram_price.price)
        plt.xlabel("년")
        plt.ylabel("가격")
        # plt.show()
        # 위와같이 현재 데이터로 결정모델 생성 후, 아래와 같이 예측

        from sklearn.tree import DecisionTreeRegressor
        from sklearn.linear_model import LinearRegression
        data_train = ram_price[ram_price['date'] < 2000]  # 2000년 이전의 데이터로 학습
        data_test = ram_price[ram_price['date'] >= 2000]  # 학습결과를 바탕으로 2000년 이후의 값을 예측. 지도학습
        x_train = data_train['date'][:, np.newaxis]  # train data 를 1열로 만든다. date만 1열로 뽑아낸다는 것
        y_train = np.log(data_train['price'])
        tree = DecisionTreeRegressor().fit(x_train, y_train)
        lr = LinearRegression().fit(x_train, y_train)
        # 알고리즘을 클래스로 만들어놓은것 DecisionTreeRegressor() -> 괄호로 생성자 표기
        # 이미 규정되어있는 선택지들(알고리즘)의 형태를 바꿀수는 없고 데이터를 적용(fit)하는 것만 가능

        # test는 모든 데이터에 대해 적용
        x_all = ram_price['date'].values.reshape(-1, 1)  # x_all 을 1열로 만든다
        pred_tree = tree.predict(x_all)
        price_tree = np.exp(pred_tree)  # log값 되돌리기
        pred_lr = lr.predict(x_all)
        price_lr = np.exp(pred_lr)  # log값 되돌리기

        # 예측결과를 차트로 표현
        plt.semilogy(ram_price['date'], pred_tree, label="TREE PREDIC", ls='-', dashes=(2, 1))
        plt.semilogy(ram_price['date'], pred_lr, label="LINEAR REGRESSION PREDIC", ls=':', dashes=(2, 1))
        plt.semilogy(data_train['date'], data_train['price'], label='TRAINING DATA', alpha=0.4)
        plt.semilogy(data_test['date'], data_test['price'], label='TEST DATA')
        plt.legend(loc=1) # 범례 설정
        plt.xlabel('year', size=15)
        plt.ylabel('price', size=15)
        plt.show()
