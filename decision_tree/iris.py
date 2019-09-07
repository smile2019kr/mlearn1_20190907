class Iris:
    def __init__(self):
        pass

    def execute(self):
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd
        import numpy as np

        np.random.seed(0) # 랜덤값을 고정시키는 로직
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names) # feature (변수) 를 가지고 데이터프레임을 만든다는 것
        print(df.head()) # 출력해서 어떤 구조인지 파악해야 함
        print(df.columns)
        """
        출력결과 바탕으로 인덱스값 확인
        Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)'],
        """

        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        # print(df.columns)
        df['is_train'] = np.random.uniform(0,1, len(df)) <= .75  # train set 75%
        #print(df.columns)

        train, test = df[df['is_train']==True], df[df['is_train']==False]
        features = df.columns[:4] #앞에서부터 4번째 컬럼(피쳐) 까지 추출. 0,1,2,3
#        print(features)
        y = pd.factorize(train['species'])[0]
        #print(y)

        # 학습 learning
        clf = RandomForestClassifier(n_jobs=2, random_state=0) # n_ : the number of ~~. ~~의 숫자를 확인
        clf.fit(train[features], y) # 4가지 데이터를 뽑아서 y에 fit시킴
        #print(clf.predict_proba(test[features])[0:10])

        # 정확도 평가
        preds = iris.target_names[clf.predict(test[features])]
        print(preds[0:5]) # 품종에 해당하는 명칭을 5개만 출력

        # 크로스탭
        i = pd.crosstab(test['species'], preds, rownames=['Actual Species'],
                    colnames=['Predicted Species'])

        # print(i)
        # 피쳐별 중요도 출력. regression에서 W에 해당
        # iris에서는 이미 주어져있는 가중치이므로 출력만 하지만 타이타닉데이터에서는 별도로 가중치 부여해야함
        print(list(zip(train[features], clf.feature_importances_)))
"""
[('꽃받침 sepal length (cm)', 0.11185992930506346), ('꽃받침 sepal width (cm)', 0.016341813006098178), 
('꽃잎 petal length (cm)', 0.36439533040889194), ('꽃잎 petal width (cm)', 0.5074029272799464)]
"""

