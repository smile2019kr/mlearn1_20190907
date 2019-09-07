class BreastCancer:
    def __init__(self):
        pass

    def execute(self):
        import sklearn
        from sklearn.tree import DecisionTreeClassifier
        import sklearn.datasets
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        # 모델을 코딩자가 선택하지 않고 train_test_split로 지도학습을 진행하라는 것.
        # 지도학습: label이 있는 자료. test데이터를 생성. -> 이후의 코딩이 지도학습 코딩을 하게 됨
        # 지도학습 / 비지도학습 / 강화학습 3가지 코딩패턴 -> 최근은 강화학습이 주흐름

        cancer = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            cancer.data, cancer.target, stratify=cancer.target, random_state=42)
        # 대문자 X 는 확률변수. 덩어리의 형태로 데이터가 들어감. 소문자 y는 데이터를 하나만 뽑는다는 것
        # randam_state=42 는 랜덤값을 일정한 값(여기서는 42)으로 고정해서 결과를 돌릴때마다 다른 값이 나오지 않도록 하기 위함
        tree = DecisionTreeClassifier(random_state=0) #값을 예측하는 것이 아니라 암 유/무 로 분류
        tree.fit(X_train, y_train)
        print('훈련세트의 정확도 :  {:.3f}'.format(tree.score(X_train, y_train)))
        print('테스트세트 정확도 :  {:.3f}'.format(tree.score(X_test, y_test)))
        """
        테스트세트의 정확도 0.937로 나옴 -> 현실데이터에 적용했을때는 낮아지는 경우가 발생. 현재 데이터에만 과최적화된 상태.
        독립변수의 개수가 많은 빅데이터에서는 과최적화(overfit)가 쉽게 발생
        => 그래서 앙상블을 사용하는 것
        """
