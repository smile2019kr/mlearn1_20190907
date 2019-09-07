class Mammal:
    def __init__(self):
        pass
    def execute(self):
        import tensorflow as tf
        import numpy as np
        # [털, 날개] -> 기타, 포유류, 조류
        # 아직 사진으로는 판단어렵지만, feature(털, 날개)를 주고 어느 종인지(기타, 포유류, 조류) 맞추도록 하는 코딩
        """
            [[0,0]], -> [1, 0, 0] 기타. 털/날개 둘다 없으면 기타로 분류 
            [[1,0]], -> [0, 1, 0] 포유류. 털만 있음
            [[1,1]], -> [0, 0, 1] 조류. 털/날개 둘다 있음
            [[0,0]], -> [1, 0, 0] 기타
            [[0,0]], -> [1, 0, 0] 기타
            [[0,1]]  -> [0, 0, 1] 조류
        """
        x_data = np.array(
            [[0,0],
            [1,0],
            [1,1],
            [0,0],
            [0,0],
            [0,1]]
        )

        y_data = np.array(
            [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
             ]
        )
        X = tf.placeholder(tf.float32) # 대문자X,Y는 확률변수값. 외부에서 주어지는 것이므로 placeholder로 처리
        Y = tf.placeholder(tf.float32)
        W = tf.Variable(tf.random_uniform([2,3], -1, 1.))
        # 가중치는 내가 부여하는 것이 아니라 내부에서 결정되는 것이므로 Variable로 처리
        # random_uniform : 정규분포에 맞는 값을 랜덤으로 뽑으라는 뜻
        # -1 은 all
        # 신경망 neural network 앞으로는 nn으로 표기
        # nn은 2차원으로 [입력층(특성), 출력층(레이블)] -> [2,3] 으로 정합니다
        b = tf.Variable(tf.zeros([3]))  # tf.zeros : 모든값을 0으로 초기화
        # b 는 편향 bias
        # W 는 가중치 weight
        # b 는 각 레이어의 아웃풋 갯수로 설정함
        # b 는 최종 결과값의 분류갯수인 3 (기타, 포유류, 조류) 으로 설정함
        L = tf.add(tf.matmul(X, W), b) # Y = WX + b
        # 가중치와 편향을 이용해 계산한 결과값에
        L = tf.nn.relu(L) # 다층신경망을 추가. 위에서 설정된 L값에서 신경망을 이룬 L 의 형태로 변화함.
        # 교재 49p MCP 뉴런모델
        model = tf.nn.softmax(L)
        """
        softmax 소프트맥스 함수는 다음 처럼 결과값을 전체 합이 1인 확률로 만들어주는 함수
        예) [8.04, 2.76, -6.52] -> [0.52, 0.24, 0.23] (scaling)
        """
        print('--------- 모델 내부 보기 -----------')
        print(model) #결과 : Tensor("Softmax:0", dtype=float32)
        cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(cost)
        # 비용함수를 최소화시키면 (=경사도를 0으로 만들면) 그 값이 최적화된 값
        # 경사도가 0이되는 순간 = 엔트로피가 0이 되는 순간
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        for step in range(100): # 학습속도를 많이 주면 결과값이 늦게나옴
            sess.run(train_op, {X: x_data, Y: y_data})
            if(step+1) % 10 == 0: #10으로 나눠서 나머지가 0인 10자리 수 10개만 출력하라는 것
                print(step+1, sess.run(cost, {X: x_data, Y: y_data}))

        # 윗줄까지는 모델학습상태
        # 결과 확인(정확도값)
        prediction = tf.argmax(model, 1)
        target = tf.argmax(Y, 1)
        print('예측값: ', sess.run(prediction, {X: x_data}))
        print('실제값: ', sess.run(target, {Y: y_data}))
        # tf.argmax: 예측값과 실제값의 행렬에서 tf.argmax를 이용해 가장 큰 값을 가져옴
        # 예) [[0,1,1][1,0,0]] -> [1,0]
        # [[0.2, 0.7, 0.1][0.9, 0.1, 0. ]] -> [1,0]
        is_correct = tf.equal(prediction, target)
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print('정확도: %.2f' % sess.run(accuracy * 100, {X: x_data, Y: y_data}))
