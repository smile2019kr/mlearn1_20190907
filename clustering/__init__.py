import mglearn # 학습용데이터를 제공하는 라이브러리
import matplotlib.pyplot as plt
mglearn.plots.plot_knn_regression(n_neighbors=1)
#이웃한 값이 1일때 알고리즘
#선형으로 분포한 자료 -> linear reg. 그룹식으로 분포한 자료 -> 군집. 중에서 최근접이웃분류 알고리즘이 knn_reg.
plt.show()
