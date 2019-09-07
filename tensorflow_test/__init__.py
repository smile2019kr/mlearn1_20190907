from tensorflow_test.mammal import Mammal
from tensorflow_test.word_sequence import WordSequence
from tensorflow_test.naive_bayes import NaiveBayes
from tensorflow_test.web_crawler import WebCrawler

if __name__ == '__main__':

#    t = Mammal()
#    t.execute()  @staticmethod를 사용하지 않을 경우의 설정. 별도로 t를 만들어 줌
#     WordSequence.execute()
#     t = WebCrawler.create_model() #각 리뷰를 하나씩 긁어모으는 것을 여러번 반복하여 csv파일 생성
     nb = NaiveBayes()
     nb.train('./data/review_train.csv')
#     print(nb.classify('내 인생에서 쓰레기 같은 영화'))
          # 1이 100점인데 위의 문장을 넣고 분류했을때의 결과값 0.09916 -> 매우 부정적인 평이라고 분류함
#     print(nb.classify('내 인생에서 최고의 영화'))
          # 위의 문장을 넣고 분류했을때의 결과값 0.966113748742778 -> 매우 긍정적인 평이라고 분류함
#     print(nb.classify('별로'))
          # 위의 문장을 넣고 분류했을때의 결과값 0.19194883925453093
     print(nb.classify('그럭저럭 볼만함'))
          # 위의 문장을 넣고 분류했을때의 결과값 0.4308430115837011

