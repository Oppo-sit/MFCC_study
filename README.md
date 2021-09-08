# MFCC_study
Study MFCC

patch-note
9/8 이전
오디오 폴더로부터 오디오 파일들을 불러와서 information이 담겨있는 csv 파일을 참조, 감정에 따라 오디오의 mfcc를 분류.

9/8
mfcc의 multithreading 구현, 그 외에 cnt와 같은 불필요한 변수의 제거로 코드가 경량화되었음. 훨씬 더 빠른 속도로 처리 가능
ipynb -> py 폴더로 변환

개선점 
코드 경량화(공부 필요), 분산 컴퓨팅, 더 좋은 성능 구현을 위한 방법 강구, 데이터 분석 및 예측을 위해 좋은 classfier를 찾고, MFCC 데이터를 어떻게 활용해야 할 지 생각해본다.
