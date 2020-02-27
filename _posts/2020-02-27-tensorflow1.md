---
title: Tensorflow 실습 1
date: 2020-02-27 13:34:00 +0900
categories: [Blogging, Programming]
tags: [ai, ml, tensorflow]
toc: true
sitemap:
  changefreq: daily
  priority: 1.0
seo:
  date_modified: 2020-02-27 13:34:00 +0900
---
tensorflow 실습 첫 번째

***

## **Cost function 그래프 그리기**

&nbsp;&nbsp;&nbsp;&nbsp;Google colab을 이용해, 이전에 작성했던 [linear regression 관련 글](https://now-man.github.io/posts/ai2/)에서 살펴봤던 cost function 관련 코딩을 해봤습니다.<br>
&nbsp;&nbsp;&nbsp;&nbsp;(1,1), (2,2), (3,3) 총 3개의 값이 있다면, 쉽게 x=y 형태의 linear regression임을 알 수 있을 것입니다. $$h_{θ}(x) = θ_{0}+θ_{1}x$$에서
$$θ_{1}$$ 값은 1, $$θ_{0}x$$ 값은 0이 됩니다. 가로축이 $$θ_{1}$$이고, 세로축이 cost function인 그래프가 있다면 $$θ_{1}$$의 최솟값이 1이되는 형태의 그래프가 나올 것입니다.


&nbsp;&nbsp;&nbsp;&nbsp;코드는 아래와 같습니다. 저는 google colab을 사용해서 plot을 바로 다운받는 형태로 작성했습니다.

```python
import tensorflow as tf
import matplotlib.pyplot as plt
from google.colab import files
X = [1,2,3]
Y = [1,2,3]

W = tf.placeholder(tf.float32)
#tf.placeholder는 데이터가 저장되는 공간, float32는 변수의 데이터 타입

hypothesis = X*W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()
#Session 객체는 중간 결과를 저장, 최종 결과를 작업 환경으로 전송하는 역할

sess.run(tf.global_variables_initializer())
#Session 객체 내의 run()을 통해 그래프 실행, 변수 초기화

W_val = []
cost_val = []
#그래프의 x축, y축 생성

for i in range(-30, 50):
  feed_W = i*0.1 #feed_W가 0.1 rksrurdmfh -3에서 5까지 움직입니다.
  curr_cost, curr_W = sess.run([cost, W], feed_dict = {W: feed_W}) 
  W_val.append(curr_W)
  cost_val.append(curr_cost)

plt.plot(W_val, cost_val)
plt.xlabel("$θ_1$")
plt.ylabel('cost')
plt.savefig('test.png', dpi = 300)
files.download('test.png')
```
![tansex1](/images/posts/2020-02-27-tensorflow1/tensex1.png){:width="65%"}{: .center}
*<center>위와 같이 최솟값이 1이되는 그래프가 잘 나오는 것을 확인했습니다!</center>*


<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
  });
</script>
<script src="//cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>