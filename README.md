# MLP, CNN, RNN và LSTM trong TensorFlow với bộ thư viện MNIST
Bài tập xây dựng MLP, CNN, RNN và LSTM trong TensorFlow và vận dụng để giải quyết bài toán nhận dạng chữ số viết tay với bộ dữ liệu MNIST.
## Cách chạy
```
python [tên_file.py]
```
Ví dụ:
```
python mlp.py
```
## Kết quả chạy
### MLP (Multilayer Perceptron)
```
Step: 0300 cost=272.817352295 acc= 0.84375
Step: 0310 cost=187.351882935 acc= 0.8984375
Step: 0320 cost=174.795898438 acc= 0.890625
Step: 0330 cost=206.264328003 acc= 0.8984375
Step: 0340 cost=265.206970215 acc= 0.875
Step: 0350 cost=101.322456360 acc= 0.9296875
Step: 0360 cost=152.056304932 acc= 0.890625
Step: 0370 cost=161.643920898 acc= 0.9140625
Step: 0380 cost=90.423835754 acc= 0.9140625
Step: 0390 cost=201.160034180 acc= 0.8984375
Step: 0400 cost=189.508758545 acc= 0.9140625
Da toi uu xong ham mat mat!
Do chinh xac:  0.9099
```
### CNN (Convolutional Neural Network)
```
Step: 0300 cost=627.192199707 acc= 0.9453125
Step: 0310 cost=874.478515625 acc= 0.9453125
Step: 0320 cost=758.593872070 acc= 0.9140625
Step: 0330 cost=1048.539062500 acc= 0.9296875
Step: 0340 cost=596.195495605 acc= 0.9140625
Step: 0350 cost=917.772460938 acc= 0.9140625
Step: 0360 cost=1159.645751953 acc= 0.9140625
Step: 0370 cost=820.761779785 acc= 0.921875
Step: 0380 cost=881.864624023 acc= 0.921875
Step: 0390 cost=673.660888672 acc= 0.9296875
Step: 0400 cost=558.677246094 acc= 0.9453125
Da toi uu xong ham mat mat!
Do chinh xac:  0.953125
```
### RNN (Recurrrent Neural Network sử dụng basic units)
```
Step: 300, cost= 0.2470, acc= 0.906
Step: 310, cost= 0.1922, acc= 0.930
Step: 320, cost= 0.1903, acc= 0.930
Step: 330, cost= 0.2309, acc= 0.914
Step: 340, cost= 0.1545, acc= 0.953
Step: 350, cost= 0.2177, acc= 0.930
Step: 360, cost= 0.2327, acc= 0.914
Step: 370, cost= 0.2556, acc= 0.938
Step: 380, cost= 0.1545, acc= 0.938
Step: 390, cost= 0.2907, acc= 0.930
Step: 400, cost= 0.2922, acc= 0.898
Da toi uu xong ham mat mat!
Do chinh xac:  0.9453125
```
## LSTM (RNN sử dụng Long Short Term Memory Cell units)
```
Step: 300, cost= 0.2927, acc= 0.906
Step: 310, cost= 0.1611, acc= 0.945
Step: 320, cost= 0.2453, acc= 0.938
Step: 330, cost= 0.1562, acc= 0.961
Step: 340, cost= 0.1453, acc= 0.938
Step: 350, cost= 0.1037, acc= 0.977
Step: 360, cost= 0.0832, acc= 0.977
Step: 370, cost= 0.1000, acc= 0.961
Step: 380, cost= 0.0884, acc= 0.961
Step: 390, cost= 0.1858, acc= 0.953
Step: 400, cost= 0.1465, acc= 0.938
Da toi uu xong ham mat mat!
Do chinh xac: 0.98046875
```