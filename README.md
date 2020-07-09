# VAE-GAN
NTUT-CSIE Machine Learning 2020 Bonus Project. Implementing with tensorflow 2.2.0

# Project Introduction
利用MNIST資料集和10k US Adult Faces Database訓練VAE-GAN製作數字產生器以及人臉產生器．訓練人臉產生器使用不同的learning rate schedule觀察圖片產生的情況．

# Dataset Introduction
## MNIST
MNIST已經預先劃分資料用途，訓練集60000筆、測試集10000筆，總共10個類別．圖片為28×28的灰階圖片．  

## 10k US Adult Faces Database
![image from database official website](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/10kfacedatabase2.jpg)  
10k US Adult Faces Database總共有10000張圖片，圖片單邊固定256像素，另一邊為不固定大小但是一定小於256的RGB圖片．圖片以去除背景，背景為白色且多為正面人臉，把圖片長寬縮放至64×64．這次將訓練集共8500張圖片，驗證集共1500張圖片．

# Variational Autoencoder
![vae1](https://i.imgur.com/sKl5XPV.png)  
VAE的想法是：是否能用一組隨機的code產生一張真實的圖片．如圖所示，普通的Autoencoder已經能還原由圖片產生的code，從已知能還原圖片的兩組code中間取一點是否也能還原出有意義的圖片?  
  
實際結果是不可行的，較合理的解釋是編解碼使用DNN做非線性轉換，在code空間上，點與點的遷移沒有規律。且普通的Autoencoder進行訓練時，為了將reconstruct error壓低，因此很容易讓模型產生overfitting，這也是為什麼我們隨機給予一個code，卻很可能會生成出沒有意義的圖片。  
  
  將圖片編碼時加入一點噪聲使得單一的編碼點變成一個編碼點可能出現的連續區間．然後我們發現給編碼器加入一點噪聲可以有效覆蓋失真區域但是我們必須保證離原編碼附近的編碼概率最高，愈遠編碼概率愈低，因此圖像的編碼由原先離散的點變成一條連續的編碼分布曲線．
![vae2](https://imgur.com/uLklWnd.jpg)  

# KL Divergence
![formula0](https://wikimedia.org/api/rest_v1/media/math/render/svg/b115c150e9bbdbffb51b9f77d4d4e279b846e204)  
簡單來說這是一個衡量兩個分布相似度的方式，其值必大於等於零且不具對稱性．當且僅當P=Q，其值為零．
![formula1](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/formula1.PNG)  

# VAE-GAN Structure
![structure](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/vaegan.PNG)  
![structure](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/structure.PNG)  
![encoder_loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/encoder_loss.PNG)  
![decoder_loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/decoder_loss.PNG)  
![discrimintor_loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/discrimintor_loss.PNG)  

# Mode Collapse
當給generator的輸入z產生變化而generator輸出沒有改變，也就是generator只會產生同一種樣本，這時GAN產生Model Collapse．利用**Minibatch discrimination**可以有效解決Model Collapse．所謂Minibatch discrimination是指在給discriminator判斷樣本時，在樣本裡摻有其他樣本的信息，因為discriminator是一個個樣本判斷真偽，而Minibatch discrimination使discriminator組合查看樣本．[4]

# Some Training Tricks used in VAE-GAN-FACE
1. Flip labels when training generator: real = fake, fake = real
2. [BatchNorm](https://github.com/linxi159/GAN-training-tricks#4-batchnorm)
3. [Use Soft and Noisy Labels](https://github.com/linxi159/GAN-training-tricks#6-use-soft-and-noisy-labels)

# Experiment - Random Generation - MNIST
| Epoch 5 | Epoch 20 | Epoch 35 | Epoch 50 |
|:---------:|:---------:|:---------:|:---------:|
|![epoch5](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/mnist/image_at_epoch_004.png)|![epoch20](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/mnist/image_at_epoch_019.png)|![epoch35](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/mnist/image_at_epoch_034.png)|![epoch50](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/mnist/image_at_epoch_049.png)|  

| Epoch 65 | Epoch 80 | Epoch 95 | GIF |
|:---------:|:---------:|:---------:|:---------:|
|![epoch65](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/mnist/image_at_epoch_064.png)|![epoch80](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/mnist/image_at_epoch_079.png)|![epoch95](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/mnist/image_at_epoch_094.png)|![gif](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/mnist/number.gif)|

## Loss - MNIST
| Encoder Loss | Decoder Loss | Discriminator Loss |
|:---------:|:---------:|:---------:|
|![Encoder Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/mnist/encoder_loss_perform.PNG)|![Decoder Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/mnist/decoder_loss_perform.PNG)|![Discriminator Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/mnist/discriminator_loss_perform.PNG)|

# Experiment - Random Generation - 10k US Adult Faces Database

## Learning Rate Schdule 1
**Constant Learning Rate = 5e-4**

| Epoch 5 | Epoch 20 | Epoch 35 | Epoch 50 |
|:---------:|:---------:|:---------:|:---------:|
|![epoch5](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr1/face_at_epoch_004.png)|![epoch20](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr1/face_at_epoch_019.png)|![epoch35](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr1/face_at_epoch_034.png)|![epoch50](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr1/face_at_epoch_049.png)|  

| Epoch 65 | Epoch 80 | Epoch 95 | GIF |
|:---------:|:---------:|:---------:|:---------:|
|![epoch65](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr1/face_at_epoch_064.png)|![epoch80](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr1/face_at_epoch_079.png)|![epoch95](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr1/face_at_epoch_094.png)|![gif](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr1/vaegan_face.gif)|

## Loss - Learning Rate Schdule 1

| Encoder Loss | Decoder Loss | Discriminator Loss |
|:---------:|:---------:|:---------:|
|![Encoder Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr1/enc_loss.PNG)|![Decoder Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr1/dec_loss.PNG)|![Discriminator Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr1/disc_loss.PNG)|

## Learning Rate Schdule 2
**Initial Learning Rate = 5e-4，Decay Rate = 0.85，Minimum Learning Rate = 1e-4，patience = 10**

| Epoch 5 | Epoch 20 | Epoch 35 | Epoch 50 |
|:---------:|:---------:|:---------:|:---------:|
|![epoch5](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr2/face_at_epoch_004.png)|![epoch20](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr2/face_at_epoch_019.png)|![epoch35](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr2/face_at_epoch_034.png)|![epoch50](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr2/face_at_epoch_049.png)|  

| Epoch 65 | Epoch 80 | Epoch 95 | GIF |
|:---------:|:---------:|:---------:|:---------:|
|![epoch65](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr2/face_at_epoch_064.png)|![epoch80](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr2/face_at_epoch_079.png)|![epoch95](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr2/face_at_epoch_094.png)|![gif](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr2/vaegan_face.gif)|

## Loss - Learning Rate Schdule 2
| Encoder Loss | Decoder Loss | Discriminator Loss | Learning Rate |
|:---------:|:---------:|:---------:|:---------:|
|![Encoder Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr2/enc_loss.PNG)|![Decoder Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr2/dec_loss.PNG)|![Discriminator Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr2/disc_loss.PNG)|![Learning Rate](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr2/lr.png)|


## Learning Rate Schdule 3
**Initial Learning Rate = 5e-4，Decay Rate = 0.75，Minimum Learning Rate = 5e-6，patience = 6**

| Epoch 5 | Epoch 20 | Epoch 35 | Epoch 50 |
|:---------:|:---------:|:---------:|:---------:|
|![epoch5](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr3/face_at_epoch_004.png)|![epoch20](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr3/face_at_epoch_019.png)|![epoch35](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr3/face_at_epoch_034.png)|![epoch50](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr3/face_at_epoch_049.png)|  

| Epoch 65 | Epoch 80 | Epoch 95 | GIF |
|:---------:|:---------:|:---------:|:---------:|
|![epoch65](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr3/face_at_epoch_064.png)|![epoch80](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr3/face_at_epoch_079.png)|![epoch95](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr3/face_at_epoch_094.png)|![gif](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr3/vaegan_face.gif)|

## Loss - Learning Rate Schdule 3

| Encoder Loss | Decoder Loss | Discriminator Loss | Learning Rate |
|:---------:|:---------:|:---------:|:---------:|
|![Encoder Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr3/enc_loss.PNG)|![Decoder Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr3/dec_loss.PNG)|![Discriminator Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr3/disc_loss.PNG)|![Learning Rate](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr3/lr.PNG)|



## Learning Rate Schdule 4
**Initial Learning Rate = 5e-4，Decay Rate = 0.6，Minimum Learning Rate = 5e-6**，一開始等待6個epochs，若下一個Validation Loss與之前6個相比沒下降則衰減Learning Rate，衰減後不清空Buffer．

| Epoch 5 | Epoch 20 | Epoch 35 | Epoch 50 |
|:---------:|:---------:|:---------:|:---------:|
|![epoch5](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr4/face_at_epoch_004.png)|![epoch20](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr4/face_at_epoch_019.png)|![epoch35](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr4/face_at_epoch_034.png)|![epoch50](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr4/face_at_epoch_049.png)| 

| Epoch 65 | Epoch 80 | Epoch 95 | GIF |
|:---------:|:---------:|:---------:|:---------:|
|![epoch65](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr4/face_at_epoch_064.png)|![epoch80](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr4/face_at_epoch_079.png)|![epoch95](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr4/face_at_epoch_094.png)|![gif](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr4/vaegan_face.gif)|

## Loss - Learning Rate Schdule 4

| Encoder Loss | Decoder Loss | Discriminator Loss | Learning Rate |
|:---------:|:---------:|:---------:|:---------:|
|![Encoder Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr4/enc_loss.png)|![Decoder Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr4/dec_loss.png)|![Discriminator Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr4/disc_loss.png)|![Learning Rate](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/lr4/lr.png)

# Conclusion
從MNIST的實驗結果發現，訓練越久，某些數字出現的概率會大幅下降，甚至沒有出現，發生Model Collapse．然後發現BatchNormalization一定要擺在Activation Function之前，不然無法產生圖片．
從10k US Adult Faces Database的實驗結果發現使用更小的batch size加上Minibatch discrimination可以有效解決model collpase．如果一開始使用較小的Learning Rate訓練，人臉細節會變得模糊，以及必須避免Discriminator過擬合，Dropout layer的位置不方便調整．可以在[Generator](https://github.com/linxi159/GAN-training-tricks#13-add-noise-to-inputs-decay-over-time)每一層的輸入加入隨epoch遞減的[高斯雜訊](https://github.com/soumith/ganhacks/issues/26)．

# References
1. Auto-Encoding Variational Bayes: Diederik P Kingma, Max Welling (2013)
2. Autoencoding beyond pixels using a learned similarity metric: Anders Boesen Lindbo Larsen, Søren Kaae Sønderby, Hugo Larochelle, Ole Winther (2015)
3. Improved Techniques for Training GANs: Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen (2016)
4. http://kissg.me/2017/11/26/papernotes_47/#minibatch-discrimination
5. https://github.com/linxi159/GAN-training-tricks
