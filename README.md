# VAE-GAN
NTUT-CSIE Machine Learning 2020 Bonus Project. Implementing with tensorflow 2.2.0

# Project Introduction
利用MNIST資料集和10k US Adult Faces Database訓練VAE-GAN製作數字產生器以及人臉產生器．

# Dataset Introduction
## MNIST
MNIST已經預先劃分資料用途，訓練集60000筆、測試集10000筆，總共10個類別．圖片為28×28的灰階圖片．  

## 10k US Adult Faces Database
![image from database official website](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/10kfacedatabase2.jpg)  
10k US Adult Faces Database總共有10000張圖片，圖片單邊固定256像素，另一邊為不固定大小但是一定小於256．圖片以去除背景，背景為白色且多為正面人臉，
這次將訓練集共8500張圖片，驗證集共1500張圖片．

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

# Model Collapse
當給generator的輸入z產生變化而generator輸出沒有改變，也就是generator只會產生同一種樣本，這時GAN產生Model Collapse．利用**Minibatch discrimination**可以有效解決Model Collapse．所謂Minibatch discrimination是指在給discriminator判斷樣本時，在樣本裡摻有其他樣本的信息，因為discriminator是一個個樣本判斷真偽，而Minibatch discrimination使discriminator組合查看樣本．[4]

# Experiment
## Random Generation - MNIST
| Epoch 5 | Epoch 20 | Epoch 35 | Epoch 50 |
|:---------:|:---------:|:---------:|:---------:|
|![epoch5](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/image_at_epoch_004.png)|![epoch20](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/image_at_epoch_019.png)|![epoch35](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/image_at_epoch_034.png)|![epoch50](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/image_at_epoch_049.png)|  

| Epoch 65 | Epoch 80 | Epoch 95 | GIF |
|:---------:|:---------:|:---------:|:---------:|
|![epoch65](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/image_at_epoch_064.png)|![epoch80](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/image_at_epoch_079.png)|![epoch95](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/image_at_epoch_094.png)|![gif](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/vaegan.gif)|

## Loss - MNIST
| Encoder Loss | Decoder Loss | Discriminator Loss |
|:---------:|:---------:|:---------:|
|![Encoder Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/encoder_loss_perform.PNG)|![Decoder Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/decoder_loss_perform.PNG)|![Discriminator Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/discriminator_loss_perform.PNG)|

## Random Generation - 10k US Adult Faces Database
| Epoch 5 | Epoch 20 | Epoch 35 | Epoch 50 |
|:---------:|:---------:|:---------:|:---------:|
|![epoch5](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/face_at_epoch_004.png)|![epoch20](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/face_at_epoch_019.png)|![epoch35](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/face_at_epoch_034.png)|![epoch50](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/face_at_epoch_049.png)|  

| Epoch 65 | Epoch 80 | Epoch 95 | GIF |
|:---------:|:---------:|:---------:|:---------:|
|![epoch65](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/face_at_epoch_064.png)|![epoch80](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/face_at_epoch_079.png)|![epoch95](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/face_at_epoch_094.png)|![gif](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/vaegan_face.gif)|

## Loss - 10k US Adult Faces Database
| Encoder Loss | Decoder Loss | Discriminator Loss | Learning Rate |
|:---------:|:---------:|:---------:|:---------:|
|![Encoder Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/encoder_loss_perform_f.png)|![Decoder Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/decoder_loss_perform_f.png)|![Discriminator Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/discriminator_loss_perform_f.png)|![Learning Rate](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/learning_rate_f.png)

# Conclusion
從MNIST的實驗結果發現，訓練越久，某些數字出現的概率會大幅下降，甚至沒有出現，發生Model Collapse．然後發現BatchNormalization一定要擺在Activation Function之前，不然無法產生圖片．
從10k US Adult Faces Database的實驗結果發現使用更小的batch size加上Minibatch discrimination可以有效解決model collpase．

# References
1. Auto-Encoding Variational Bayes: Diederik P Kingma, Max Welling (2013)
2. Autoencoding beyond pixels using a learned similarity metric: Anders Boesen Lindbo Larsen, Søren Kaae Sønderby, Hugo Larochelle, Ole Winther (2015)
3. Improved Techniques for Training GANs: Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen (2016)
4. http://kissg.me/2017/11/26/papernotes_47/#minibatch-discrimination
