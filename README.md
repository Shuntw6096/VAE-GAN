# VAE-GAN
NTUT-CSIE Machine Learning 2020 Bonus Project. Using MNIST and implementing with tensorflow 2.2.0

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
![structure](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/structure.PNG)  
![encoder_loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/encoder_loss.PNG)  
![decoder_loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/decoder_loss.PNG)  
![discrimintor_loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/discrimintor_loss.PNG)  

# Experiment
## Random Generation
| Epoch 5 | Epoch 20 | Epoch 35 | Epoch 50 |
|:---------:|:---------:|:---------:|:---------:|
|![epoch5](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/image_at_epoch_004.png)|![epoch20](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/image_at_epoch_019.png)|![epoch35](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/image_at_epoch_034.png)|![epoch50](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/image_at_epoch_049.png)|  

| Epoch 65 | Epoch 80 | Epoch 95 | GIF |
|:---------:|:---------:|:---------:|:---------:|
|![epoch65](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/image_at_epoch_064.png)|![epoch80](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/image_at_epoch_079.png)|![epoch95](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/image_at_epoch_094.png)|![gif](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/vaegan.gif)|

## Loss
| Encoder Loss | Decoder Loss | Discriminator Loss |
|:---------:|:---------:|:---------:|
|![Encoder Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/encoder_loss_perform.PNG)|![Decoder Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/decoder_loss_perform.PNG)|![Discriminator Loss](https://github.com/Shuntw6096/VAE-GAN/blob/master/img/discriminator_loss_perform.PNG)|

# Conclusion
從實驗結果發現，訓練越久，某些數字出現的概率會大幅下降，甚至沒有出現，發生Model Collapse．然後發現BatchNormalization一定要擺在Activation Function之前，不然無法產生圖片．

# References
1. Auto-Encoding Variational Bayes: Diederik P Kingma, Max Welling (2013)
2. Autoencoding beyond pixels using a learned similarity metric: Anders Boesen Lindbo Larsen, Søren Kaae Sønderby, Hugo Larochelle, Ole Winther (2015)
3. Improved Techniques for Training GANs: Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen (2016)










