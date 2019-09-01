# Kaggle: Denoising Dirty Documents ([link](https://www.kaggle.com/c/denoising-dirty-documents/overview))

Data: 177 images of text documents 

Task: clean the images of noise and artifacts (eg. stains, spots, wrinkles)

Evalution: root mean square error between the cleaned vs actual pixel intensities (0...1)

Solution: simple 2D-CNN Autoencoder

Success: 0.087 RMSE

![](predictions.png)
