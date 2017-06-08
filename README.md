# Crystal-Gems
A dataset of mineral images and labels for machine learning purposes. The dataset is created from this website http://www.minerals.net/ . The dataset consists of around 4,000 images of mineral specimines with labels. There are numouros labels included such as mineral name, crystal stystem, chemical groups, rock types and fracture.  A complete list of labels is found in `mineral.py`. To generate this dataset run `mineral_net.py` and this will scrape all the images and mineral labels as well as generate tfrecords for them. The images are saved in `data` and the tfrecords are saved in `tfrecords`. When generating records there is a test train split of 80 percent. Here are a few images for show.

![alt tag](https://github.com/loliverhennigh/Crystal-Gems/blob/master/data/aquamarine/aquamarine-terminated-pyramidal-brazil-thb.jpg)
![alt tag](https://github.com/loliverhennigh/Crystal-Gems/blob/master/data/clinochlore/clinochlore-kammererite-woods-chrome-thb.jpg)
![alt tag](https://github.com/loliverhennigh/Crystal-Gems/blob/master/data/inesite/inesite-botryoidal-wessels-s-africa-t.jpg)
![alt tag](https://github.com/loliverhennigh/Crystal-Gems/blob/master/data/vivianite/vivianite-tomokoni-potosi-bolivia-t.jpg)


# Training Neural Network
I have also included a few scripts to train and test neural networks in predicting labels from images. This proved an extremely difficult task due to the small dataset size. I made a few attempts to overcome this by using a network pretrained on cifar however this did not help enough. In my opinion this problem is definetly solvable but it will require a larger training set and better transfer learning techniques. I believe that using a well proven network such as Inception 3 or 4 thats trained on imagenet would be a good step at getting transfer learning working. I also think that the dataset could be expanded and refined heavily by scraping images from ebay or other cites that sell minerals. Right now I dont really have the time to tackel these problems though.
