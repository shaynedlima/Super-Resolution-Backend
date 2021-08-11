# Super Resolution Web Application
## Background
### [Demo](http://super-res.herokuapp.com/)

For my Engineering Final Year Project, I worked alongside Dr. Mehrtash Harandi to develop a web application that allows users to get a closer look into state-of-the-art research in the field of computer vision and machine learning when it comes to enhancing images.

This web application focusses on the concept of Super Resolution. Super Resolution involves the process of training a Neural Network to increase the resolution of images by inputting a lower resolution image into the network and training the network to output a higher resolution version of that image.

The main source of inspiration for this project came from ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"](https://arxiv.org/pdf/1609.04802.pdf) (Ledig et al., 2017). This paper investigates the use of Generative Adversarial Networks (GANs) to increase the resolution of images, with impressive results.

An implementation of this research paper was made publicly available on Github by [sgrvinod](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution). This served as a great starting point for this application; adapting and building on this code allowed me to train this network on various datasets and explore other ML concepts such as image inpainting.

## High-level overview
### Simplified model

![Simplified model](/assets/GAN_training.png?raw=true "Title")

### User and data flow diagram

![User and data flow diagram](/assets/user_flow.png?raw=true "Title")

## Training Procedure
Due to the adversarial nature of GANs, training GANs is a difficult and time-consuming process...

![Training process](/assets/training_vid.gif?raw=true "Title")

## Image Inpainting
We also investigated simultaneously applying super resolution and image inpainting.

![Image inpainting](/assets/GAN_inpainting.png?raw=true "Title")


## Video Super Resolution
We also looked into how we could use these techniques to enhance the quality of videos, we took the following approach to improve videos:

![Video super resolution](/assets/video_super_res.png?raw=true "Title")



https://user-images.githubusercontent.com/21117482/129003578-b9427136-36a8-48b6-b280-5843f9d2f393.mp4



## Further Work
- Mobile App with on-device inference
- Improve video enhancement by making use of the information between frames, rather than viewing frames individually
- Further model training
