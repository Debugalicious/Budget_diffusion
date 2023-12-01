## Text2Image Generation On Google Coral EdgeTPU

Ahoy, my friends! Have you ever wanted to run a model similar to Stable Diffusion on a Google Coral? Well, keep searching because this isn't it. This is a basic text-to-image generation model running on the Google Coral. It's unstable, the quality is mediocre, but it's fun. How does it work? The model, a CNN, receives a simple text prompt. It then performs some magic on it, combines it with a random vector, and voilà, an image pops out. I've used a simple for-loop to re-feed the image into itself for refinement, but usually, after two iterations, it deteriorates. Initially, the model was intended to be a GAN, but I couldn't make it work, so I switched to a basic CNN to see if the Coral could generate any kind of image. This project is more of a starting point, aiming to train multiple models on individual segments of an image.

Inference output: 5it/s
![1701355262893](image/readme/1701355262893.png)

End goal diagram:

![1701354281258](image/readme/1701354281258.png)

Training Dataset: [m1guelpf/nouns · Datasets at Hugging Face](https://huggingface.co/datasets/m1guelpf/nouns)

Dunno if it will work, but so far it has been very promising. I've been able to generate some pretty cool images in a reasonable amount of time. Anyways all content is under the MIT license so feel free to use it for whatever you want. I've included a requirements.txt file so you can install all the dependencies. Any advice or contributions are welcome.

***More indepth readme is comming soon***
