
import os
import matplotlib.pyplot as plt
import time
import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from super_token import Tokenizer, tokenizer_from_json
import json


def pad_sequences(sequences, maxlen, padding='post', truncating='post'):
    padded_sequences = []
    for seq in sequences:
        if len(seq) >= maxlen:
            if truncating == 'pre':
                padded_seq = seq[-maxlen:]
            else:
                padded_seq = seq[:maxlen]
        else:
            if padding == 'post':
                padded_seq = seq + [0] * (maxlen - len(seq))
            else:
                padded_seq = [0] * (maxlen - len(seq)) + seq
        padded_sequences.append(padded_seq)
    return np.array(padded_sequences)


def texts_to_padded_sequences(texts, max_length, num_words):

    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(
        sequences, maxlen=max_length, padding="post", truncating="post"
    )

    return padded_sequences, tokenizer


runs_dir = "runs/" + str(int(round(time.time() * 1000))) + "/"
if not os.path.exists(runs_dir):
    os.makedirs(runs_dir)

prompt = "light green glasses, a bar chart-shaped head and a grayscale-colored body on a warm background"
#prompt = "a pixel art character with square dark green glasses, a rainbow-shaped head and a orange-colored body on a cool background"

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)


sequences = tokenizer.texts_to_sequences([prompt])


interpreter = make_interpreter("budget_diffusion_edgetpu.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)


arrLen = len(input_details[0]['shape'])
flipped = False if arrLen == 2 else True
dim = input_details[0]['shape'][1] if flipped else input_details[1]['shape'][1]
max_length = input_details[1]['shape'][1] if flipped else input_details[0]['shape'][1]

padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
input_text = np.array(padded_sequences).astype(np.uint8)


imageits = []
iterations = 10
input_image = np.random.rand(1, dim, dim, 3).astype(np.uint8)
for i in range(iterations):

    print("Progress: {}/{}".format(i, iterations), end="\r")

    interpreter.set_tensor(
        input_details[0 if flipped else 1]["index"], input_image)
    interpreter.set_tensor(
        input_details[1 if flipped else 0]["index"], input_text)
    interpreter.invoke()
    input_image = interpreter.get_tensor(output_details[0]["index"])
    imageits.append(input_image)

    plt.imshow(input_image[0])
    plt.axis('off')
    plt.savefig(runs_dir + str(i) + '.jpeg')


grid_size = int(np.ceil(np.sqrt(iterations)))
fig, axs = plt.subplots(2, 5)
plt.axis('off')
plt.tight_layout()

fig.subplots_adjust(hspace=0)
fig.subplots_adjust(wspace=0)


for i in range(10):
    x, y = divmod(i, 5)

    # I dunno why but i scale the images back to 0-255 it makes everything go poo poo
    #output_image = np.clip(imageits[i][0] * 255, 0, 255).astype(np.uint8)
    output_image = imageits[i][0]
    axs[x, y].imshow(output_image)
    axs[x, y].axis('off')


for i in range(iterations - 5, iterations):
    x, y = divmod(i - (iterations - 5), 5)
    # Same issue as above
    #output_image = np.clip(imageits[i][0] * 255, 0, 255).astype(np.uint8)
    output_image = imageits[i][0]

    axs[x + 1, y].imshow(output_image)
    axs[x + 1, y].axis('off')

pltname = str(int(round(time.time() * 1000)))
plt.savefig(runs_dir + pltname + 'b.png')
plt.show()
