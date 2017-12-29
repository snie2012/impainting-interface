import sys
sys.path.append('model/')

from options.test_options import TestOptions
from models.models import create_model
from data.base_dataset import get_transform
import torch
import numpy as np
from PIL import Image

# --dataroot /home/snie/Desktop/impaiting_interface/static/ --name celeba1024_progressive_v2 --model pix2pix --which_model_netG progressive --which_direction AtoB --how_many 40 --class_index_B 52 --batchSize 1 --norm batch --conditionalCAM --which_epoch latest --which_resl 8 --end_resl 8 --gpu_ids -1

opt = TestOptions().parse()

opt.nThreads = 1   # test code only supports nThreads = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

transform = get_transform(opt)
model = create_model(opt)

# Flask code
from flask import Flask, render_template, request
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/api/input', methods=['POST'])
def serve_input():
    print ('Input received')
    response = request.json
    # mask = response['mask']
    image = response['image']
    image = np.asarray(image).reshape(1024, 1024, 3)
    # image = torch.FloatTensor(image)
    image = Image.fromarray(np.uint8(image), mode='RGB')
    image = torch.FloatTensor(np.expand_dims(transform(image), 0))

    data = {
        'A': image,
        'B': image,
        'A_paths': 'results/',
        'B_paths': 'results/'
    }

    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()

    # fake_image = visuals['fake_B'].numpy()
    # fake_image = np.pad(fake_image, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=255)

    return json.dumps({
        'fake': visuals['fake_B'].tolist()
    })

if __name__ == '__main__':
    app.run(port=9001, debug=True)
