### Describe the goal of the project:

1. a user choose an image in the front end
2. allow a user to draw arbitrary mask on the image
3. send the image and mask to the backend
4. the backend takes the image and mask as input and output a completed image
5. transmit the image from the backend to the frontend
6. display the completed image on the frontend

major tasks:
* the ability to choose image
* the ability to draw mask
* send image and mask to backend
* capture the completed image and send back to the front end
* display the image on the front end


* Design the layout of the homepage

### Running command
python server.py --dataroot /home/snie/Desktop/impaiting_interface/static/ --name celeba1024_progressive_v2 --model pix2pix --which_model_netG progressive --which_direction AtoB --how_many 40 --class_index_B 52 --batchSize 1 --norm batch --conditionalCAM --which_epoch latest --which_resl 8 --end_resl 8 --gpu_ids 0

