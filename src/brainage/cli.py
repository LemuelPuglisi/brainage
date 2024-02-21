import os
import argparse

import torch
import pandas as pd
from monai.networks.nets import DenseNet
from monai import transforms
from tqdm import tqdm


def run_brainage():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', type=str, required=True, help="DenseNet checkpoints path")
    parser.add_argument('--inputs', type=str, required=True, help="CSV file indicating the inputs")
    parser.add_argument('--output', type=str, required=True, help="Path where to store the output")
    parser.add_argument('--gpu', action='store_true', help="Use GPU")
    args = parser.parse_args()

    assert os.path.exists(args.checkpoints), "Invalid checkpoints"
    assert os.path.exists(args.inputs), "Invalid input CSV"
    device = 'cuda' if args.gpu else 'cpu'

    transforms_fn =  transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']), 
        transforms.SpacingD(pixdim=1.4, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=(130, 130, 130), mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

    # Load the trained model
    net = DenseNet(3,1,1).to(device)
    net.load_state_dict(torch.load(args.checkpoints))
    net.eval()

    # Load the input CSV
    inputs_df = pd.read_csv(args.inputs)
    assert set(['id','image_path']).issubset(inputs_df.columns), 'Input CSV must have 2 columns: [id, image_path]'

    # run the inference
    data = []
    for input_dict in tqdm(inputs_df.to_dict(orient='records')):
        loaded_data = transforms_fn(input_dict)
        input_mri = loaded_data['image'].to(device).float().unsqueeze(0)
        pred_age = round(net(input_mri).item())
        data.append({ 'id': input_dict['id'], 'pred_age': pred_age })
    
    # save the outputs
    output_df = pd.DataFrame(data)
    output_df.to_csv(os.path.join(args.output, 'brainage-output.csv'), index=False)

