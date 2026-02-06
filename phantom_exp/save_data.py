import argparse
import importlib
import wandb
import torch
import numpy as np
from PIL import Image
import json

from utils import get_fourier_sampling_mask, get_coords, get_inr, plot_inr_data

def main():
    #set up config
    parser = argparse.ArgumentParser()
    #config for data set up
    parser.add_argument("--datapath", type=str, help = "path to wandb artifact containing presaved data")
    parser.add_argument("--nx", type=int, default=1024, help = "recon grid size")
    parser.add_argument("--K", type=int, default=64, help = "sampling frequency cutoff")
    parser.add_argument('--device', required=False, help="Device to use: 'cuda:0', 'cuda:1', etc, or 'cpu'. Defaults to GPU 0 if available.")
    args = parser.parse_args()

    settings = vars(args)

    # Set the device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    with wandb.init(project="middle_linear", config=settings, group="super INRs", save_code=True, job_type = "load-data") as run:
        #create artifact folder for storing dataset
        dataset_artifact = wandb.Artifact(
            f"{args.datapath}nx{args.nx}K{args.K}", type="dataset",
            description=f"data for super resolution.",
            metadata=settings
        )

        #load data
        phantom_name = args.datapath
        path = '/home/sueparkinson/deeprelu/super_inrs/phantom_exp/'
        x0 = np.load(path+"data/"+phantom_name+"_lowpass_1024.npy") #ideal low-pass version of phantom computed from *exact* fourier coefficients
        x00 = np.load(path+"data/"+phantom_name+"_rasterized_1024.npy") #hi-res rasterized phantom for reference
        x0 = torch.from_numpy(x0).float().to(device)
        x00 = torch.from_numpy(x00).float().to(device)

        # define fourier sampling mask
        K = args.K #sampling frequency cutoff
        nx = args.nx #recon grid size
        res = (nx,nx) #image resolution over which to perform FFTs
        mask = get_fourier_sampling_mask(nx,K)

        # get measurement vector y
        y = torch.fft.fft2(x0,norm="ortho")[mask]  #low-pass Fourier coefficients

        #save measurement vector y
        with dataset_artifact.new_file(f"measurements_y.npy",mode="wb") as file:
            np.save(file, y.cpu().numpy()) 

        # define MSE metric
        MSE = torch.nn.MSELoss()

        # compute zero-filled ifft recon as a baseline
        x1 = torch.zeros(res, dtype=torch.complex64, device=device)
        x1[mask] = y
        x1 = torch.real(torch.fft.ifft2(x1,norm="ortho"))
        initmse = MSE(x1,x00)
        wandb.log({"MSE of zero-filled IFFT": initmse.item()})

        #convert to numpy arrs
        x0 = x0.cpu().numpy()
        x00 = x00.cpu().numpy()
        x1 = x1.cpu().numpy()

        #save rasterized low-pass inr
        with dataset_artifact.new_file(f"rasterized_low_pass_image.npy",mode="wb") as file:
            np.save(file, x0) 

        #also save rasterized low-pass  image as .png
        # note: this is a lossy conversion
        img = (np.clip(x0,0,1) * 255).astype(np.uint8)
        im = Image.fromarray(img)
        with dataset_artifact.new_file(f"rasterized_low_pass_image.png",mode="wb") as file:
            im.save(file)     

        #save rasterized hi-res inr
        with dataset_artifact.new_file(f"rasterized_ground_truth_image.npy",mode="wb") as file:
            np.save(file, x00) 

        #also save rasterized hi-res image as .png
        # note: this is a lossy conversion
        img = (np.clip(x00,0,1) * 255).astype(np.uint8)
        im = Image.fromarray(img)
        with dataset_artifact.new_file(f"rasterized_ground_truth_image.png",mode="wb") as file:
            im.save(file)       

        #save rasterized ifft inr
        with dataset_artifact.new_file(f"rasterized_zero_filled_ifft_image.npy",mode="wb") as file:
            np.save(file, x1) 

        #also save rasterized ifft image as .png
        # note: this is a lossy conversion
        img = (np.clip(x1,0,1) * 255).astype(np.uint8)
        im = Image.fromarray(img)
        with dataset_artifact.new_file(f"rasterized_zero_filled_ifft_image.png",mode="wb") as file:
            im.save(file)       

        #push artifact to wandb
        print("logging dataset")
        run.log_artifact(dataset_artifact)        

        print(f"\nDone.")

if __name__ == "__main__":
    main()