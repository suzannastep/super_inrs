import argparse
import importlib
import wandb
import torch
import numpy as np
from PIL import Image
import json

from utils import get_fourier_sampling_mask, get_coords, get_inr, plot_inr_data

def get_str(datapath,Lambda,layers,wd):
    return f"{datapath}Lam{Lambda}L{layers}wd{wd}"

def main():
    #set up config
    parser = argparse.ArgumentParser()
    #config for data set up
    parser.add_argument("--datapath", type=str, help = "path to wandb artifact containing presaved data")
    parser.add_argument("--nx", type=int, default=1024, help = "recon grid size")
    parser.add_argument("--K", type=int, default=64, help = "sampling frequency cutoff")
    #config for network
    parser.add_argument("--Lambda", type=int, help = "number of ReLU layers at start and end of network. Should be 0 for a deep ReLU network")
    parser.add_argument("--layers", type=int, help = "total number of neural network layers")
    parser.add_argument("--width", type=int, default=100, help = "hidden width")
    parser.add_argument("--K0", type=int, default=10, help = "fourier features grid size")
    parser.add_argument("--ff_freq", type=str, default="gridded", help = "what kind of fourier features to use")
    #config for training
    parser.add_argument("--seed", type=int, default=4, help = "seed for initialization")
    parser.add_argument("--wd", type=float, help = "regularization parameter")
    parser.add_argument("--epochs", type=int, default=50000, help = "number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-3, help = "learning rate")
    parser.add_argument("--step_size", type=int, default=40000, help = "how frequently to decay learning rate")
    parser.add_argument("--gamma", type=float, default=0.1, help = "how much to decay learning rate")
    parser.add_argument("--logfreq", type=int, default=100, help = "frequency of wandb logging. log every 'logfreq' epochs.")
    parser.add_argument('--device', required=False, help="Device to use: 'cuda:0', 'cuda:1', etc, or 'cpu'. Defaults to GPU 0 if available.")
    args = parser.parse_args()

    settings = vars(args)

    if args.layers == 2:
        settings["arch"] = "ffrelu_shallow"
    elif args.Lambda == 0:
        settings["arch"] = "ffrelu_deep"
    else:
        settings["arch"] = "ffmidlin_deep"

    settings["arch_options"] = {
        "Lambda":args.Lambda,
        "layers":args.layers,
        "width":args.width,
        "K0":args.K0,
        "ff_freq":args.ff_freq,
    }

    settings["coords_range"] = (0,1)

    # Set the device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    with wandb.init(project="middle_linear", config=settings, group="super INRs", save_code=True, job_type = "train") as run:
        #create artifact folder for storing model
        model_artifact = wandb.Artifact(
            get_str(args.datapath,args.Lambda,args.layers,args.wd), type="model",
            description=f"MLP for middle linear experiments.",
            metadata=settings
        )

        print("\nLoaded settings:")
        for k, v in settings.items():
            print(f"{k}: {v}")

        inr, x, metrics = run_experiment(settings,device)

        print(f"\nFinished! Saving outputs.")

        #log trained model
        with model_artifact.new_file("trained_model.pt",mode="wb") as file:
            torch.save(inr.state_dict(), file)

        #save rasterized inr output image (x)
        with model_artifact.new_file(f"rasterized_inr_output_image.npy",mode="wb") as file:
            np.save(file, x) 

        #also save rasterized output image as .png for easy visualization 
        # note: this is a lossy conversion
        img = (np.clip(x,0,1) * 255).astype(np.uint8)
        im = Image.fromarray(img)
        with model_artifact.new_file(f"rasterized_inr_output_image.png",mode="wb") as file:
            im.save(file)       

        #push artifact to wandb
        print("logging trained models")
        run.log_artifact(model_artifact)        

        print(f"\nDone.")

def run_experiment(settings,device):
    #load data
    phantom_name = settings["datapath"]
    path = '/home/sueparkinson/deeprelu/super_inrs/phantom_exp/'
    x0 = np.load(path+"data/"+phantom_name+"_lowpass_1024.npy") #ideal low-pass version of phantom computed from *exact* fourier coefficients
    x00 = np.load(path+"data/"+phantom_name+"_rasterized_1024.npy") #hi-res rasterized phantom for reference
    x0 = torch.from_numpy(x0).float().to(device)
    x00 = torch.from_numpy(x00).float().to(device)

    nchannels=1

    # define fourier sampling mask
    K = settings["K"] #sampling frequency cutoff
    nx = settings["nx"] #recon grid size
    res = (nx,nx) #image resolution over which to perform FFTs
    mask = get_fourier_sampling_mask(nx,K)

    # get measurement vector y
    y = torch.fft.fft2(x0,norm="ortho")[mask]  #low-pass Fourier coefficients

    # define MSE metric
    MSE = torch.nn.MSELoss()

    # compute zero-filled ifft recon as a baseline
    x1 = torch.zeros(res, dtype=torch.complex64, device=device)
    x1[mask] = y
    x1 = torch.real(torch.fft.ifft2(x1,norm="ortho"))
    initmse = MSE(x1,x00)
    wandb.log({"MSE of zero-filled IFFT": initmse.item()})

    # get INR coordinates
    coords_range = settings["coords_range"]
    coords = get_coords(nx, range = coords_range, dim=2)
    coords = coords.to(device)

    # problem-specific variables to pass to INR as needed
    vars = {"coords": coords, "res": res, "mask": mask}

    # define INR architecture
    torch.manual_seed(settings["seed"]) #set seed for reproducibility
    inr = get_inr(settings["arch"],settings["arch_options"])
    if "arch_init_wts" in settings: #apply custom initialization if specificed
        inr.load_state_dict(torch.load(path+f"wts/{settings['arch_init_wts']}", weights_only=True))
        inr.eval()
    inr.register(vars) #register extra vars if needed (mainly for ffrelu_shallow INR with modified WD-reg)
    inr = inr.to(device)

    print(f"architecture is \n{inr}")

    # define optimizer
    optimizer = torch.optim.Adam(inr.parameters(),lr=settings["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=settings["step_size"],gamma=settings["gamma"])

    lam = settings["wd"]
    epochs = settings["epochs"]

    # run training loop
    for iter in range(epochs):
        optimizer.zero_grad()

        x = inr(coords).view(res)
        Ax = torch.fft.fft2(x,norm="ortho")[mask]
        mseloss = MSE(torch.real(Ax),torch.real(y))+MSE(torch.imag(Ax),torch.imag(y))
        wd_reg = inr.weight_decay()
        loss = mseloss + lam*wd_reg

        loss.backward()
        optimizer.step()
        scheduler.step()

        # log metrics
        if (iter+1) % settings["logfreq"] == 0:
            with torch.no_grad():
                imgmse = MSE(x,x00)  #MSE with ground truth rasterized phantom
                wandb.log({
                    "iter": iter+1,
                    "loss": loss.item(),
                    "DataMSE": mseloss.item(),
                    "WDReg":wd_reg.item(),
                    "ImgMSE":imgmse.item()
                })
                plot_inr_data(coords,inr(coords),
                    nrows=nx,ncols=nx,nchannels=nchannels,
                title=f"current INR",subtitle=f' at iter {iter}')
                plot_inr_data(coords,1-torch.abs(inr(coords)-x00.reshape(-1,1)),
                    nrows=nx,ncols=nx,nchannels=nchannels,
                    title=f"errors in current INR",subtitle=f' at iter {iter}')

    #compute final metrics
    with torch.no_grad():
        x = inr(coords).view(res)
        Ax = torch.fft.fft2(x,norm="ortho")[mask]
        mseloss = MSE(torch.real(Ax),torch.real(y))+MSE(torch.imag(Ax),torch.imag(y))
        wd_reg = inr.weight_decay()
        loss = mseloss + lam*wd_reg
        imgmse = MSE(x,x00)

        wandb.log({
            "iter": iter+1,
            "loss": loss.item(),
            "DataMSE": mseloss.item(),
            "WDReg":wd_reg.item(),
            "ImgMSE":imgmse.item()
        })
        plot_inr_data(coords,inr(coords),
            nrows=nx,ncols=nx,nchannels=nchannels,
        title=f"current INR",subtitle=f' at end of training')
        plot_inr_data(coords,1-torch.abs(inr(coords)-x00.reshape(-1,1)),
            nrows=nx,ncols=nx,nchannels=nchannels,
            title=f"errors in current INR",subtitle=f' at end of training')

    x = x.cpu().numpy()
    metrics = {}
    metrics["final_loss"] = loss.item()
    metrics["final_mseloss"] = mseloss.item()
    metrics["final_wd_reg"] = wd_reg.item()
    metrics["final_imgmse"] = imgmse.item()
    
    return inr, x, metrics

if __name__ == "__main__":
    main()