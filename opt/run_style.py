import sys
import argparse
import GPUtil
import numpy as np
import matplotlib.pyplot as lossplt
from train_style import StyleTrainRunner


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='DTU_style.conf')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--timestamp', default='latest', type=str, help='The timestamp of the run to be used when styling.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The checkpoint epoch number of the run to be used when styling.')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--style', type=str, default="vangogh_starry_night", help='Style image to use, should be filename.jpg found in StyleImages.')

    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu
    
    trainrunner = StyleTrainRunner(conf=opt.conf,
                                    batch_size=opt.batch_size,
                                    nepochs=opt.nepoch,
                                    expname=opt.expname,
                                    gpu_index=gpu,
                                    exps_folder_name='exps_style',
                                    timestamp=opt.timestamp,
                                    checkpoint=opt.checkpoint,
                                    scan_id=opt.scan_id,
                                    style=opt.style
                                    )

    
    
    trainrunner.style()
    
    trainrunner.render("latest")