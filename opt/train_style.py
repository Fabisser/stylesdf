import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch

from torch.autograd import Variable
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline

import utils.general as utils
from utils import rend_util
import utils.plots as plt
from implicitSDF_style import StyleNetwork
from implicitSDF import IDRNetwork
from loss_style import LossStyle

class StyleTrainRunner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(os.path.join('../', 'conf', kwargs['conf']))
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.train_cameras = False #Don't need to train cameras anymore
        self.styleimg = kwargs['style']
        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_int('dataset.scan_id', default=-1)
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        if kwargs['timestamp'] == 'latest':
            timestamps = os.listdir(os.path.join('../', 'exps', self.expname))
            timestamp = sorted(timestamps)[-1]
        else:
            timestamp = kwargs['timestamp']
        
        self.num_pixels = self.conf.get_list('style.num_pixels')
        self.loss_names = self.conf.get_list('style.losses')
        self.content = self.conf.get_bool("style.content")
        self.block = self.conf.get_list('style.block')
        self.contw = self.conf.get_float("style.contentweight")
        self.gramw = self.conf.get_float("style.gramweight")

        
        utils.mkdir_ifnotexists(os.path.join('../', self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

        os.system(
            """cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(self.train_cameras,
                                                                                          **dataset_conf)

        print('Finish loading data ...')

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )

        self.model = utils.get_class(self.conf.get_string('style.model_class'))(conf=self.conf.get_config('model'))
        
        IDR_checkpoint_dir = os.path.join('../', 'exps', self.expname, timestamp, 'checkpoints')
        
        saved_model_state = torch.load(os.path.join(IDR_checkpoint_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
        self.model.load_state_dict(saved_model_state["model_state_dict"])
        if torch.cuda.is_available():
            self.model.cuda()
        device = torch.device("cuda:0")  
        self.loss = utils.get_class(self.conf.get_string('style.loss_class'))(device=device)
        
        self.lr = self.conf.get_float('style.learning_rate')
        self.optimizer = torch.optim.Adam(self.model.rendering_network.parameters(), lr=self.lr)
        
        self.sched_milestones = self.conf.get_int('style.step_size', default=0)
        self.sched_factor = self.conf.get_float('self.sched_factor', default=0.0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.sched_milestones, gamma=self.sched_factor)
        self.start_epoch = 0

        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('style.plot_freq')
        self.plot_conf = self.conf.get_config('plot')

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))
        
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))
            
    def style(self):
        print("Training style...")
        
        
        prepstyle = transforms.Compose([
                    transforms.Resize(self.num_pixels, antialias=True),
                    transforms.ToTensor(),
                    ])   
        for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
            
            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input["object_mask"] = model_input["object_mask"].cuda()
            model_input['pose'] = model_input['pose'].cuda()
            
            split = utils.split_input(model_input, self.total_pixels)
            res = []
            for s in split:
                out = self.model(s)
                res.append({
                    'points': out['points'].detach(),
                    'surface_mask': out['surface_mask'].detach(),
                    'differentiable_surface_points': out['differentiable_surface_points'].detach(),
                    'normals': out['normals'].detach(),
                    'view': out['view'].detach(),
                    'feature_vectors': out['feature_vectors'].detach()
                }) 

            batch_size = ground_truth['rgb'].shape[0]
                   
            if self.content:
                gt = ground_truth["rgb"].cuda().squeeze().permute(1, 0).reshape([3, self.img_res[0], self.img_res[1]])
                gt = transforms.Resize(self.num_pixels, antialias=True)(gt)
                content_image = Variable(gt.unsqueeze(0).cuda())
            else:
                content_image = None
            
            styleimg = Image.open('../Style_Images/{}.jpg'.format(self.styleimg))
            
            styleimg = prepstyle(styleimg)
            
            style_image = Variable(styleimg.unsqueeze(0).cuda())

            n_iter=[0]          

            while n_iter[0] <= self.nepochs:
                self.optimizer.zero_grad()
                def styler():
                    render_out = []
                    
                    for r in res:
                        rgb_values = torch.ones_like(r['points']).float()

                        with torch.no_grad():
                            render_values = self.model.rendering_network(r['differentiable_surface_points'],
                                                                                            r['normals'],
                                                                                            r['view'],
                                                                                            r['feature_vectors'])
                        render_values.cuda().requires_grad_(True)
                        
                        rgb_values[r['surface_mask']] = render_values

                        render_out.append({"input" : rgb_values, "render_values" : render_values, "mask" : r['surface_mask']})
                    
                    rgb_values = utils.merge_output(render_out, self.total_pixels, batch_size)
                    input = (rgb_values["input"] + 1.) / 2.
                                           
                    input = input.permute(1, 0).reshape([3, self.img_res[0], self.img_res[1]]).contiguous()
                                   

                    opt_img = torch.unsqueeze(transforms.Resize(self.num_pixels, antialias=True)(input), 0)

                    loss = self.loss(opt_img, style_image, blocks = self.block, loss_names=self.loss_names, contents=content_image, contw=self.contw, gramw=self.gramw)
                    
                    loss.backward()
                    n_iter[0]+=1
                    
                    print('{0} [{1}/{2}] ({3}/{4}): loss = {5}'.format(self.expname, n_iter[0], self.nepochs, data_index, self.n_batches, loss.item()))
                    
                    if n_iter[0] % self.plot_freq == 0:
                        
                        input[input>1] = 1    
                        input[input<0] = 0
                        
                        out_img = transforms.ToPILImage()(input)
                        out_img.save(os.path.join(self.plots_dir, str(n_iter[0]) + ".jpg"))

                        self.save_checkpoints(data_index)
                    output = []
                    for r in render_out:
                        
                        rgb_pred_grad = r["render_values"].grad.contiguous().clone().detach().view(-1, 3)
                        rgb_pred = r["render_values"].contiguous().clone().detach()

                        output.append({"rgb_pred" : rgb_pred, "grad" :  rgb_pred_grad})
                    
                    return loss, output
            
    
                loss, rgb_pred = styler()
                

                for idx,r in enumerate(res):

                    rgb_values = self.model.rendering_network(r['differentiable_surface_points'],
                                                                                    r['normals'],
                                                                                    r['view'],
                                                                                    r['feature_vectors'])
                    
                    rgb_values.backward(rgb_pred[idx]["grad"])

                self.optimizer.step()

            self.scheduler.step()
  
    def render(self, epoch):
        
        saved_model_state = torch.load(os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        self.model.load_state_dict(saved_model_state["model_state_dict"])
        self.model.eval()

        gt_pose = self.train_dataset.get_gt_pose(scaled=True).cuda()
        gt_quat = rend_util.rot_to_quat(gt_pose[:, :3, :3])
        gt_pose_vec = torch.cat([gt_quat, gt_pose[:, :3, 3]], 1)

        indices_all = [11, 16, 34, 28, 11]
        pose = gt_pose_vec[indices_all, :]
        t_in = np.array([0, 2, 3, 5, 6])

        n_inter = 5

        t_out = np.linspace(t_in[0], t_in[-1], n_inter * t_in[-1]).astype(np.float32)
        scales = np.array([4.2, 4.2, 3.8, 3.8, 4.2]).astype(np.float32)
        #scales = np.array([1.2, 1.2, 0.8, 0.8, 1.2]).astype(np.float32)

        s_new = CubicSpline(t_in, scales, bc_type='periodic')
        s_new = s_new(t_out)

        q_new = CubicSpline(t_in, pose[:, :4].detach().cpu().numpy(), bc_type='periodic')
        q_new = q_new(t_out)
        q_new = q_new / np.linalg.norm(q_new, 2, 1)[:, None]
        q_new = torch.from_numpy(q_new).cuda().float()

        images_dir = "../Render_{}".format(self.expname)
        utils.mkdir_ifnotexists(images_dir)

        indices, model_input, ground_truth = next(iter(self.train_dataloader))

        for i, (new_q, scale) in enumerate(zip(q_new, s_new)):
            torch.cuda.empty_cache()

            new_q = new_q.unsqueeze(0)
            new_t = -rend_util.quat_to_rot(new_q)[:, :, 2] * scale

            new_p = torch.eye(4).float().cuda().unsqueeze(0)
            new_p[:, :3, :3] = rend_util.quat_to_rot(new_q)
            new_p[:, :3, 3] = new_t

            sample = {
                "object_mask": torch.zeros_like(model_input['object_mask']).cuda().bool(),
                "uv": model_input['uv'].cuda(),
                "intrinsics": model_input['intrinsics'].cuda(),
                "pose": new_p
            }

            split = utils.split_input(sample, self.total_pixels)
            res = []
            for s in split:
                out = self.model(s)
                rgb_values = torch.ones_like(out['points']).float()
                out["view"] = torch.zeros_like(out['view']).float()
                rgb_values[out['surface_mask']] = self.model.rendering_network(out['differentiable_surface_points'],
                                                                out['normals'],
                                                                out['view'],
                                                                out['feature_vectors'])
                res.append({
                    'rgb_values': rgb_values.detach(),
                })

            batch_size = 1
            model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
            rgb_eval = model_outputs['rgb_values']
            rgb_eval = rgb_eval.reshape(batch_size, self.total_pixels, 3)

            rgb_eval = (rgb_eval + 1.) / 2.
            rgb_eval = plt.lin2img(rgb_eval, self.img_res).detach().cpu().numpy()[0]
            rgb_eval = rgb_eval.transpose(1, 2, 0)
            img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
            img.save('{0}/eval_{1}.png'.format(images_dir,'%03d' % i))
                    
