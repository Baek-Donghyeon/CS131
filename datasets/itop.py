import os
import numpy as np
from torch.utils.data import Dataset
import h5py


def pixel2world(x, y, z, img_width, img_height, fx, fy):
    w_x = (x - img_width / 2) * z / fx
    w_y = (img_height / 2 - y) * z / fy
    w_z = z
    return w_x, w_y, w_z


def depthmap2points(image, fx, fy):
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:,:,0], points[:,:,1], points[:,:,2] = pixel2world(x, y, image, w, h, fx, fy)
    return points


def load_depthmap(filename, index):
    with h5py.File(filename, 'r') as f:
        data = f['data'][index]
        data = np.asarray(data)
    return data


class ITOPDataset(Dataset):
    def __init__(self, root, center_dir, point_of_view, mode, transform=None):
        self.img_width = 320
        self.img_height = 240
        self.min_depth = 100
        self.max_depth = 700
        self.fx = 285.714
        self.fy = 285.714
        self.joint_num = 15
        self.world_dim = 3
        self.subject_num = 9
        self.train_size = 39795
        self.test_size = 10501

        self.root = root
        self.center_dir = center_dir
        self.point_of_view = point_of_view
        self.mode = mode
        self.transform = transform

        if not self.mode in ['train', 'test']: raise ValueError('Invalid mode')
        
        self._load()
        
    def get_data(self):
        return self.names, self.joints_world, self.ref_pts
    
    def __getitem__(self, index):
        filename = os.path.join(self.root, 'ITOP_' + self.point_of_view + '_' + self.mode + '_depth_map.h5')
        depthmap = load_depthmap(filename, self.names[index])
        points = depthmap2points(depthmap, self.fx, self.fy)
        points = points.reshape((-1, 3))
        sample = {
            'name': self.names[index],
            'points': points,
            'joints': self.joints_world[index],
            'refpoint': self.ref_pts[index]
        }

        if self.transform: sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.num_samples

    def _load(self):
        self.num_samples = self.train_size if self.mode == 'train' else self.test_size
        self.joints_world = np.zeros((self.num_samples, self.joint_num, self.world_dim))
        self.ref_pts = np.zeros((self.num_samples, self.world_dim))
        self.names = []
        
        with h5py.File(os.path.join(self.root, 'ITOP_' + self.point_of_view + '_' + self.mode + '_labels.h5'), 'r') as f:
            joints_world_original = f['real_world_coordinates'][:]

        with open(os.path.join(self.center_dir, self.point_of_view + '_center_' + self.mode + '.txt')) as f:
            ref_pt_str = [l.rstrip() for l in f]

        frame_id = 0
        invalid_frame_num = 0

        for fid in range(self.num_samples):
        
            splitted = ref_pt_str[fid].split()
            if splitted[0] == 'invalid':
                invalid_frame_num += 1
                continue
            else:
                self.ref_pts[frame_id, 0] = float(splitted[0])
                self.ref_pts[frame_id, 1] = float(splitted[1])
                self.ref_pts[frame_id, 2] = float(splitted[2])

            self.joints_world[frame_id] = joints_world_original[fid]
            self.names.append(fid)

            frame_id += 1
         
        if invalid_frame_num != 0:    
            self.joints_world = self.joints_world[:-invalid_frame_num]
            self.ref_pts = self.ref_pts[:-invalid_frame_num]
        self.num_samples -= invalid_frame_num
