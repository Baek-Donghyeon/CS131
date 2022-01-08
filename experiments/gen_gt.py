##%%
import numpy as np
from datasets.itop import ITOPDataset

##%%
# Generate train_subject3_gt.txt and test_subject3_gt.txt
data_dir = r'./datasets/itop'
center_dir = r'./datasets/itop/itop_center'

##%%
def save_keypoints(filename, keypoints):
    # Reshape one sample keypoints into one line
    keypoints = keypoints.reshape(keypoints.shape[0], -1)
    np.savetxt(filename, keypoints, fmt='%0.4f')


##%%
train_dataset = ITOPDataset(root=data_dir, center_dir=center_dir, point_of_view='side', mode='train')

names, joints_world, ref_pts = train_dataset.get_data()
print('save train reslt ..')
save_keypoints('./res/train_s3_gt.txt', joints_world)
np.savetxt('./res/train_name_gt.txt', names, fmt='%d')
print('done ..')


##%%
test_dataset = ITOPDataset(root=data_dir, center_dir=center_dir, point_of_view='side', mode='test')
names, joints_world, ref_pts = test_dataset.get_data()
print('save test reslt ..')
save_keypoints('./res/test_s3_gt.txt', joints_world)
np.savetxt('./res/test_name_gt.txt', names, fmt='%d')
print('done ..')