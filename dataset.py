import glob
import cv2
import numpy as np
import scipy.io as sio

class __AbstractDataset(object):
    def load_img(self, path):
        raise NotImplementedError

    def load_ann(self, path, with_type=False):
        raise NotImplementedError


class __Kumar(__AbstractDataset):
    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        assert not with_type, 'Not support'
        ann_inst = sio.loadmat(path)['inst_map']
        ann_inst = ann_inst.astype('int32')
        ann = np.expand_dims(ann_inst, axis=-1)
        return ann


class __CPM17(__AbstractDataset):
    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        assert not with_type, "Not support"
        ann_inst = sio.loadmat(path)["inst_map"]
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)
        return ann


class __CPM15(__AbstractDataset):
    def name(self):
        print('cpm15')


    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        assert not with_type, "Not support"
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)["inst_map"]
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)
        return ann


class __TNBC(__AbstractDataset):
    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        assert not with_type, "Not support"
        ann_inst = sio.loadmat(path)["inst_map"]
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)
        return ann


def get_dataset(name):
    name_dict = {
        'kumar': lambda: __Kumar(),
        'cpm17': lambda: __CPM17(),
        'cpm15': lambda: __CPM15(),
        'tnbc': lambda: __TNBC(),
    }
    if name.lower() in name_dict:
        return name_dict[name]()
    else:
        assert False, "Unknown dataset `%s`" % name
