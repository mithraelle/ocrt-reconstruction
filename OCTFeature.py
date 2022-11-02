import numpy as np


class OCTFeatureMatch:
    x0: [float] = []
    x1: [float] = []
    active: bool = True

    def __init__(self, x0: [float], x1: [float]):
        self.x0 = x0
        self.x1 = x1
        self.active = True

    def get_center(self):
        return 0.5*(self.x0+self.x1)

    def set_state(self, active: bool):
        self.active = active

    def get_state(self):
        return self.active


class OCTFeatureMatchSet:
    idx0: int = 0
    idx1: int = 0

    features: [OCTFeatureMatch]

    raw_count: int = 0

    def __init__(self, idx0: int, idx1: int):
        self.idx0 = idx0
        self.idx1 = idx1
        self.features = []

    def add_feature(self, x0: [float], x1: [float]):
        self.features.append(OCTFeatureMatch(x0, x1))

    def add_feature_set(self, feature_set):
        for feature in feature_set:
            self.add_feature(feature[0], feature[1])

    def filter_matches(self, img_size: [float], max_dist: float = 50, active_area: float = 0.66, z_scale: float = 1.0, t: [float] = []):
        res = OCTFeatureMatchSet(self.idx0, self.idx1)

        window_size = np.dot(active_area, img_size)
        c0 = (img_size - window_size) / 2
        c1 = (img_size + window_size) / 2

        for i in range(len(self.features)):
            feature = self.features[i]

            if c0[0] <= feature.x0[0] <= c1[0] and c0[1] <= feature.x0[1] <= c1[1]:
                x0 = np.array(feature.x0)
                x0[1] = x0[1] * z_scale
                x1 = np.array(feature.x1)
                x1[1] = x1[1] * z_scale

                if len(t) != 0:
                    x1[0] = x1[0] * t[0, 0] + x1[1] * t[0, 1] + t[0, 2]
                    x1[1] = x1[0] * t[1, 0] + x1[1] * t[1, 1] + t[1, 2]

                dist = np.linalg.norm(x0 - x1)
                if dist <= max_dist:
                    res.add_feature(feature.x0, feature.x1)

        return res


class OCTFeatureMatchSetList:
    feature_match_set: [OCTFeatureMatchSet]

    def __init__(self):
        self.feature_match_set = []

    def append(self, match_set: OCTFeatureMatchSet):
        self.feature_match_set.append(match_set)

    def filter_matches(self, img_size: [float], dist: float = 50, active_area: float = 0.66, z_scale: float = 1, t: [float] = []):
        res = OCTFeatureMatchSetList()
        for i in self.feature_match_set:
            res.append(i.filter_matches(img_size, dist, active_area, z_scale, t))

        return res

    def get_feature_match_set(self, idx: int):
        return self.feature_match_set[idx]
