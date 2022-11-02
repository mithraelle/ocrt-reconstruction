import cv2


def OCTFeatureDetectionORB(img0, img1):
    orb = cv2.ORB_create()

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    kp0 = orb.detect(img0, None)
    kp0, des0 = orb.compute(img0, kp0)

    kp1 = orb.detect(img1, None)
    kp1, des1 = orb.compute(img1, kp1)

    matches = flann.knnMatch(des0, des1, k=2)

    return matches, kp0, kp1


def OCTFeatureDetectionSIFT(img0, img1):
    sift = cv2.SIFT_create()

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    kp0, des0 = sift.detectAndCompute(img0, None)
    kp1, des1 = sift.detectAndCompute(img1, None)

    matches = flann.knnMatch(des0, des1, k=2)

    return matches, kp0, kp1


def OCTFeatureDetectionSURF(img0, img1):
    surf = cv2.xfeatures2d.SURF_create(400)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    kp0, des0 = surf.detectAndCompute(img0, None)
    kp1, des1 = surf.detectAndCompute(img1, None)

    matches = flann.knnMatch(des0, des1, k=2)

    return matches, kp0, kp1


OCTFeatureDetectionMethodsList = [
    {"name": "SIFT", "method": OCTFeatureDetectionSIFT},
    {"name": "SURF", "method": OCTFeatureDetectionSURF},
    {"name": "ORB", "method": OCTFeatureDetectionORB},
]


def get_method_by_name(name: str):
    for i in OCTFeatureDetectionMethodsList:
        if i["name"] == name:
            return i["method"]
    return False
