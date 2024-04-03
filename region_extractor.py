import cv2
import numpy as np
from ultralytics import YOLO
import torch
from facemesh import FaceMesh
# import line_profiler

def pad_bbox(image, bbox, padding=10):
    """
    Selects the bbox with some padding applied to all sides and checks that the padding is valid
    else doesn't apply it to that particular side
    :param image:
    :param bbox:
    :param padding:
    :return:
    """
    x1, y1, x2, y2 = bbox.astype(int)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    return image[y1:y2, x1:x2]


class FaceExtractor:
    NOSE = np.array([8, 193, 245, 114, 126, 129, 98, 240, 97, 328, 460, 294, 278, 360, 465, 417, 8], dtype=np.int32)
    NOSE_BRIDGE = np.array([203, 98, 97, 2, 326, 327, 423, 426, 436, 410, 409, 270, 269, 267, 0, 37, 39, 40, 186, 216,
                            206, 203], dtype=np.int32)
    FOREHEAD = np.array(
        [10, 338, 297, 332, 284, 251, 389, 368, 383, 300, 293, 334, 296, 336, 9, 55, 65, 52, 53, 46, 70, 71, 21,
         54, 103, 67, 109, 10], dtype=np.int32)
    CHEEK_LEFT = np.array([121, 111, 216, 129, 121], dtype=np.int32)
    CHEEK_LEFT = np.array(
        [143, 35, 31, 228, 229, 230, 231, 120, 100, 142, 129, 203, 206, 216, 214, 192, 213, 177, 137, 227, 34,
         143], dtype=np.int32)

    def __init__(self, yolov8_weigths_path, facemesh_weights_path):
        # Masked Regions stored in format [N, 3, H, W, C], N = number of images, 3 = number of regions
        self.detection_model = YOLO(yolov8_weigths_path)
        self.facemesh_model = FaceMesh()
        self.facemesh_model.load_weights(facemesh_weights_path)
        print("Models Initialized")
        self.images = []
        self.masks = None
        self.keypoints_list = []
        self.regions_of_interest = []
        self.motions = []
        self.bboxes = []
        self.face_ROIs = []
        self.coord_landmarks = None

    # @profile
    def add_image(self, image, resolution=(1920,1080)):
        image = cv2.resize(image, (resolution[0], resolution[1]))
        self.images.append(image)

    # @profile
    def process_images(self):
        self.get_face_landmarks()
        self.select_regions()

    # @profile
    def clear_images(self):
        self.images = []
        self.keypoints_list = []
        self.bboxes = []
        self.face_ROIs = []
        self.masks = None
        self.coord_landmarks = None

    # @profile
    def show_results(self):
        for image in self.images:
            cv2.imshow("Images", image)
            cv2.waitKey(40)
        for (image, mask) in zip(self.images, self.masks):
                cv2.imshow("Region", cv2.bitwise_and(image, image, mask=mask[0]))
                cv2.waitKey(40)

    # @profile
    def get_face_landmarks(self):
        results = self.detection_model(self.images, stream=True)
        for i, result in enumerate(results):
            try:
                self.bboxes.append(result.boxes.xyxy[0].cpu().numpy())
                self.face_ROIs.append(pad_bbox(self.images[i], result.boxes.xyxy[0].cpu().numpy()))
                self.keypoints_list.append(result.keypoints.xy.cpu().numpy()[0])
                if len(self.keypoints_list) < 2:
                    self.motions.append(0)
                    continue
                center = np.mean(self.keypoints_list[-1], axis=0)
                center_old = np.mean(self.keypoints_list[-2], axis=0)
                motion = np.linalg.norm(center - center_old)
                self.motions.append(motion)
            except IndexError:
                continue

    # @profile
    def generate_mesh(self):
        self.bboxes = np.array(self.bboxes)
        resized_images = [cv2.resize(image, (192, 192)) for image in self.face_ROIs]
        resized_images = np.array(resized_images)
        face_mesh_points = self.facemesh_model.predict_on_batch(resized_images)[0].to('cpu').numpy()
        xs, ys = [x[:, 0] for x in face_mesh_points], [x[:, 1] for x in face_mesh_points]
        coord_landmarks = np.array([xs, ys]).transpose(1, 2, 0)

        heights = self.bboxes[:, 3] - self.bboxes[:, 1] + 20
        widths = self.bboxes[:, 2] - self.bboxes[:, 0] + 20
        scale_x = [x / 192 for x in widths]
        scale_y = [y / 192 for y in heights]
        scale = np.array([scale_x, scale_y]).transpose(1, 0)
        coord_landmarks *= scale.reshape((-1, 1, 2))
        coord_landmarks += self.bboxes[:, :2].reshape((-1, 1, 2)) - 10
        self.coord_landmarks = coord_landmarks

    # @profile
    def select_regions(self):
        self.generate_mesh()
        # Selecting the regions
        points_nose = np.array(self.coord_landmarks[:, self.NOSE], dtype=np.int32)
        points_nose_bridge = np.array(self.coord_landmarks[:, self.NOSE_BRIDGE], dtype=np.int32)
        points_forehead = np.array(self.coord_landmarks[:, self.FOREHEAD], dtype=np.int32)
        self.mesh_points = [points_nose, points_nose_bridge, points_forehead]
        self.images = np.array(self.images)
        masks_nose = np.zeros(shape=(*self.images.shape[:3], 1), dtype=np.uint8)
        masks_nose = [cv2.fillPoly(mask_nose, [points_nose[i]], 255) for i, mask_nose in
                      enumerate(masks_nose)]
        masks_nose_bridge = np.zeros(shape=(*self.images.shape[:3], 1), dtype=np.uint8)
        masks_nose_bridge = [cv2.fillPoly(mask_nose_bridge, [points_nose_bridge[i]], 255) for
                             i, mask_nose_bridge in enumerate(masks_nose_bridge)]
        masks_forehead = np.zeros(shape=(*self.images.shape[:3], 1), dtype=np.uint8)
        masks_forehead = [cv2.fillPoly(mask_forehead, [points_forehead[i]], 255) for i, mask_forehead in
                          enumerate(masks_forehead)]
        # Nose region is sum of nose and nose bridge
        masks_nose = np.bitwise_or(masks_nose, masks_nose_bridge)

        self.masks = np.array([masks_nose, masks_nose_bridge, masks_forehead]).transpose(1, 0, 2, 3, 4)
