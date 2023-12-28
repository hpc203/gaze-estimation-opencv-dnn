import cv2
import numpy as np
import math
import argparse

class YOLOv8_face:
    def __init__(self, path, conf_thres=0.2, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.class_names = ['face']
        self.num_classes = len(self.class_names)
        # Initialize model
        self.net = cv2.dnn.readNet(path)
        self.input_height = 640
        self.input_width = 640
        self.reg_max = 16

        self.project = np.arange(self.reg_max)
        self.strides = (8, 16, 32)
        self.feats_hw = [(math.ceil(self.input_height / self.strides[i]), math.ceil(self.input_width / self.strides[i]))
                         for i in range(len(self.strides))]
        self.anchors = self.make_anchors(self.feats_hw)

    def make_anchors(self, feats_hw, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h, w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset  # shift x
            y = np.arange(0, h) + grid_cell_offset  # shift y
            sx, sy = np.meshgrid(x, y)
            # sy, sx = np.meshgrid(y, x)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points

    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_width, self.input_height
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(self.input_width / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_width - neww - left, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))  # add border
            else:
                newh, neww = int(self.input_height * hw_scale), self.input_width
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_height - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_height - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
        else:
            img = cv2.resize(srcimg, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def detect(self, srcimg):
        input_img, newh, neww, padh, padw = self.resize_image(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB))
        scale_h, scale_w = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        input_img = input_img.astype(np.float32) / 255.0

        blob = cv2.dnn.blobFromImage(input_img)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        # if isinstance(outputs, tuple):
        #     outputs = list(outputs)
        # if float(cv2.__version__[:3])>=4.7:
        #     outputs = [outputs[2], outputs[0], outputs[1]] ###opencv4.7需要这一步，opencv4.5不需要
        # Perform inference on the image
        det_bboxes, det_conf, det_classid, landmarks = self.post_process(outputs, scale_h, scale_w, padh, padw)
        return det_bboxes, det_conf, det_classid, landmarks

    def post_process(self, preds, scale_h, scale_w, padh, padw):
        bboxes, scores, landmarks = [], [], []
        for i, pred in enumerate(preds):
            stride = int(self.input_height / pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))

            box = pred[..., :self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1, 1))
            kpts = pred[..., -15:].reshape((-1, 15))  ### x1,y1,score1, ..., x5,y5,score5

            # tmp = box.reshape(self.feats_hw[i][0], self.feats_hw[i][1], 4, self.reg_max)
            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = self.softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1, 4))

            bbox = self.distance2bbox(self.anchors[stride], bbox_pred,
                                      max_shape=(self.input_height, self.input_width)) * stride
            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (self.anchors[stride][:, 0].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (self.anchors[stride][:, 1].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 2::3] = 1 / (1 + np.exp(-kpts[:, 2::3]))

            bbox -= np.array([[padw, padh, padw, padh]])  ###合理使用广播法则
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            kpts -= np.tile(np.array([padw, padh, 0]), 5).reshape((1, 15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1, 15))

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)

        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  ####xywh
        classIds = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)  ####max_class_confidence

        mask = confidences > self.conf_threshold
        bboxes_wh = bboxes_wh[mask]  ###合理使用广播法则
        confidences = confidences[mask]
        classIds = classIds[mask]
        landmarks = landmarks[mask]

        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.conf_threshold, self.iou_threshold)
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        if len(indices) > 0:
            mlvl_bboxes = bboxes_wh[indices]
            confidences = confidences[indices]
            classIds = classIds[indices]
            landmarks = landmarks[indices]
            return mlvl_bboxes, confidences, classIds, landmarks
        else:
            return np.array([]), np.array([]), np.array([]), np.array([])

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

def angles_from_vec(vec):
    x, y, z = -vec[2], vec[1], -vec[0]
    theta = np.arctan2(y, x)
    phi = np.arctan2(np.sqrt(x**2 + y**2), z) - np.pi/2
    theta_x, theta_y = phi, theta
    return theta_x, theta_y

def vec_from_eye(eye, iris_lms_idx):
    p_iris = eye[iris_lms_idx] - eye[:32].mean(axis=0)
    vec = p_iris.mean(axis=0)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def angles_and_vec_from_eye(eye, iris_lms_idx):
    vec = vec_from_eye(eye, iris_lms_idx)
    theta_x, theta_y = angles_from_vec(vec)
    return theta_x, theta_y, vec


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale
    return new_pts
    
class GazeEstimation:
    def __init__(self, model_path):
        self.net = cv2.dnn.readNet(model_path)
        self.iris_idx_481 = np.array([248, 252, 224, 228, 232, 236, 240, 244], dtype=np.int64)
        self.num_eye = 481
        self.input_size = 160
    
    def transform(self, data, center, output_size, scale):
        cx = center[0] * scale
        cy = center[1] * scale
        M = np.array([[scale, 0, -cx + output_size*0.5],[0,scale, -cy+output_size*0.5]], dtype=np.float32)
        cropped = cv2.warpAffine(
            data,
            M,
            (output_size, output_size),
            borderValue=0.0
        )
        return cropped, M
    
    def draw_item(self, eimg, item):
        eye_kps = item
        eye_l = eye_kps[:self.num_eye,:]
        eye_r = eye_kps[self.num_eye:,:]
        for _eye in [eye_l, eye_r]:
            tmp = _eye[:,0].copy()
            _eye[:,0] = _eye[:,1].copy()  ###交换x,y
            _eye[:,1] = tmp

        theta_x_l, theta_y_l, vec_l = angles_and_vec_from_eye(eye_l, self.iris_idx_481)
        theta_x_r, theta_y_r, vec_r = angles_and_vec_from_eye(eye_r, self.iris_idx_481)
        gaze_pred = np.array([(theta_x_l + theta_x_r) / 2, (theta_y_l + theta_y_r) / 2])

        diag = np.sqrt(float(eimg.shape[0]*eimg.shape[1]))

        eye_pos_left = eye_l[self.iris_idx_481].mean(axis=0)[[0, 1]]
        eye_pos_right = eye_r[self.iris_idx_481].mean(axis=0)[[0, 1]]

        ## pred
        gaze_pred = np.array([theta_x_l, theta_y_l])
        dx = 0.4*diag * np.sin(gaze_pred[1])
        dy = 0.4*diag * np.sin(gaze_pred[0])
        x = np.array([eye_pos_left[1], eye_pos_left[0]])
        y = x.copy()
        y[0] += dx
        y[1] += dy
        x = x.astype(np.int32)
        y = y.astype(np.int32)
        color = (0,0,255)
        cv2.arrowedLine(eimg, x, y, color, 5)
        # Yaw, Pitch
        yaw_deg_l = theta_y_l * (180 / np.pi)
        pitch_deg_l = -(theta_x_l * (180 / np.pi))


        gaze_pred = np.array([theta_x_r, theta_y_r])
        dx = 0.4*diag * np.sin(gaze_pred[1])
        dy = 0.4*diag * np.sin(gaze_pred[0])
        x = np.array([eye_pos_right[1], eye_pos_right[0]])
        y = x.copy()
        y[0] += dx
        y[1] += dy
        x = x.astype(np.int32)
        y = y.astype(np.int32)
        color = (0,0,255)
        cv2.arrowedLine(eimg, x, y, color, 5)
        # Yaw, Pitch
        yaw_deg_r = theta_y_r * (180 / np.pi)
        pitch_deg_r = -(theta_x_r * (180 / np.pi))
        return eimg, yaw_deg_l, pitch_deg_l, yaw_deg_r, pitch_deg_r
    def draw_on(self, eimg, result):
        max_face_size = result[0][2] - result[0][0]
        rescale = 300.0 / max_face_size
        oimg = eimg.copy()
        eimg = cv2.resize(eimg, None, fx=rescale, fy=rescale)

        _, _, eye_kps = result
        eye_kps = eye_kps.copy()
        eye_kps *= rescale
        eimg, yaw_deg_l, pitch_deg_l, yaw_deg_r, pitch_deg_r = self.draw_item(eimg, eye_kps)
            
        eimg = cv2.resize(eimg, (oimg.shape[1], oimg.shape[0]))
        # Yaw, Pitch
        cv2.putText(
            eimg,
            f"L-Yaw : {yaw_deg_l:5.2f}",
            (int(eimg.shape[1]-200), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            eimg,
            f"L-Pitch : {pitch_deg_l:5.2f}",
            (int(eimg.shape[1]-200), 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            eimg,
            f"R-Yaw : {yaw_deg_r:5.2f}",
            (int(eimg.shape[1]-200), 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            eimg,
            f"R-Pitch : {pitch_deg_r:5.2f}",
            (int(eimg.shape[1]-200), 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            cv2.LINE_AA,
        )
        return eimg
    
    def detect(self, facebox, img):
        image_width = img.shape[1]
        image_height = img.shape[0]
        
        x_min = max(int(facebox[0]), 0)
        y_min = max(int(facebox[1]), 0)
        x_max = min(int(facebox[2]), image_width)
        y_max = min(int(facebox[3]), image_height)

        bbox = [x_min, y_min, x_max, y_max]
        kps = facebox[4:]
        kps_right_eye = np.asarray([int(facebox[4]), int(facebox[5])], dtype=np.int32) # [x, y]
        kps_left_eye = np.asarray([int(facebox[6]), int(facebox[7])], dtype=np.int32) # [x, y]
        width = x_max - x_min
        center = (kps_left_eye + kps_right_eye) / 2.0 # (lx + rx) / 2, (ly + ry) / 2

        _size = max(width/1.5, np.abs(kps_right_eye[0] - kps_left_eye[0])) * 1.5
        _scale = self.input_size  / _size
        aimg, M = self.transform(img, center, self.input_size, _scale)
        aimg = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)

        input_face_image = (aimg.astype(np.float32) / 255.0 - 0.5) / 0.5
        blob = cv2.dnn.blobFromImage(input_face_image)
        self.net.setInput(blob)
        opred = self.net.forward(self.net.getUnconnectedOutLayersNames())[0].squeeze(axis=0)
        IM = cv2.invertAffineTransform(M)
        pred = trans_points3d(opred, IM)
        result = (bbox, kps, pred)
        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='images/1.jpeg', help="image path")
    parser.add_argument('--confThreshold', default=0.45, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    parser.add_argument('--videopath', type=str, default=None, help="video path")
    args = parser.parse_args()

    # Initialize YOLOv8_face object detector
    face_detector = YOLOv8_face("weights/yolov8n-face.onnx", conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)
    gaze_predictor = GazeEstimation("weights/generalizing_gaze_estimation_with_weak_supervision_from_synthetic_views_1x3x160x160.onnx")

    if args.videopath is None:
        srcimg = cv2.imread(args.imgpath)
        # Detect Objects
        boxes, scores, classids, kpts = face_detector.detect(srcimg)
        drawimg = srcimg.copy()
        for i, box in enumerate(boxes):
            x, y, w, h = box.astype(int)
            facebox = [x, y, x+w, y+h, kpts[i, 3], kpts[i, 4], kpts[i, 0], kpts[i, 1]]
            result = gaze_predictor.detect(facebox, srcimg)
            drawimg = gaze_predictor.draw_on(drawimg, result)

        # cv2.imwrite('result.jpg', drawimg)
        winName = 'Deep learning eye_gaze_estimation use OpenCV'
        cv2.namedWindow(winName, 0)
        cv2.imshow(winName, drawimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(args.videopath)
        cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
        video_writer = cv2.VideoWriter(
            filename='output.mp4',
            fourcc=fourcc,
            fps=cap_fps,
            frameSize=(cap_width, cap_height),
        )

        while True:
            # Capture read
            ret, frame = cap.read()
            if not ret:
                break

            boxes, scores, classids, kpts = face_detector.detect(frame)
            drawimg = frame.copy()
            for i, box in enumerate(boxes):
                x, y, w, h = box.astype(int)
                facebox = [x, y, x+w, y+h, kpts[i, 3], kpts[i, 4], kpts[i, 0], kpts[i, 1]]
                result = gaze_predictor.detect(facebox, frame)
                drawimg = gaze_predictor.draw_on(drawimg, result)

            video_writer.write(drawimg)

            # cv2.imshow('Deep learning eye_gaze_estimation use OpenCV', drawimg)
            # key = cv2.waitKey(1)
            # if key == 27:  # ESC
            #     break

        if video_writer:
            video_writer.release()
        if cap:
            cap.release()
        cv2.destroyAllWindows()