import cv2
import numpy as np

class SkinKLTTracker:
    def __init__(self):
        self.initialized = False
        self.points = None
        self.old_gray = None
        self.box = None
        self.box_history = []
        self.smooth_window = 5

    def select_roi(self, frame, box = None):
        x, y, w, h = [int(v) for v in box]
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        points = cv2.goodFeaturesToTrack(roi_gray, maxCorners=50, qualityLevel=0.01, minDistance=5, blockSize=5, useHarrisDetector=False)
        if points is not None:
            points[:,0,0] += x
            points[:,0,1] += y
        self.points = points
        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.initialized = True
        self.box_history = [np.array([x, y, w, h])]

    # def track(self, frame):
    #     if not self.initialized or self.points is None:
    #         return None, None
    #     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     new_points, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.points, None)
    #     good_new = new_points[st==1]
    #     if len(good_new) < 2:
    #         self.initialized = False
    #         return None, None
    #     x_min, y_min = np.min(good_new, axis=0)
    #     x_max, y_max = np.max(good_new, axis=0)
    #     width = x_max - x_min
    #     height = y_max - y_min
    #     tracked_box = np.array([x_min, y_min, width, height])
    #     self.box_history.append(tracked_box)
    #     if len(self.box_history) > self.smooth_window:
    #         self.box_history.pop(0)
    #     smooth_box = np.mean(self.box_history, axis=0)
    #     self.points = good_new.reshape(-1,1,2)
    #     self.old_gray = frame_gray.copy()
    #     return smooth_box, self.points
    def track(self, frame):
        if not self.initialized or self.points is None:
            return None, None
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 初始化变量
        good_new = None
        
        try:
            # 第一轮前向跟踪
            new_points, st, err = cv2.calcOpticalFlowPyrLK(
                self.old_gray, frame_gray, self.points, None, 
                winSize=(15, 15), maxLevel=3, 
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            
            # 检查跟踪结果
            if new_points is None or st is None:
                self.initialized = False
                return None, None
            
            # 第二轮反向跟踪
            back_points, st_back, err_back = cv2.calcOpticalFlowPyrLK(
                frame_gray, self.old_gray, new_points, None,
                winSize=(15, 15), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            
            if back_points is None or st_back is None:
                self.initialized = False
                return None, None
            
            # 计算双向误差
            points_flat = self.points.reshape(-1, 2)
            back_points_flat = back_points.reshape(-1, 2)
            forward_backward_error = np.linalg.norm(points_flat - back_points_flat, axis=1)
            
            # 安全的筛选条件
            # good_indices = np.where(
            #     (st.flatten() == 1) & 
            #     (st_back.flatten() == 1) & 
            #     (forward_backward_error < 2.0) & 
            #     (err.flatten() < 15.0)
            # )[0]
            good_indices = np.where(
            (st.flatten() == 1) & 
            (st_back.flatten() == 1) & 
            (forward_backward_error < 2.0))[0]  # 只保留双向误差检查，与Matlab的MaxBidirectionalError逻辑一致
                    # 检查有效点数量
            if len(good_indices) < 2:
                self.initialized = False
                return None, None
            
            # 赋值good_new
            good_new = new_points[good_indices]
            
        except Exception as e:
            print(f"跟踪过程中发生错误: {e}")
            self.initialized = False
            return None, None
        
        # 现在可以安全地使用good_new
        if good_new is None:
            self.initialized = False
            return None, None
        
        # 确保正确的形状
        if len(good_new.shape) == 3:
            good_new = good_new.reshape(-1, 2)
        
        # 安全的边界计算
        x_coords = good_new[:, 0]
        y_coords = good_new[:, 1]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # 边界检查
        img_h, img_w = frame.shape[:2]
        # 1. 计算当前帧的原始边界框（与Matlab一致）
        x_min, y_min = np.min(good_new, axis=0)
        x_max, y_max = np.max(good_new, axis=0)
        width = x_max - x_min
        height = y_max - y_min
        tracked_box = np.array([x_min, y_min, width, height])

        # 2. 将原始框加入历史并进行平滑处理（与Matlab一致）
        self.box_history.append(tracked_box)
        if len(self.box_history) > self.smooth_window:
            self.box_history.pop(0)
        smooth_box = np.mean(self.box_history, axis=0)

        # 3. 对平滑后的框进行边界约束（与Matlab一致）
        img_h, img_w = frame.shape[:2]
        x1 = round(smooth_box[0])
        y1 = round(smooth_box[1])
        x2 = round(smooth_box[0] + smooth_box[2])
        y2 = round(smooth_box[1] + smooth_box[3])

        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y1 + (y2 - y1)) # 防止高度为负

        # 根据约束后的坐标重新计算最终返回的框
        final_width = x2 - x1
        final_height = y2 - y1

        # 【修改】尺寸检查移到最后，检查的是最终框的尺寸
        min_size = 10
        if final_width < min_size or final_height < min_size:
            print("警告：平滑后的跟踪框尺寸过小，跳过此帧")
            return None, None

        # 【修改】返回的是经过边界约束的平滑框
        final_tracked_box = np.array([x1, y1, final_width, final_height])

        # 4. 为下次跟踪准备点
        self.points = good_new.reshape(-1, 1, 2)
        self.old_gray = frame_gray.copy()

        # 5. 返回最终框
        return final_tracked_box, self.points
        # x_min = max(0, x_min)
        # y_min = max(0, y_min)
        # x_max = min(img_w, x_max)
        # y_max = min(img_h, y_max)
        
        # width = x_max - x_min
        # height = y_max - y_min
        
        # # 尺寸检查
        # min_size = 10
        # max_width_ratio = 0.5
        # max_height_ratio = 0.5
        # if width < min_size or height < min_size or \
        # width > img_w * max_width_ratio or height > img_h * max_height_ratio:
        #     print("警告：跟踪框尺寸异常，跳过此帧")
        #     return None, None
        
        # tracked_box = np.array([x_min, y_min, width, height])
        
        # # 平滑处理
        # self.box_history.append(tracked_box)
        # if len(self.box_history) > self.smooth_window:
        #     self.box_history.pop(0)
        # smooth_box = np.mean(self.box_history, axis=0)
        
        # # 为下次跟踪准备点
        # self.points = good_new.reshape(-1, 1, 2)
        # self.old_gray = frame_gray.copy()
        
        # return smooth_box, self.points


    def get_roi_and_mask(self, frame, smooth_box):
        x1 = int(max(0, round(smooth_box[0])))
        y1 = int(max(0, round(smooth_box[1])))
        x2 = int(min(frame.shape[1], round(smooth_box[0] + smooth_box[2])))
        y2 = int(min(frame.shape[0], round(smooth_box[1] + smooth_box[3])))
        roi = frame[y1:y2, x1:x2]
        skin_mask = skin_detection(roi)
        # 形态学处理
        # roi_size = max(roi.shape[:2])
        # small_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1,round(0.01*roi_size)),)*2)
        # medium_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1,round(0.02*roi_size)),)*2)
        # large_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1,round(0.03*roi_size)),)*2)
        # skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, small_se)
        # skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, medium_se)
        # skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, large_se)
        # skin_mask = cv2.dilate(skin_mask, small_se, iterations=1)
        # 填充孔洞
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        # 合成到全图mask
        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = skin_mask
        return full_mask

def skin_detection(rgb_image):
    # 输入BGR，转为RGB
    rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ycbcr = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    R = rgb[:,:,0]
    G = rgb[:,:,1]
    B = rgb[:,:,2]
    H = hsv[:,:,0] * 2  # OpenCV H范围0-180
    S = hsv[:,:,1] / 255.0
    Y = ycbcr[:,:,0]
    Cb = ycbcr[:,:,2]
    Cr = ycbcr[:,:,1]
    rule1 = (H >= 0) & (H <= 50) & (S >= 0.23) & (S <= 0.68)
    rule2 = (R > 95) & (G > 40) & (B > 20) & (R > G) & (R > B) & (np.abs(R-G) > 15)
    rule3 = (Cr > 135) & (Cb > 85) & (Y > 80) & \
            (Cr <= (1.5862*Cb)+20) & \
            (Cr >= (0.3448*Cb)+76.2069) & \
            (Cr >= (-4.5652*Cb + 234.5652)) & \
            (Cr <= (-1.15 * Cb) + 301.75) & \
            (Cr <= (-2.2857 * Cb) + 432.85)
    skin_mask = ((rule1 & rule2) | (rule2 & rule3)).astype(np.uint8) * 255
    return skin_mask

def compute_region_rgb_means(image, mask):
    skin_pixels = image[mask==255]
    nonskin_pixels = image[mask==0]
    skin_mean = np.mean(skin_pixels, axis=0) if len(skin_pixels) > 0 else [0,0,0]
    nonskin_mean = np.mean(nonskin_pixels, axis=0) if len(nonskin_pixels) > 0 else [0,0,0]
    return skin_mean