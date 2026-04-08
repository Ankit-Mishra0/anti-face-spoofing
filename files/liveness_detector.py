import cv2
import numpy as np
from collections import deque
import os

class BlinkDetector:
    
    def __init__(self):
        self.detector = None
        self.predictor = None
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            if os.path.exists("shape_predictor_68_face_landmarks.dat"):
                self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        except:
            pass
        
        self.blink_threshold = 0.3
        self.blink_history = deque(maxlen=30)
        self.blink_count = 0
    
    def get_eye_aspect_ratio(self, eye_points):
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_blink(self, frame):
        if self.detector is None or self.predictor is None:
            return False, 0, 0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            return False, self.blink_count, 0
        
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        
        left_ear = self.get_eye_aspect_ratio(left_eye)
        right_ear = self.get_eye_aspect_ratio(right_eye)
        
        avg_ear = (left_ear + right_ear) / 2.0
        
        is_blinking = avg_ear < self.blink_threshold
        self.blink_history.append(is_blinking)
        
        if len(self.blink_history) > 1:
            if self.blink_history[-2] and not self.blink_history[-1]:
                self.blink_count += 1
        
        return is_blinking, self.blink_count, avg_ear
    
    def get_blink_stats(self):
        if len(self.blink_history) == 0:
            return 0, 0
        
        blink_frequency = sum(self.blink_history) / len(self.blink_history)
        return self.blink_count, blink_frequency


class MotionDetector:
    
    def __init__(self, history_size=5):
        self.face_history = deque(maxlen=history_size)
        self.motion_threshold = 5 
    
    def detect_motion(self, frame, face_bbox):
        x, y, w, h = face_bbox
        face_center = (x + w//2, y + h//2)
        
        self.face_history.append(face_center)
        
        if len(self.face_history) < 2:
            return False, 0
        

        prev_center = self.face_history[-2]
        curr_center = self.face_history[-1]
        
        motion = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                        (curr_center[1] - prev_center[1])**2)
        
        motion_detected = motion > self.motion_threshold
        return motion_detected, motion
    
    def get_motion_stats(self):
        if len(self.face_history) < 2:
            return 0, 0
        
        motions = []
        for i in range(1, len(self.face_history)):
            prev = self.face_history[i-1]
            curr = self.face_history[i]
            motion = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
            motions.append(motion)
        
        avg_motion = np.mean(motions) if motions else 0
        max_motion = np.max(motions) if motions else 0
        
        return avg_motion, max_motion


class TextureAnalyzer:
    
    @staticmethod
    def analyze_texture(face_image):
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor((face_image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_score = np.var(laplacian)
        
        return texture_score
    
    @staticmethod
    def detect_screen_reflection(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        saturation = hsv[:, :, 1]
        s_std = np.std(saturation)
        
        value = hsv[:, :, 2]
        v_mean = np.mean(value)
        
        reflection_score = (100 - s_std) + (v_mean - 127) * 0.5
        
        return reflection_score


class LivenessAnalyzer:
    
    def __init__(self):
        self.blink_detector = BlinkDetector()
        self.motion_detector = MotionDetector()
        self.texture_analyzer = TextureAnalyzer()
        
        self.blink_threshold = 0.05
        self.motion_threshold = 100
        self.texture_threshold = 30
        self.reflection_threshold = 150
        
        self.weights = {
            'blink': 0.15,
            'motion': 0.15,
            'texture': 0.20,
            'model': 0.50
        }
    
    def analyze_frame(self, frame, face_bbox, model_confidence):
        details = {}
        
        x, y, w, h = face_bbox
        face = frame[y:y+h, x:x+w]
        
        is_blinking, blink_count, ear = self.blink_detector.detect_blink(frame)
        blink_stats = self.blink_detector.get_blink_stats()
        blink_score = min(blink_stats[1] * 100, 1.0)
        details['blink_count'] = blink_count
        details['blink_frequency'] = blink_stats[1]
        
        motion_detected, motion_mag = self.motion_detector.detect_motion(frame, face_bbox)
        motion_stats = self.motion_detector.get_motion_stats()
        motion_score = min(motion_stats[0] / self.motion_threshold, 1.0)
        details['avg_motion'] = motion_stats[0]
        details['max_motion'] = motion_stats[1]
        
        texture_score_raw = self.texture_analyzer.analyze_texture(face)
        texture_score = min(texture_score_raw / self.texture_threshold, 1.0)
        reflection_score = self.texture_analyzer.detect_screen_reflection(face)
        details['texture_variance'] = texture_score_raw
        details['reflection_score'] = reflection_score
        
        model_score = model_confidence
        details['model_confidence'] = model_score
        
        # Combine scores
        weighted_score = (
            self.weights['blink'] * blink_score +
            self.weights['motion'] * motion_score +
            self.weights['texture'] * texture_score +
            self.weights['model'] * model_score
        )
        
        details['scores'] = {
            'blink': blink_score,
            'motion': motion_score,
            'texture': texture_score,
            'model': model_score
        }
        
        return weighted_score, details
    
    def reset(self):
        self.blink_detector.blink_history.clear()
        self.blink_detector.blink_count = 0
        self.motion_detector.face_history.clear()
