#!/usr/bin/env python3

import cv2
import sys
import os
import urllib.request
import numpy as np
from datetime import datetime

class FaceAnonymizer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.output_dir = "output_videos"
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self._setup_haar_detector()
        self._setup_dnn_detector()
        
        self.confidence_threshold = 0.3
        self.nms_threshold = 0.4
        
    def _setup_haar_detector(self):
        self.haar_cascades = []
        cascade_file = 'haarcascade_frontalface_default.xml'
        cascade_path = cv2.data.haarcascades + cascade_file
        if os.path.exists(cascade_path):
            self.haar_cascades.append(cv2.CascadeClassifier(cascade_path))
    
    def _setup_dnn_detector(self):
        try:
            prototxt_path = "deploy.prototxt"
            model_path = "res10_300x300_ssd_iter_140000.caffemodel"
            
            if not os.path.exists(prototxt_path):
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                    prototxt_path
                )
            
            if not os.path.exists(model_path):
                urllib.request.urlretrieve(
                    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                    model_path
                )
            
            self.dnn_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            
            # Enable GPU acceleration
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("GPU acceleration enabled for DNN")
            else:
                print("GPU not available, using CPU")
                
        except:
            self.dnn_net = None
    
    def preprocess_frame(self, frame):
        # Only return original frame to speed up processing
        return [frame]
    
    def _adjust_gamma(self, image, gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def detect_faces_haar(self, frame):
        if not self.haar_cascades:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use only the default cascade with optimized parameters
        cascade = self.haar_cascades[0]  # Use only the first (default) cascade
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            maxSize=(200, 200),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces_with_conf = [(x, y, w, h, 0.8) for (x, y, w, h) in faces]
        return self._apply_nms(faces_with_conf)
    
    def detect_faces_dnn(self, frame):
        if self.dnn_net is None:
            return []
        
        all_faces = []
        h, w = frame.shape[:2]
        
        # Use GPU memory efficiently with single larger input size
        input_size = (416, 416)
        
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, input_size),
            1.0,
            input_size,
            (104.0, 117.0, 123.0)
        )
        
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                all_faces.append((x, y, x2-x, y2-y, confidence))
        
        return self._apply_nms(all_faces)
    
    def _apply_nms(self, faces):
        if len(faces) == 0:
            return []
        
        boxes = []
        scores = []
        
        for face in faces:
            if len(face) == 5:
                x, y, w, h, conf = face
            else:
                x, y, w, h = face
                conf = 0.8
            
            boxes.append([x, y, x + w, y + h])
            scores.append(conf)
        
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.nms_threshold)
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [(boxes[i][0], boxes[i][1], 
                    boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]) 
                   for i in indices]
        
        return []
    
    def blur_faces(self, frame, faces):
        for face in faces:
            if len(face) == 4:
                x, y, w, h = face
            else:
                x, y, w, h = face[:4]
            
            padding = max(5, min(w, h) // 10)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            face_region = frame[y:y+h, x:x+w]
            
            if face_region.size > 0:
                kernel_size = max(15, min(w, h) // 3)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                blurred_face = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 0)
                frame[y:y+h, x:x+w] = blurred_face
        
        return frame
    
    def create_output_filename(self, method_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        return os.path.join(self.output_dir, f"{base_name}_{method_name}_{timestamp}.mp4")
    
    def process_with_method(self, method_name, detect_function):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_path = self.create_output_filename(method_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            return None
        
        frame_number = 0
        total_faces = 0
        
        print(f"Processing {method_name}: {frame_count} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            faces = []
            processed_frames = self.preprocess_frame(frame)
            
            for processed_frame in processed_frames:
                method_faces = detect_function(processed_frame)
                faces.extend(method_faces)
            
            if faces:
                faces = self._apply_nms(faces)
            
            total_faces += len(faces)
            frame = self.blur_faces(frame, faces)
            out.write(frame)
            
            if frame_number % 30 == 0:
                progress = (frame_number / frame_count) * 100
                avg_faces = total_faces / frame_number
                print(f"{method_name}: Frame {frame_number}/{frame_count} ({progress:.1f}%) - Avg faces/frame: {avg_faces:.2f}")
        
        cap.release()
        out.release()
        print(f"{method_name} complete: {total_faces} total faces detected")
        return output_path
    
    def process_all_methods(self):
        results = {}
        
        if self.dnn_net is not None:
            output_path = self.process_with_method("DNN", self.detect_faces_dnn)
            if output_path:
                results["DNN"] = output_path
        
        if self.haar_cascades:
            output_path = self.process_with_method("Haar", self.detect_faces_haar)
            if output_path:
                results["Haar"] = output_path
        
        return results

def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "video.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        sys.exit(1)
    
    anonymizer = FaceAnonymizer(video_path)
    results = anonymizer.process_all_methods()

if __name__ == "__main__":
    main()