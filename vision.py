# vision.py
from __future__ import annotations
import cv2
from typing import List, Tuple

class Vision:
    def __init__(self, camera_index: int = 0, top_k: int = 3, conf: float = 0.35):
        self.camera_index = camera_index
        self.top_k = top_k
        self.conf = conf
        try:
            from ultralytics import YOLO  # lazy import
            self.YOLO = YOLO
            self.model = YOLO("yolov8n.pt")
        except Exception as e:
            self.YOLO = None
            self.model = None
            print("[Vision] YOLO not available:", e)

    def describe_scene(self) -> str:
        if self.model is None:
            return "Vision is not enabled yet."

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            return "I cannot access the camera."

        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "I couldn't read from the camera."

        results = self.model.predict(frame, conf=self.conf, verbose=False)
        names = self.model.names
        counts = {}
        for r in results:
            for c in r.boxes.cls.tolist():
                label = names[int(c)]
                counts[label] = counts.get(label, 0) + 1

        if not counts:
            return "I don't see anything notable."

        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[: self.top_k]
        phrases = [f"{n} {lbl}" for lbl, n in top]
        return "I see " + ", ".join(phrases) + "."
