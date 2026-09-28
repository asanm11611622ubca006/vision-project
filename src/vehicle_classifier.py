"""
Vehicle Type Classifier & Detector
===================================
Uses a lightweight MobileNetV3 SSD detector to identify vehicle types
(Car, Motorcycle, Bus, Truck, Bicycle) in video frames and associate
each detected number plate with its corresponding vehicle.
"""
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)

# COCO dataset class indices for vehicles
VEHICLE_CLASSES: Dict[int, str] = {
    3: "Car",
    4: "Bike",
    6: "Bus",
    8: "Truck",
    2: "Bicycle",
}


class VehicleClassifier:
    """Detects vehicles in frames and associates plates with their parent vehicle."""

    def __init__(self, confidence_threshold: float = 0.30):
        self.conf_thresh = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load SSDLite320 with pre-trained weights
        self.model = ssdlite320_mobilenet_v3_large(
            weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.model.to(self.device)
        self.model.eval()

    def detect_vehicles(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, str, float]]:
        """
        Detect all vehicles in a given BGR frame.

        Returns
        -------
        List of tuples: (x1, y1, x2, y2, vehicle_type, score)
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        if w > 640:
            scale = 640.0 / w
            proc_frame = cv2.resize(frame, (640, int(h * scale)))
            inv_scale = 1.0 / scale
        else:
            proc_frame = frame
            inv_scale = 1.0

        # Convert BGR to RGB and scale to float32 tensor
        rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            preds = self.model([tensor])[0]

        boxes = preds["boxes"].detach().cpu().numpy()
        scores = preds["scores"].detach().cpu().numpy()
        labels = preds["labels"].detach().cpu().numpy()

        results = []
        for box, score, label_id in zip(boxes, scores, labels):
            if score < self.conf_thresh:
                continue
            if int(label_id) in VEHICLE_CLASSES:
                x1, y1, x2, y2 = [int(v * inv_scale) for v in box]
                # Clamp within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                v_type = VEHICLE_CLASSES[int(label_id)]
                results.append((x1, y1, x2, y2, v_type, float(score)))

        return results

    def classify_plate_vehicle(
        self,
        frame: np.ndarray,
        plate_box: Tuple[int, int, int, int],
        detected_vehicles: Optional[List[Tuple[int, int, int, int, str, float]]] = None,
    ) -> str:
        """
        Determine the vehicle type corresponding to a detected plate.

        Parameters
        ----------
        frame : np.ndarray
            The full video frame.
        plate_box : tuple
            (px, py, pw, ph) of the plate bounding box.
        detected_vehicles : list, optional
            Pre-computed list of detected vehicles in this frame.

        Returns
        -------
        str: "Car", "Motorcycle", "Bus", "Truck", etc. Default is "Car".
        """
        if detected_vehicles is None:
            detected_vehicles = self.detect_vehicles(frame)

        if not detected_vehicles:
            return "Car"

        px, py, pw, ph = plate_box
        pcx = px + pw / 2.0
        pcy = py + ph / 2.0

        # 1. Enclosing vehicle box check: plate center is inside the vehicle bbox
        enclosing = []
        for vx1, vy1, vx2, vy2, v_type, score in detected_vehicles:
            if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                area = (vx2 - vx1) * (vy2 - vy1)
                enclosing.append((area, score, v_type))

        if enclosing:
            # Pick enclosing vehicle with highest score / most specific box
            enclosing.sort(key=lambda item: (-item[1], item[0]))
            return enclosing[0][2]

        # 2. Proximity check: vehicle closest to the plate
        best_dist = float("inf")
        best_type = "Car"
        for vx1, vy1, vx2, vy2, v_type, score in detected_vehicles:
            vcx = (vx1 + vx2) / 2.0
            vcy = (vy1 + vy2) / 2.0
            dist = np.hypot(pcx - vcx, pcy - vcy)
            max_dim = max(vx2 - vx1, vy2 - vy1, 50)
            if dist < max_dim * 1.2 and dist < best_dist:
                best_dist = dist
                best_type = v_type

        return best_type
