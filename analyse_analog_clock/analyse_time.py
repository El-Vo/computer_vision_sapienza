import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ClockHand:
    angle_deg: float
    length: float
    kind: str = "unknown"


def detect_clock_face(gray: np.ndarray) -> Tuple[int, int, int]:
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=gray.shape[0] // 2,
        param1=150,
        param2=60,
        minRadius=gray.shape[0] // 4,
        maxRadius=gray.shape[0] // 2,
    )
    if circles is not None:
        circle = circles[0][0]
        return int(circle[0]), int(circle[1]), int(circle[2])
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Could not locate clock face")
    contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(contour)
    return int(x), int(y), int(radius)


def extract_clock_roi(
    image: np.ndarray, cx: int, cy: int, radius: int
) -> Tuple[np.ndarray, Tuple[float, float]]:
    pad = int(radius * 1.1)
    x0 = max(cx - pad, 0)
    y0 = max(cy - pad, 0)
    x1 = min(cx + pad, image.shape[1])
    y1 = min(cy + pad, image.shape[0])
    roi = image[y0:y1, x0:x1].copy()
    mask = np.zeros(roi.shape[:2], dtype="uint8")
    local_center = (cx - x0, cy - y0)
    cv2.circle(mask, (int(local_center[0]), int(local_center[1])), radius, 255, -1)
    masked = cv2.bitwise_and(roi, roi, mask=mask)
    return masked, local_center


def vector_to_angle(center: Tuple[float, float], point: Tuple[float, float]) -> float:
    dx = point[0] - center[0]
    dy = center[1] - point[1]
    angle = math.degrees(math.atan2(dx, dy))
    return angle % 360


def dedupe_hands(hands: List[ClockHand]) -> List[ClockHand]:
    kept: List[ClockHand] = []
    for hand in sorted(hands, key=lambda h: h.length, reverse=True):
        if all(min(abs(hand.angle_deg - other.angle_deg), 360 - abs(hand.angle_deg - other.angle_deg)) > 5 for other in kept):
            kept.append(hand)
    return kept


def detect_hands(
    face: np.ndarray, center: Tuple[float, float], radius: int, debug: bool = False
) -> List[ClockHand]:
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    radius = int(radius * 0.98)
    filtered = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(filtered, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype="uint8"), iterations=1)
    mask = np.zeros_like(edges)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius * 0.95), 255, -1)
    edges = cv2.bitwise_and(edges, edges, mask=mask)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(radius * 0.6),
        minLineLength=int(radius * 0.5),
        maxLineGap=int(radius * 0.1),
    )
    if lines is None:
        raise ValueError("Could not detect clock hands")
    hands: List[ClockHand] = []
    overlay = face.copy()
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        dist1 = math.hypot(x1 - center[0], y1 - center[1])
        dist2 = math.hypot(x2 - center[0], y2 - center[1])
        near_dist = min(dist1, dist2)
        far_point = (x1, y1) if dist1 > dist2 else (x2, y2)
        length = max(dist1, dist2)
        if near_dist > radius * 0.7:
            continue
        if length < radius * 0.15:
            continue
        angle = vector_to_angle(center, far_point)
        ratio = length / radius
        hands.append(ClockHand(angle, length, "unknown"))
        if debug:
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
    hands = dedupe_hands(hands)
    if debug:
        cv2.imshow("Detected Hands", overlay)
        cv2.waitKey(0)
        cv2.destroyWindow("Detected Hands")
    if not hands:
        raise ValueError("No hands survived filtering")
    return hands


def estimate_time(hands: List[ClockHand]) -> Tuple[int, int, Optional[int]]:
    if len(hands) < 2:
        raise ValueError("Need at least two hands to estimate time")
    hour_hand = min(hands, key=lambda h: h.length)
    remaining = [hand for hand in hands if hand is not hour_hand]
    minute_hand = max(remaining, key=lambda h: h.length)
    remaining = [hand for hand in remaining if hand is not minute_hand]
    second_hand = max(remaining, key=lambda h: h.length) if remaining else None
    if minute_hand is None or hour_hand is None:
        raise ValueError("Could not reliably determine minute and hour hands")
    minute_est = int(round(minute_hand.angle_deg / 6.0))
    minute = minute_est % 60
    carry = minute_est // 60
    hour_float = (hour_hand.angle_deg / 30.0) % 12
    hour = (int(math.floor(hour_float)) + carry) % 12
    hour = 12 if hour == 0 else hour
    second = None
    if second_hand is not None:
        second = int(round(second_hand.angle_deg / 6.0)) % 60
    return hour, minute, second


def analyse_clock(image_path: str, debug: bool = False) -> Tuple[int, int, Optional[int]]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cx, cy, radius = detect_clock_face(gray)
    face, local_center = extract_clock_roi(image, cx, cy, radius)
    hands = detect_hands(face, local_center, radius, debug=debug)
    return estimate_time(hands)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="path to the clock image")
    parser.add_argument("--debug", action="store_true", help="show intermediate steps")
    args = parser.parse_args()
    hour, minute, second = analyse_clock(args.image, debug=args.debug)
    if second is None:
        print(f"Detected time: {hour:02d}:{minute:02d}")
    else:
        print(f"Detected time: {hour:02d}:{minute:02d}:{second:02d}")


if __name__ == "__main__":
    main()
