"""
Real-time Hand VFX System with Neon Effects
Webcam application with glowing neon effects around hand movements.

Dependencies: OpenCV, MediaPipe, NumPy
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration class for all visual effects parameters"""

    # Visual Effects
    glow_intensity: int = 6          # Number of glow layers (kept small to avoid fat bars)
    line_thickness: int = 2
    smoothing_factor: float = 0.3

    # Performance
    max_hands: int = 2
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5

    # Effects toggles
    enable_particles: bool = False
    enable_trails: bool = True
    enable_aura: bool = True
    enable_fingertip_pulse: bool = True

    # Cinematic VFX
    cinematic_darkening: float = 0.25
    bloom_intensity: float = 0.3
    pulse_speed: float = 3.0

    # Magic Shield Effect
    enable_shield: bool = True
    shield_radius: int = 120
    shield_rotation_speed: float = 2.0
    shield_glow_intensity: float = 0.6
    shield_color: Tuple[int, int, int] = (0, 140, 255)  # BGR orange

    # Skeleton glow falloff (lower = thinner/softer outer glow)
    skeleton_glow_alpha: float = 0.35

    # Aura enable/intensity
    aura_alpha: float = 0.08

    # Gesture-triggered superhero VFX
    enable_gesture_vfx: bool = True
    gesture_fade_speed: float = 6.0

    # Repulsor Beam (point)
    repulsor_beam_length: int = 700
    repulsor_pulse_speed: float = 2.2

    # Energy Blast (pinch)
    blast_max_radius: int = 70
    blast_charge_speed: float = 1.5

    # Power Charge (fist)
    power_charge_pulse_speed: float = 8.0

    # Laser / Scan (two fingers)
    laser_scan_speed: float = 2.0

    # Power Activated (thumbs up)
    flash_duration: float = 0.35

    # Two-hand combo gestures: Portal vs Energy Ball
    portal_min_hand_distance: float = 220.0
    energy_ball_max_radius: int = 90

    # Energy Trail (moving hand)
    trail_speed_threshold: float = 250.0
    trail_max_points: int = 18

    # Star Nova (rock-on gesture)
    nova_burst_interval: float = 0.45

    # Colors (BGR format for OpenCV)
    colors: dict = field(default_factory=lambda: {
        'primary':   (0, 255, 255),      # Cyan
        'secondary': (255, 0, 255),      # Magenta
        'accent':    (255, 255, 0),      # Yellow
        'energy':    (0, 255, 0),        # Green
        'purple':    (128, 0, 255),      # Purple
        'pink':      (255, 105, 180),    # Pink

        # Beam core colors
        'beam_core':      (255, 255, 255),
        'beam_highlight': (200, 220, 255),

        # Per-finger neon colors (BGR)
        'thumb_line':  (20, 80, 255),    # Orange-red
        'index_line':  (255, 255, 0),    # Cyan
        'middle_line': (220, 0, 255),    # Magenta
        'ring_line':   (255, 60, 140),   # Purple-blue
        'pinky_line':  (0, 220, 255),    # Yellow-green

        # Superhero gesture VFX colors (BGR)
        'repulsor_beam':      (255, 210, 110),  # pale energy blue
        'energy_blast':       (0, 140, 255),    # orange-red blast
        'power_charge_core':  (0, 180, 255),    # golden core
        'power_charge_outer': (0, 70, 255),     # red-orange outer glow
        'laser_scan':         (0, 0, 255),      # red scan beam
        'power_activated':    (0, 255, 255),    # cyan-yellow particles
        'portal_a':           (255, 180, 40),   # teal-blue
        'portal_b':           (255, 60, 200),   # purple-magenta
        'energy_ball_core':   (255, 150, 30),   # deep blue core
        'energy_ball_outer':  (255, 220, 150),  # light blue glow
        'trail_color':        (255, 240, 200),  # soft white-blue trail

        # Star Nova colors (BGR)
        'nova_core':          (180, 235, 255),  # bright warm gold-white star core
        'nova_outer':         (90, 190, 255),   # golden amber outer glow
    })

# ============================================================================
# HAND TRACKING
# ============================================================================

class HandTracker:
    """Handles hand detection using MediaPipe"""

    def __init__(self, config: Config):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.max_hands,
            min_detection_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence,
            model_complexity=1
        )

        # Hand skeleton connections
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (0, 17)                                 # Palm base
        ]

    def detect_hands(self, frame: np.ndarray) -> Optional[List[List[Tuple[int, int]]]]:
        """Detect hands and return pixel landmark positions"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None

        h, w = frame.shape[:2]
        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks.append((x, y))
            all_landmarks.append(landmarks)

        return all_landmarks

# ============================================================================
# GESTURE RECOGNITION
# ============================================================================

class GestureRecognizer:
    """
    Classifies simple static hand poses from MediaPipe landmarks.
    Detection is rotation-invariant (distance-from-wrist based), matching
    the technique already used for the open-palm shield check.
    """

    # Single-hand gesture -> effect labels
    GESTURE_LABELS = {
        'open_palm':   '🛡️ Energy Shield',
        'point':       '⚡ Repulsor Beam',
        'pinch':       '💥 Energy Blast',
        'fist':        '🔥 Power Charge',
        'peace':       '🔴 Laser Scan',
        'thumbs_up':   '⚡ Power Activated',
        'thumbs_down': '🚫 Power Disabled',
        'rock_on':     '⭐ Star Nova',
    }

    # Two-hand combo gesture labels (open palm + open palm), resolved by
    # GestureEffectManager based on how far apart the two hands are.
    COMBO_LABELS = {
        'portal':      '🌀 Portal Effect',
        'energy_ball': '🔵 Energy Ball',
    }

    @staticmethod
    def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @classmethod
    def classify(cls, landmarks: List[Tuple[int, int]]) -> Optional[str]:
        """Returns a gesture key (see GESTURE_LABELS) or None if no match."""
        if landmarks is None or len(landmarks) < 21:
            return None

        wrist = landmarks[0]
        dist = cls._dist

        index_ext  = dist(landmarks[8],  wrist) > dist(landmarks[6],  wrist) * 1.05
        middle_ext = dist(landmarks[12], wrist) > dist(landmarks[10], wrist) * 1.05
        ring_ext   = dist(landmarks[16], wrist) > dist(landmarks[14], wrist) * 1.05
        pinky_ext  = dist(landmarks[20], wrist) > dist(landmarks[18], wrist) * 1.05
        thumb_ext  = dist(landmarks[4],  wrist) > dist(landmarks[2],  wrist) * 1.2

        hand_size = max(1.0, dist(landmarks[0], landmarks[9]))
        pinch_dist = dist(landmarks[4], landmarks[8])
        is_pinch = pinch_dist < hand_size * 0.4
        fingers_curled = not index_ext and not middle_ext and not ring_ext and not pinky_ext

        # Thumb extended, everything else curled, thumb pointing clearly
        # up or down relative to the wrist → Power Activated / Disabled
        if thumb_ext and fingers_curled and not is_pinch:
            vertical_offset = wrist[1] - landmarks[4][1]  # >0 if thumb is above the wrist
            if abs(vertical_offset) > hand_size * 0.35:
                return 'thumbs_up' if vertical_offset > 0 else 'thumbs_down'

        # Thumb + index pinched together → Energy Blast
        if is_pinch and not middle_ext and not ring_ext:
            return 'pinch'
        # Index + pinky extended, middle & ring curled (rock-on sign) → Star Nova
        if index_ext and pinky_ext and not middle_ext and not ring_ext:
            return 'rock_on'
        # Everything curled including the thumb → Power Charge
        if fingers_curled and not thumb_ext:
            return 'fist'
        # Index + middle extended (peace sign) → Laser / Scan
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return 'peace'
        # Only index extended → Repulsor Beam
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return 'point'
        # All four fingers extended and spread → Energy Shield
        # (also the base pose for the two-hand Portal / Energy Ball combos)
        if index_ext and middle_ext and ring_ext and pinky_ext:
            return 'open_palm'

        return None


def _palm_center(landmarks: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Average of palm-base landmarks — used as a stable anchor point."""
    idxs = [0, 5, 9, 13, 17]
    x = int(sum(landmarks[i][0] for i in idxs) / len(idxs))
    y = int(sum(landmarks[i][1] for i in idxs) / len(idxs))
    return (x, y)


def glow_line(overlay: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
              color: Tuple[int, int, int], alpha: float = 1.0,
              base_thickness: int = 2, layers: int = 5):
    """Shared multi-pass neon line helper used by the gesture VFX effects."""
    for i in range(layers, 0, -1):
        thickness = base_thickness + (i - 1) * 2
        a = alpha * (0.9 if i == 1 else 0.12 * (1.0 - (i - 1) / layers))
        a = max(0.0, min(1.0, a))
        c = tuple(int(ch * a) for ch in color)
        cv2.line(overlay, pt1, pt2, c, thickness, cv2.LINE_AA)


def glow_dot(overlay: np.ndarray, center: Tuple[int, int], radius: int,
             color: Tuple[int, int, int], alpha: float = 1.0, layers: int = 4):
    """Shared multi-pass neon dot/orb helper used by the gesture VFX effects."""
    if radius <= 0 or alpha <= 0:
        return
    for i in range(layers, 0, -1):
        r = max(1, int(radius * (0.4 + 0.6 * i / layers)))
        a = alpha * (0.95 if i == 1 else (0.15 * (1.0 - (i - 1) / layers) + 0.05))
        a = max(0.0, min(1.0, a))
        c = tuple(int(ch * a) for ch in color)
        cv2.circle(overlay, center, r, c, -1, cv2.LINE_AA)


@dataclass
class GestureParticle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    size: float
    color: Tuple[int, int, int]

# ============================================================================
# NEON RENDERING
# ============================================================================

class NeonRenderer:
    """Handles neon skeleton and aura rendering"""

    def __init__(self, config: Config):
        self.config = config
        self.pulse_time = 0.0
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

    def update(self, dt: float):
        self.pulse_time += dt

    def draw_neon_line_on_overlay(self, overlay: np.ndarray,
                                   pt1: Tuple[int, int], pt2: Tuple[int, int],
                                   color: Tuple[int, int, int],
                                   base_thickness: int = 2):
        """
        Draw a neon glow line onto an overlay image.
        Uses only colored layers — no black, no dark outlines.
        Outer layers are dim, inner core is bright.
        """
        n = self.config.glow_intensity  # e.g. 6

        for i in range(n, 0, -1):
            # i=n is outermost, i=1 is innermost core
            thickness = base_thickness + (i - 1) * 2
            # Alpha: very faint outer, solid bright inner
            alpha = self.config.skeleton_glow_alpha * (1.0 - (i - 1) / n)
            alpha = max(0.0, min(1.0, alpha))
            if i == 1:
                alpha = 0.9  # Core line: nearly full brightness

            glow_color = tuple(int(c * alpha) for c in color)
            cv2.line(overlay, pt1, pt2, glow_color, thickness, cv2.LINE_AA)

    def draw_hand_skeleton(self, frame: np.ndarray,
                            landmarks: List[Tuple[int, int]],
                            color: Tuple[int, int, int]):
        """
        Draw hand skeleton using overlay blending so lines look glowing,
        not painted. Blends onto frame in-place.
        """
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        for c0, c1 in self.connections:
            if c0 < len(landmarks) and c1 < len(landmarks):
                pt1 = (int(landmarks[c0][0]), int(landmarks[c0][1]))
                pt2 = (int(landmarks[c1][0]), int(landmarks[c1][1]))
                self.draw_neon_line_on_overlay(overlay, pt1, pt2, color,
                                               self.config.line_thickness)

        # Blend skeleton overlay onto frame: additive-style blending
        # cv2.add clamps at 255 which gives the glow-on-dark look
        cv2.add(frame, overlay, dst=frame)

    def draw_hand_aura(self, frame: np.ndarray,
                       landmarks: List[Tuple[int, int]],
                       color: Tuple[int, int, int]):
        """
        Draw a soft pulsing aura ring around the palm center.
        Uses overlay blending — no dark blobs.
        """
        if not self.config.enable_aura or len(landmarks) < 21:
            return

        palm_indices = [0, 5, 9, 13, 17]
        palm_x = int(sum(landmarks[i][0] for i in palm_indices) / len(palm_indices))
        palm_y = int(sum(landmarks[i][1] for i in palm_indices) / len(palm_indices))

        pulse = math.sin(self.pulse_time * self.config.pulse_speed) * 0.15 + 1.0
        base_radius = int(35 * pulse)

        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw soft rings — only on overlay, blended additively
        num_rings = 5
        for i in range(num_rings):
            radius = base_radius + i * 7
            # Fades out toward outer rings
            alpha = self.config.aura_alpha * (1.0 - i / num_rings)
            ring_color = tuple(int(c * alpha) for c in color)
            cv2.circle(overlay, (palm_x, palm_y), radius, ring_color, 2, cv2.LINE_AA)

        cv2.add(frame, overlay, dst=frame)

# ============================================================================
# MAGIC SHIELD EFFECT
# ============================================================================

class MagicShieldEffect:
    """Dr Strange style magical circular shield effect"""

    def __init__(self, config: Config):
        self.config = config
        self.time = 0.0
        self.current_alpha = 0.0
        self.fade_speed = 8.0
        self.last_center = None

    def update(self, dt: float, landmarks_list):
        self.time += dt
        
        target_alpha = 0.0
        
        # Activate ONLY when exactly one hand is detected
        if landmarks_list and len(landmarks_list) == 1:
            landmarks = landmarks_list[0]
            
            # Simple open palm gesture check:
            # Check if fingertips are further from wrist than PIP joints
            wrist = (landmarks[0][0], landmarks[0][1])
            is_open = True
            for tip_idx, pip_idx in zip([8, 12, 16, 20], [6, 10, 14, 18]):
                tip = (landmarks[tip_idx][0], landmarks[tip_idx][1])
                pip = (landmarks[pip_idx][0], landmarks[pip_idx][1])
                dist_tip = math.hypot(tip[0] - wrist[0], tip[1] - wrist[1])
                dist_pip = math.hypot(pip[0] - wrist[0], pip[1] - wrist[1])
                if dist_tip < dist_pip:
                    is_open = False
                    break
                    
            if is_open:
                target_alpha = 1.0
                # Update last known center safely
                palm_indices = [0, 5, 9, 13, 17]
                cx = int(sum(landmarks[i][0] for i in palm_indices) / len(palm_indices))
                cy = int(sum(landmarks[i][1] for i in palm_indices) / len(palm_indices))
                self.last_center = (cx, cy)
                
        alpha_diff = target_alpha - self.current_alpha
        self.current_alpha += alpha_diff * dt * self.fade_speed
        self.current_alpha = max(0.0, min(1.0, self.current_alpha))

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if not self.config.enable_shield or self.current_alpha <= 0.01:
            return frame
        if self.last_center is None:
            return frame

        h, w = frame.shape[:2]
        shield_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Mild flicker on glow intensity
        flicker = 1.0 + math.sin(self.time * 30.0) * 0.05
        intensity = self.config.shield_glow_intensity * self.current_alpha * flicker
        
        # Subtle scale pulsing
        pulse = math.sin(self.time * 5.0) * 0.02 + 1.0
        base_radius = int(self.config.shield_radius * pulse)
        
        color = self.config.shield_color
        
        angle_outer = self.time * self.config.shield_rotation_speed
        angle_inner = -self.time * self.config.shield_rotation_speed * 1.5
        
        self._draw_magic_circle(shield_overlay, self.last_center, base_radius, angle_outer, angle_inner, color, intensity)
        
        cv2.add(frame, shield_overlay, dst=frame)
        return frame

    def _draw_magic_circle(self, overlay: np.ndarray, center, radius, angle_outer, angle_inner, color, intensity):
        """Draw circular magic shield elements onto overlay with neon glow technique"""
        # Multi-layer drawing for neon glow
        thick_passes = [
            (24, 0.05),
            (16, 0.15),
            (8,  0.40),
            (3,  0.80),
            (1,  1.0)
        ]
        
        cx, cy = center
        
        for thickness, alpha in thick_passes:
            layer_alpha = alpha * intensity
            if layer_alpha <= 0: continue
            
            # gradient color effect: core is closer to white/bright yellow
            if thickness <= 3:
                r_c = min(255, color[0] + 100)
                g_c = min(255, color[1] + 100)
                b_c = min(255, color[2] + 100)
                c = (r_c, g_c, b_c)
            else:
                c = color
                
            layer_color = tuple(int(ch * layer_alpha) for ch in c)
            
            # 1. Main Outer Ring
            cv2.circle(overlay, center, radius, layer_color, thickness, cv2.LINE_AA)
            cv2.circle(overlay, center, max(1, radius - 15), layer_color, max(1, thickness - 1), cv2.LINE_AA)
            
            # 2. Outer Octagon
            if radius > 15:
                pts_oct = []
                for i in range(8):
                    theta = angle_outer + i * (math.pi / 4)
                    x = int(cx + (radius - 15) * math.cos(theta))
                    y = int(cy + (radius - 15) * math.sin(theta))
                    pts_oct.append((x, y))
                for i in range(8):
                    cv2.line(overlay, pts_oct[i], pts_oct[(i + 1) % 8], layer_color, max(1, thickness-1), cv2.LINE_AA)
            
            # 3. Inner Rotating Squares (giving that layered look)
            inner_r = max(5, radius - 45)
            pts_sq = []
            for i in range(4):
                theta = angle_inner + i * (math.pi / 2)
                x = int(cx + inner_r * math.cos(theta))
                y = int(cy + inner_r * math.sin(theta))
                pts_sq.append((x, y))
            for i in range(4):
                cv2.line(overlay, pts_sq[i], pts_sq[(i + 1) % 4], layer_color, thickness, cv2.LINE_AA)
                
            # 4. Connecting Radial Lines from Inner to Outer
            for i in range(12):
                theta = angle_outer + i * (math.pi / 6)
                x1 = int(cx + inner_r * math.cos(theta))
                y1 = int(cy + inner_r * math.sin(theta))
                x2 = int(cx + max(1, radius - 15) * math.cos(theta))
                y2 = int(cy + max(1, radius - 15) * math.sin(theta))
                cv2.line(overlay, (x1,y1), (x2,y2), layer_color, max(1, thickness-1), cv2.LINE_AA)
                
            # 5. Glowing Core (soft blend orb + small dot)
            core_r = int(25 * (1.0 + 0.1 * math.sin(self.time * 10.0)))
            if thickness > 3:
                cv2.circle(overlay, center, core_r, layer_color, -1, cv2.LINE_AA)
            elif thickness == 1:
                cv2.circle(overlay, center, 8, layer_color, -1, cv2.LINE_AA)
                
            # 6. Outer Runes/Arcs
            # Drawing disjoint arcs via cv2.ellipse
            arc_radius = int(radius + 20)
            axes = (arc_radius, arc_radius)
            for i in range(4):
                theta_mid = angle_inner * 1.5 + i * (math.pi / 2)
                start_rad = theta_mid - 0.2
                end_rad = theta_mid + 0.2
                cv2.ellipse(overlay, center, axes, 0, math.degrees(start_rad), math.degrees(end_rad), layer_color, thickness, cv2.LINE_AA)

# ============================================================================
# GESTURE-TRIGGERED SUPERHERO VFX EFFECTS
# ============================================================================
#
#   Gesture               Effect
#   -------------------   ---------------------------------------------
#   🖐️  Open Palm          🛡️  Energy Shield   (see MagicShieldEffect above)
#   ☝️  Point               ⚡ Repulsor Beam    -> RepulsorBeamEffect
#   🤏 Pinch                💥 Energy Blast     -> EnergyBlastEffect
#   ✊ Fist                 🔥 Power Charge     -> PowerChargeEffect
#   👉 Move hand            ✨ Energy Trail     -> EnergyTrailEffect
#   🖐️+🖐️ (far apart)      🌀 Portal Effect    -> PortalEffect
#   ✌️  Two fingers         🔴 Laser / Scan     -> LaserScanEffect
#   👍 Thumbs up            ⚡ Power Activated  -> PowerActivatedEffect
#   👎 Thumbs down          🚫 Power Disabled   (fades every other effect)
#   🤲 Two palms (close)    🔵 Energy Ball      -> EnergyBallEffect
#   🤟 Rock on              ⭐ Star Nova        -> StarNovaEffect
#
# ============================================================================

class RepulsorBeamEffect:
    """⚡ Repulsor Beam — an energy beam shoots forward from a pointing fingertip."""

    def __init__(self, config: Config):
        self.config = config
        self.alpha = 0.0
        self.time = 0.0
        self.tip: Optional[Tuple[int, int]] = None
        self.direction: Optional[Tuple[float, float]] = None

    def update(self, dt: float, landmarks: Optional[List[Tuple[int, int]]]):
        self.time += dt
        target = 1.0 if landmarks is not None else 0.0
        self.alpha += (target - self.alpha) * dt * self.config.gesture_fade_speed
        self.alpha = max(0.0, min(1.0, self.alpha))

        if landmarks is not None and len(landmarks) > 8:
            pip = landmarks[6]
            tip = landmarks[8]
            vx, vy = tip[0] - pip[0], tip[1] - pip[1]
            norm = math.hypot(vx, vy)
            if norm > 1e-3:
                self.direction = (vx / norm, vy / norm)
            self.tip = (int(tip[0]), int(tip[1]))

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if self.alpha <= 0.01 or self.tip is None or self.direction is None:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        color = self.config.colors['repulsor_beam']
        length = self.config.repulsor_beam_length
        end = (int(self.tip[0] + self.direction[0] * length),
               int(self.tip[1] + self.direction[1] * length))

        glow_line(overlay, self.tip, end, color, alpha=self.alpha, base_thickness=3, layers=6)

        # Traveling energy pulses racing along the beam
        for i in range(3):
            t = ((self.time * self.config.repulsor_pulse_speed) + i / 3.0) % 1.0
            px = int(self.tip[0] + self.direction[0] * length * t)
            py = int(self.tip[1] + self.direction[1] * length * t)
            glow_dot(overlay, (px, py), 8, (255, 255, 255), alpha=(1.0 - t) * self.alpha, layers=3)

        # Glowing emitter at the fingertip
        pulse = 1.0 + 0.2 * math.sin(self.time * 10.0)
        glow_dot(overlay, self.tip, int(16 * pulse), color, alpha=self.alpha, layers=4)
        glow_dot(overlay, self.tip, 6, (255, 255, 255), alpha=self.alpha, layers=2)

        cv2.add(frame, overlay, dst=frame)
        return frame


class EnergyBlastEffect:
    """💥 Energy Blast — a small orb forms at a pinch point and expands as it's charged."""

    def __init__(self, config: Config):
        self.config = config
        self.alpha = 0.0
        self.time = 0.0
        self.center: Optional[Tuple[int, int]] = None
        self.charge = 0.0
        self.was_active = False
        self.bursts: List[dict] = []

    def update(self, dt: float, landmarks: Optional[List[Tuple[int, int]]]):
        self.time += dt
        active = landmarks is not None
        target = 1.0 if active else 0.0
        self.alpha += (target - self.alpha) * dt * self.config.gesture_fade_speed
        self.alpha = max(0.0, min(1.0, self.alpha))

        if active and len(landmarks) > 8:
            x = int((landmarks[4][0] + landmarks[8][0]) / 2)
            y = int((landmarks[4][1] + landmarks[8][1]) / 2)
            self.center = (x, y)
            self.charge = min(1.0, self.charge + dt * self.config.blast_charge_speed)
        else:
            if self.was_active and self.center is not None:
                self.bursts.append({'radius': 10.0 + 40.0 * self.charge, 'alpha': 1.0,
                                     'center': self.center})
            self.charge = max(0.0, self.charge - dt * self.config.blast_charge_speed * 2)

        self.was_active = active

        alive = []
        for b in self.bursts:
            b['radius'] += 220 * dt
            b['alpha'] -= dt * 2.5
            if b['alpha'] > 0:
                alive.append(b)
        self.bursts = alive

    def draw(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        color = self.config.colors['energy_blast']
        drew = False

        if self.alpha > 0.01 and self.center is not None:
            radius = int(10 + self.config.blast_max_radius * self.charge)
            pulse = 1.0 + 0.08 * math.sin(self.time * 14.0)
            glow_dot(overlay, self.center, int(radius * pulse), color, alpha=self.alpha, layers=5)
            glow_dot(overlay, self.center, max(2, int(radius * 0.35)), (255, 255, 255),
                      alpha=self.alpha, layers=2)
            drew = True

        for b in self.bursts:
            glow_dot(overlay, b['center'], int(b['radius']), color, alpha=max(0.0, b['alpha']), layers=2)
            drew = True

        if drew:
            cv2.add(frame, overlay, dst=frame)
        return frame


class PowerChargeEffect:
    """🔥 Power Charge — a closed fist builds a crackling glowing aura."""

    def __init__(self, config: Config):
        self.config = config
        self.alpha = 0.0
        self.time = 0.0
        self.landmarks: Optional[List[Tuple[int, int]]] = None
        self.charge = 0.0

    def update(self, dt: float, landmarks: Optional[List[Tuple[int, int]]]):
        self.time += dt
        target = 1.0 if landmarks is not None else 0.0
        self.alpha += (target - self.alpha) * dt * self.config.gesture_fade_speed
        self.alpha = max(0.0, min(1.0, self.alpha))
        if landmarks is not None:
            self.landmarks = landmarks
            self.charge = min(1.0, self.charge + dt * 0.8)
        else:
            self.charge = max(0.0, self.charge - dt * 1.5)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if self.alpha <= 0.01 or self.landmarks is None:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        center = _palm_center(self.landmarks)
        core_color = self.config.colors['power_charge_core']
        outer_color = self.config.colors['power_charge_outer']

        pulse = 1.0 + 0.15 * math.sin(self.time * self.config.power_charge_pulse_speed)
        base_r = int((30 + 18 * self.charge) * pulse)

        glow_dot(overlay, center, base_r, outer_color, alpha=self.alpha * 0.7, layers=4)
        glow_dot(overlay, center, int(base_r * 0.55), core_color, alpha=self.alpha, layers=4)

        # Crackling energy rings expanding outward from the fist
        for i in range(3):
            phase = (self.time * 1.5 + i / 3.0) % 1.0
            r = int(base_r * 0.6 + phase * 70)
            a = self.alpha * (1.0 - phase) * 0.6
            c = tuple(int(ch * a) for ch in outer_color)
            cv2.circle(overlay, center, r, c, 2, cv2.LINE_AA)

        cv2.add(frame, overlay, dst=frame)
        return frame


class LaserScanEffect:
    """🔴 Laser / Scan — a red scanning beam follows two extended fingers."""

    def __init__(self, config: Config):
        self.config = config
        self.alpha = 0.0
        self.time = 0.0
        self.tip1: Optional[Tuple[int, int]] = None
        self.tip2: Optional[Tuple[int, int]] = None

    def update(self, dt: float, landmarks: Optional[List[Tuple[int, int]]]):
        self.time += dt
        target = 1.0 if landmarks is not None else 0.0
        self.alpha += (target - self.alpha) * dt * self.config.gesture_fade_speed
        self.alpha = max(0.0, min(1.0, self.alpha))
        if landmarks is not None and len(landmarks) > 12:
            self.tip1 = (landmarks[8][0], landmarks[8][1])
            self.tip2 = (landmarks[12][0], landmarks[12][1])

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if self.alpha <= 0.01 or self.tip1 is None or self.tip2 is None:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        color = self.config.colors['laser_scan']
        my = int((self.tip1[1] + self.tip2[1]) / 2)

        # Full-width scanning line at fingertip height
        glow_line(overlay, (0, my), (w, my), color, alpha=self.alpha * 0.6, base_thickness=1, layers=3)

        # Bright glint travelling back and forth along the scan line
        glint_x = int((math.sin(self.time * self.config.laser_scan_speed) * 0.5 + 0.5) * w)
        glow_dot(overlay, (glint_x, my), 10, (255, 255, 255), alpha=self.alpha, layers=3)

        # Targeting brackets around each fingertip
        for tip in (self.tip1, self.tip2):
            s = 14
            corners = [
                ((tip[0] - s, tip[1] - s), (tip[0] - s + 6, tip[1] - s)),
                ((tip[0] - s, tip[1] - s), (tip[0] - s, tip[1] - s + 6)),
                ((tip[0] + s, tip[1] - s), (tip[0] + s - 6, tip[1] - s)),
                ((tip[0] + s, tip[1] - s), (tip[0] + s, tip[1] - s + 6)),
                ((tip[0] - s, tip[1] + s), (tip[0] - s + 6, tip[1] + s)),
                ((tip[0] - s, tip[1] + s), (tip[0] - s, tip[1] + s - 6)),
                ((tip[0] + s, tip[1] + s), (tip[0] + s - 6, tip[1] + s)),
                ((tip[0] + s, tip[1] + s), (tip[0] + s, tip[1] + s - 6)),
            ]
            for p1, p2 in corners:
                glow_line(overlay, p1, p2, color, alpha=self.alpha, base_thickness=1, layers=2)
            glow_dot(overlay, tip, 4, color, alpha=self.alpha, layers=2)

        cv2.add(frame, overlay, dst=frame)
        return frame


class PowerActivatedEffect:
    """⚡ Power Activated — a thumbs-up triggers a screen flash and particle burst."""

    def __init__(self, config: Config):
        self.config = config
        self.time = 0.0
        self.was_active = False
        self.flash_timer = 0.0
        self.particles: List[GestureParticle] = []
        self.alpha = 0.0  # activity level, used for the "Active:" HUD label

    def update(self, dt: float, landmarks: Optional[List[Tuple[int, int]]]):
        self.time += dt
        active = landmarks is not None

        if active and not self.was_active:
            # Rising edge: gesture just started -> fire the activation burst
            self.flash_timer = self.config.flash_duration
            cx, cy = _palm_center(landmarks) if len(landmarks) >= 21 else (0, 0)
            for _ in range(60):
                ang = random.uniform(0, 2 * math.pi)
                speed = random.uniform(80, 320)
                self.particles.append(GestureParticle(
                    x=cx, y=cy, vx=math.cos(ang) * speed, vy=math.sin(ang) * speed,
                    life=0.0, max_life=random.uniform(0.4, 0.9),
                    size=random.uniform(3, 6),
                    color=self.config.colors['power_activated'],
                ))
        self.was_active = active

        if self.flash_timer > 0:
            self.flash_timer -= dt

        alive = []
        for p in self.particles:
            p.life += dt
            if p.life < p.max_life:
                p.x += p.vx * dt
                p.y += p.vy * dt
                p.vx *= (1.0 - 2.0 * dt)
                p.vy *= (1.0 - 2.0 * dt)
                alive.append(p)
        self.particles = alive[-300:]

        # Alpha reflects current activity: full while the flash is playing,
        # otherwise fades based on whether particles are still alive. Used
        # only for the "Active:" HUD label, not for drawing.
        target = 1.0 if (self.flash_timer > 0 or self.particles) else 0.0
        self.alpha += (target - self.alpha) * dt * self.config.gesture_fade_speed
        self.alpha = max(0.0, min(1.0, self.alpha))

    def draw(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        if self.flash_timer > 0:
            t = self.flash_timer / self.config.flash_duration
            flash = np.full_like(frame, 255, dtype=np.uint8)
            cv2.addWeighted(frame, 1.0 - 0.5 * t, flash, 0.5 * t, 0, dst=frame)

        if self.particles:
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            for p in self.particles:
                t = 1.0 - (p.life / p.max_life)
                glow_dot(overlay, (int(p.x), int(p.y)), max(1, int(p.size * t)), p.color, alpha=t, layers=2)
            cv2.add(frame, overlay, dst=frame)

        return frame


class PortalEffect:
    """🌀 Portal Effect — a large swirling portal opens between two spread-apart open palms."""

    def __init__(self, config: Config):
        self.config = config
        self.alpha = 0.0
        self.time = 0.0
        self.center: Optional[Tuple[int, int]] = None
        self.scale = 1.0

    def update(self, dt: float, active: bool, center: Optional[Tuple[int, int]] = None,
               span: Optional[float] = None):
        self.time += dt
        target = 1.0 if active else 0.0
        self.alpha += (target - self.alpha) * dt * self.config.gesture_fade_speed
        self.alpha = max(0.0, min(1.0, self.alpha))
        if active and center is not None:
            self.center = center
            self.scale = max(0.6, min(2.2, span / 260.0)) if span else 1.0

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if self.alpha <= 0.01 or self.center is None:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        cx, cy = self.center
        colors = [self.config.colors['portal_a'], self.config.colors['portal_b']]
        base_radius = int(60 * self.scale)

        for ring in range(7):
            radius = int(base_radius * 0.3) + ring * int(10 * self.scale)
            rot = self.time * (2.0 + ring * 0.25) * (1 if ring % 2 == 0 else -1)
            color = colors[ring % 2]
            a = self.alpha * (1.0 - ring / 9.0)
            axes = (radius, int(radius * 0.92))
            c = tuple(int(ch * a) for ch in color)
            cv2.ellipse(overlay, (cx, cy), axes, math.degrees(rot), 0, 300, c, 2, cv2.LINE_AA)

        for arm in range(3):
            pts = []
            for i in range(16):
                t = i / 15.0
                ang = self.time * 3.0 + arm * (2 * math.pi / 3) + t * 6.0
                r = base_radius * 0.9 * (1 - t)
                pts.append((int(cx + r * math.cos(ang)), int(cy + r * math.sin(ang))))
            for i in range(len(pts) - 1):
                glow_line(overlay, pts[i], pts[i + 1], colors[arm % 2], alpha=self.alpha,
                          base_thickness=1, layers=2)

        # Tiny galaxy-dust stars scattered inside the disc — the golden angle
        # gives an even, non-jittery scatter that only twinkles in place
        for i in range(14):
            frac = (i * 0.6180339887) % 1.0
            ang = i * 2.399963 + self.time * 0.6
            rad = base_radius * 0.85 * math.sqrt(frac)
            sx = int(cx + rad * math.cos(ang))
            sy = int(cy + rad * math.sin(ang) * 0.92)
            twinkle = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(self.time * 3.0 + i * 1.7))
            glow_dot(overlay, (sx, sy), 2, (255, 255, 255), alpha=self.alpha * twinkle, layers=2)

        glow_dot(overlay, (cx, cy), int(8 * self.scale), (255, 255, 255), alpha=self.alpha, layers=2)
        cv2.add(frame, overlay, dst=frame)
        return frame


class EnergyBallEffect:
    """🔵 Energy Ball — a glowing sphere grows in the space between two cupped palms."""

    def __init__(self, config: Config):
        self.config = config
        self.alpha = 0.0
        self.time = 0.0
        self.center: Optional[Tuple[int, int]] = None
        self.charge = 0.0

    def update(self, dt: float, active: bool, center: Optional[Tuple[int, int]] = None):
        self.time += dt
        target = 1.0 if active else 0.0
        self.alpha += (target - self.alpha) * dt * self.config.gesture_fade_speed
        self.alpha = max(0.0, min(1.0, self.alpha))
        if active:
            self.charge = min(1.0, self.charge + dt * 0.6)
            if center is not None:
                self.center = center
        else:
            self.charge = max(0.0, self.charge - dt * 1.2)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if self.alpha <= 0.01 or self.center is None:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        cx, cy = self.center
        core = self.config.colors['energy_ball_core']
        outer = self.config.colors['energy_ball_outer']

        radius = int(18 + self.config.energy_ball_max_radius * self.charge)
        pulse = 1.0 + 0.08 * math.sin(self.time * 10.0)
        r = int(radius * pulse)

        glow_dot(overlay, (cx, cy), r, outer, alpha=self.alpha * 0.8, layers=5)
        glow_dot(overlay, (cx, cy), int(r * 0.55), core, alpha=self.alpha, layers=4)
        glow_dot(overlay, (cx, cy), max(2, int(r * 0.2)), (255, 255, 255), alpha=self.alpha, layers=2)

        for i in range(4):
            ang = self.time * 3.0 + i * (math.pi / 2)
            sx = int(cx + r * 1.15 * math.cos(ang))
            sy = int(cy + r * 1.15 * math.sin(ang) * 0.6)
            glow_dot(overlay, (sx, sy), 4, (255, 255, 255), alpha=self.alpha * 0.8, layers=2)

        cv2.add(frame, overlay, dst=frame)
        return frame


class EnergyTrailEffect:
    """✨ Energy Trail — a fading light trail follows a hand as it moves."""

    def __init__(self, config: Config):
        self.config = config
        self.points: List[dict] = []
        self.last_pos: Optional[Tuple[int, int]] = None

    def update(self, dt: float, landmarks_list: Optional[List[List[Tuple[int, int]]]]):
        pos = None
        speed = 0.0
        if landmarks_list:
            best_speed, best_pos = 0.0, None
            for lm in landmarks_list:
                p = _palm_center(lm)
                s = (math.hypot(p[0] - self.last_pos[0], p[1] - self.last_pos[1]) / max(dt, 1e-4)
                     if self.last_pos is not None else 0.0)
                if s >= best_speed:
                    best_speed, best_pos = s, p
            pos, speed = best_pos, best_speed
            self.last_pos = pos
        else:
            self.last_pos = None

        if pos is not None and speed > self.config.trail_speed_threshold:
            self.points.append({'x': pos[0], 'y': pos[1], 'age': 0.0})

        alive = []
        for p in self.points:
            p['age'] += dt
            if p['age'] < 0.5:
                alive.append(p)
        self.points = alive[-self.config.trail_max_points:]

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if not self.points:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        color = self.config.colors['trail_color']
        n = len(self.points)

        for i, p in enumerate(self.points):
            t = 1.0 - (p['age'] / 0.5)
            size = max(1, int(4 + 10 * (i / max(1, n - 1)) * t))
            glow_dot(overlay, (int(p['x']), int(p['y'])), size, color, alpha=t * 0.85, layers=2)

        for i in range(len(self.points) - 1):
            p1, p2 = self.points[i], self.points[i + 1]
            t = 1.0 - (p2['age'] / 0.5)
            glow_line(overlay, (int(p1['x']), int(p1['y'])), (int(p2['x']), int(p2['y'])),
                      color, alpha=t * 0.5, base_thickness=1, layers=2)

        cv2.add(frame, overlay, dst=frame)
        return frame


class StarNovaEffect:
    """⭐ Star Nova — rock-on sign (index + pinky extended) summons a pulsing
    five-point star at the palm, ringed by orbiting sparkles, that periodically
    fires small shooting stars outward."""

    def __init__(self, config: Config):
        self.config = config
        self.alpha = 0.0
        self.time = 0.0
        self.center: Optional[Tuple[int, int]] = None
        self.charge = 0.0
        self.shooting_stars: List[dict] = []
        self.spawn_timer = 0.0

    def update(self, dt: float, landmarks: Optional[List[Tuple[int, int]]]):
        self.time += dt
        active = landmarks is not None
        target = 1.0 if active else 0.0
        self.alpha += (target - self.alpha) * dt * self.config.gesture_fade_speed
        self.alpha = max(0.0, min(1.0, self.alpha))

        if active and len(landmarks) >= 21:
            self.center = _palm_center(landmarks)
            self.charge = min(1.0, self.charge + dt * 0.9)
        else:
            self.charge = max(0.0, self.charge - dt * 1.5)

        if active and self.center is not None:
            self.spawn_timer -= dt
            if self.spawn_timer <= 0:
                self.spawn_timer = self.config.nova_burst_interval * random.uniform(0.7, 1.3)
                ang = random.uniform(0, 2 * math.pi)
                speed = random.uniform(180, 320)
                self.shooting_stars.append({
                    'x': float(self.center[0]), 'y': float(self.center[1]),
                    'vx': math.cos(ang) * speed, 'vy': math.sin(ang) * speed,
                    'life': 0.0, 'max_life': random.uniform(0.5, 0.9),
                })

        alive = []
        for s in self.shooting_stars:
            s['life'] += dt
            if s['life'] < s['max_life']:
                s['x'] += s['vx'] * dt
                s['y'] += s['vy'] * dt
                alive.append(s)
        self.shooting_stars = alive[-40:]

    def draw(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        drew = False
        core = self.config.colors['nova_core']
        outer = self.config.colors['nova_outer']

        if self.alpha > 0.01 and self.center is not None:
            cx, cy = self.center
            pulse = 1.0 + 0.12 * math.sin(self.time * 9.0)
            r = int((26 + 22 * self.charge) * pulse)

            glow_dot(overlay, (cx, cy), int(r * 1.6), outer, alpha=self.alpha * 0.5, layers=4)
            glow_dot(overlay, (cx, cy), r, core, alpha=self.alpha, layers=4)
            glow_dot(overlay, (cx, cy), max(2, int(r * 0.3)), (255, 255, 255), alpha=self.alpha, layers=2)

            # Rotating five-point star spikes
            spikes = 5
            spike_len = r * 1.8
            rot = self.time * 1.5
            for i in range(spikes):
                ang = rot + i * (2 * math.pi / spikes)
                tip = (int(cx + spike_len * math.cos(ang)), int(cy + spike_len * math.sin(ang)))
                glow_line(overlay, (cx, cy), tip, core, alpha=self.alpha * 0.7, base_thickness=2, layers=3)

            # Small orbiting sparkles
            for i in range(6):
                ang = self.time * 2.2 + i * (2 * math.pi / 6)
                orbit_r = r * 1.3
                sx = int(cx + orbit_r * math.cos(ang))
                sy = int(cy + orbit_r * math.sin(ang))
                glow_dot(overlay, (sx, sy), 3, (255, 255, 255), alpha=self.alpha * 0.8, layers=2)

            drew = True

        for s in self.shooting_stars:
            t = 1.0 - (s['life'] / s['max_life'])
            tail = (int(s['x'] - s['vx'] * 0.05), int(s['y'] - s['vy'] * 0.05))
            glow_line(overlay, tail, (int(s['x']), int(s['y'])), (255, 255, 255), alpha=t, base_thickness=1, layers=3)
            glow_dot(overlay, (int(s['x']), int(s['y'])), 3, core, alpha=t, layers=2)
            drew = True

        if drew:
            cv2.add(frame, overlay, dst=frame)
        return frame


class GestureEffectManager:
    """
    Detects hand gestures every frame and drives the matching superhero VFX,
    including the two-hand combo gestures (Portal / Energy Ball) and an
    always-on motion trail. A thumbs-down ("Power Disabled") instantly
    fades every other active effect out.
    """

    def __init__(self, config: Config):
        self.config = config
        self.single_effects = {
            'point':     RepulsorBeamEffect(config),
            'pinch':     EnergyBlastEffect(config),
            'fist':      PowerChargeEffect(config),
            'peace':     LaserScanEffect(config),
            'thumbs_up': PowerActivatedEffect(config),
            'rock_on':   StarNovaEffect(config),
        }
        self.portal = PortalEffect(config)
        self.energy_ball = EnergyBallEffect(config)
        self.trail = EnergyTrailEffect(config)
        self.power_disabled = False

    def update(self, dt: float, landmarks_list: Optional[List[List[Tuple[int, int]]]]):
        detected = []
        if landmarks_list:
            for lm in landmarks_list:
                g = GestureRecognizer.classify(lm)
                if g:
                    detected.append((g, lm))

        thumbs_down = any(g == 'thumbs_down' for g, _ in detected)
        self.power_disabled = thumbs_down

        # Two-hand combo: both hands showing an open palm.
        # Hands held far apart -> Portal. Hands held close together -> Energy Ball.
        open_palms = [lm for g, lm in detected if g == 'open_palm']
        portal_active = False
        ball_active = False
        combo_center = None
        span = None
        if not thumbs_down and len(open_palms) == 2:
            c1 = _palm_center(open_palms[0])
            c2 = _palm_center(open_palms[1])
            span = math.hypot(c2[0] - c1[0], c2[1] - c1[1])
            combo_center = (int((c1[0] + c2[0]) / 2), int((c1[1] + c2[1]) / 2))
            if span >= self.config.portal_min_hand_distance:
                portal_active = True
            else:
                ball_active = True

        self.portal.update(dt, portal_active, combo_center, span)
        self.energy_ball.update(dt, ball_active, combo_center)

        for name, effect in self.single_effects.items():
            match = None if thumbs_down else next((lm for g, lm in detected if g == name), None)
            effect.update(dt, match)

        # Motion trail: independent of pose, but suppressed while powered down
        self.trail.update(dt, [] if thumbs_down else landmarks_list)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if not self.config.enable_gesture_vfx:
            return frame
        if self.config.enable_trails:
            frame = self.trail.draw(frame)
        for effect in self.single_effects.values():
            frame = effect.draw(frame)
        frame = self.portal.draw(frame)
        frame = self.energy_ball.draw(frame)
        return frame

    def active_label(self) -> Optional[str]:
        if self.power_disabled:
            return GestureRecognizer.GESTURE_LABELS['thumbs_down']

        best_name, best_alpha = None, 0.0
        for name, effect in self.single_effects.items():
            if effect.alpha > best_alpha:
                best_alpha, best_name = effect.alpha, name
        if self.portal.alpha > best_alpha:
            best_alpha, best_name = self.portal.alpha, 'portal'
        if self.energy_ball.alpha > best_alpha:
            best_alpha, best_name = self.energy_ball.alpha, 'energy_ball'

        if best_name and best_alpha > 0.25:
            if best_name in GestureRecognizer.COMBO_LABELS:
                return GestureRecognizer.COMBO_LABELS[best_name]
            return GestureRecognizer.GESTURE_LABELS[best_name]
        return None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class HandVFXApp:
    """Main application class"""

    def __init__(self):
        self.config = Config()
        self.hand_tracker = HandTracker(self.config)
        self.neon_renderer = NeonRenderer(self.config)
        self.shield_effect = MagicShieldEffect(self.config)
        self.gesture_vfx = GestureEffectManager(self.config)
        self.last_time = time.time()
        self.frame_count = 0

        # Camera init with macOS fallback
        print("[VFX] Opening webcam...")
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            print("[VFX] CAP_AVFOUNDATION failed, trying default backend...")
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("[VFX] ERROR: Could not open webcam on index 0, trying index 1...")
            self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            print("[VFX] ERROR: No webcam found. Exiting.")
            self.cap = None
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("[VFX] Webcam opened successfully.")

        cv2.namedWindow("Hand VFX System", cv2.WINDOW_NORMAL)
        print("[VFX] Window created.")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame and return it with VFX applied"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        self.neon_renderer.update(dt)

        landmarks_list = self.hand_tracker.detect_hands(frame)

        if landmarks_list:
            hand_colors = [
                self.config.colors['primary'],
                self.config.colors['secondary'],
            ]
            for hand_idx, landmarks in enumerate(landmarks_list):
                color = hand_colors[hand_idx % len(hand_colors)]
                self.neon_renderer.draw_hand_skeleton(frame, landmarks, color)
                self.neon_renderer.draw_hand_aura(frame, landmarks, color)

        # Update and draw magic shield (🖐️ Open Palm -> 🛡️ Energy Shield)
        self.shield_effect.update(dt, landmarks_list)
        frame = self.shield_effect.draw(frame)

        # Update and draw the rest of the gesture-triggered superhero VFX
        # (Repulsor Beam, Energy Blast, Power Charge, Laser/Scan, Power
        # Activated/Disabled, Portal, Energy Ball, Energy Trail)
        self.gesture_vfx.update(dt, landmarks_list)
        frame = self.gesture_vfx.draw(frame)

        # Cinematic tint overlay
        if self.config.cinematic_darkening > 0:
            tint = np.full_like(frame, (10, 5, 20), dtype=np.uint8)
            cv2.addWeighted(frame, 1.0 - self.config.cinematic_darkening,
                            tint, self.config.cinematic_darkening, 0, dst=frame)

        # Subtle bloom
        if self.config.bloom_intensity > 0:
            bloom = cv2.GaussianBlur(frame, (21, 21), 0)
            cv2.addWeighted(frame, 1.0, bloom, self.config.bloom_intensity, 0, dst=frame)

        self.draw_info(frame)
        return frame

    def draw_info(self, frame: np.ndarray):
        """Draw HUD text"""
        lines = [
            "Hand VFX  |  q: quit",
            f"Aura: {'ON' if self.config.enable_aura else 'OFF'} (a)  "
            f"Trail: {'ON' if self.config.enable_trails else 'OFF'} (t)  "
            f"Shield: {'ON' if self.config.enable_shield else 'OFF'} (s)  "
            f"Gestures: {'ON' if self.config.enable_gesture_vfx else 'OFF'} (g)",
        ]

        active_gesture = self.gesture_vfx.active_label()
        if not active_gesture and self.shield_effect.current_alpha > 0.25:
            active_gesture = GestureRecognizer.GESTURE_LABELS['open_palm']
        if active_gesture:
            lines.append(f"Active: {active_gesture}")

        y = 28
        for text in lines:
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            y += 24

    def handle_key(self, key: int):
        """Handle keyboard input"""
        if key == ord('a'):
            self.config.enable_aura = not self.config.enable_aura
            print(f"[VFX] Aura: {'ON' if self.config.enable_aura else 'OFF'}")
        elif key == ord('t'):
            self.config.enable_trails = not self.config.enable_trails
            print(f"[VFX] Trail: {'ON' if self.config.enable_trails else 'OFF'}")
        elif key == ord('s'):
            self.config.enable_shield = not self.config.enable_shield
            print(f"[VFX] Shield: {'ON' if self.config.enable_shield else 'OFF'}")
        elif key == ord('g'):
            self.config.enable_gesture_vfx = not self.config.enable_gesture_vfx
            print(f"[VFX] Gesture VFX: {'ON' if self.config.enable_gesture_vfx else 'OFF'}")

    def run(self):
        """Main application loop"""
        if self.cap is None:
            print("[VFX] Cannot run — no camera available.")
            return

        print("[VFX] Starting. Controls: q=quit  a=aura  t=trail  s=shield  g=gesture VFX")
        print("[VFX] Gestures:")
        print("       Open palm (4 fingers spread)        -> Energy Shield")
        print("       Point (only index extended)         -> Repulsor Beam")
        print("       Pinch (thumb+index touching)        -> Energy Blast")
        print("       Closed fist                         -> Power Charge")
        print("       Move your hand                      -> Energy Trail")
        print("       Two open palms, held far apart      -> Portal Effect")
        print("       Two fingers (index+middle)          -> Laser / Scan")
        print("       Thumbs up                           -> Power Activated")
        print("       Thumbs down                         -> Power Disabled")
        print("       Two open palms, held close together -> Energy Ball")
        print("       Rock on (index+pinky, others curled)-> Star Nova")

        # Validate first frame
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("[VFX] ERROR: Could not read first frame.")
            self.cap.release()
            return
        print("[VFX] First frame received. Entering main loop.")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("[VFX] WARNING: Frame read failed, retrying...")
                    continue

                frame = cv2.flip(frame, 1)
                processed = self.process_frame(frame)

                cv2.imshow("Hand VFX System", processed)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[VFX] Quit key pressed.")
                    break
                self.handle_key(key)

                self.frame_count += 1

        except KeyboardInterrupt:
            print("[VFX] Interrupted by user.")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("[VFX] Application closed.")


def main():
    app = HandVFXApp()
    app.run()


if __name__ == "__main__":
    main()