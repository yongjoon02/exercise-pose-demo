"""
Video-based Pose Correction System
영상 입력 분석 전용 
"""

import argparse
import os
import sys
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple
import time


class VideoPoseCorrector:
    def __init__(self):
        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # 운동 모드/상태
        self.exercise_mode = "squat"
        self.frame_count = 0
        self.rep_count = 0
        self.exercise_state = "up"
        self.prev_exercise_state = "up"
        
        # 피드백
        self.feedback_messages = []
        self.last_feedback_time = 0
        self.feedback_cooldown = 0.0
        
        # 스무딩
        self.prev_landmarks = None
        self.smoothing_alpha = 0.7
        
        # 최근 포즈 검출 여부
        self.last_pose_detected = False
        # 최근 프레임 점수/에러
        self.last_score = 0
        self.last_errors = []
        
    def calculate_angle(self, a: List[float], b: List[float], c: List[float]) -> float:
        a = np.array(a); b = np.array(b); c = np.array(c)
        ba = a - b; bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))
    
    def calculate_distance(self, a: List[float], b: List[float]) -> float:
        return np.linalg.norm(np.array(a) - np.array(b))
    
    def get_landmark_coords(self, landmarks, idx: int) -> List[float]:
        lm = landmarks.landmark[idx]
        return [lm.x, lm.y, lm.z]
    
    def smooth_landmarks(self, current_landmarks):
        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks
            return current_landmarks
        smoothed = []
        for curr, prev in zip(current_landmarks.landmark, self.prev_landmarks.landmark):
            smoothed_lm = type(curr)()
            smoothed_lm.x = self.smoothing_alpha * curr.x + (1 - self.smoothing_alpha) * prev.x
            smoothed_lm.y = self.smoothing_alpha * curr.y + (1 - self.smoothing_alpha) * prev.y
            smoothed_lm.z = self.smoothing_alpha * curr.z + (1 - self.smoothing_alpha) * prev.z
            smoothed_lm.visibility = curr.visibility
            smoothed.append(smoothed_lm)
        self.prev_landmarks = current_landmarks
        return type(current_landmarks)(landmark=smoothed)
    
    def check_squat(self, landmarks) -> Tuple[float, List[str]]:
        errors, score = [], 100.0
        
        LSH = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        RSH = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        LHIP = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP.value)
        RHIP = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP.value)
        LKNE = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE.value)
        RKNE = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE.value)
        LANK = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE.value)
        RANK = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ANKLE.value)
        
        left_knee_angle  = self.calculate_angle(LHIP, LKNE, LANK)
        right_knee_angle = self.calculate_angle(RHIP, RKNE, RANK)
        avg_knee_angle   = (left_knee_angle + right_knee_angle) / 2.0

        # 히스테리시스 기준(100°까지 내려가도록)
        UP_ENTER, DOWN_ENTER = 160.0, 100.0
        if avg_knee_angle >= UP_ENTER:
            self.exercise_state = "up"
        elif avg_knee_angle <= DOWN_ENTER:
            self.exercise_state = "down"

        # DOWN 상태에서 깊이 체크 
        if self.exercise_state == "down" and avg_knee_angle > 115:
            errors.append("Bend your knees more"); score -= 15

        mid_sh = [(LSH[i] + RSH[i]) / 2 for i in range(3)]
        mid_hip = [(LHIP[i] + RHIP[i]) / 2 for i in range(3)]
        mid_knee = [(LKNE[i] + RKNE[i]) / 2 for i in range(3)]
        torso_angle = self.calculate_angle(mid_sh, mid_hip, mid_knee)
        if torso_angle < 45:
            errors.append("Keep your torso upright"); score -= 15

        # 무릎-발끝 정렬 
        if LKNE[0] < LANK[0] - 0.05:
            errors.append("Left knee passes toes"); score -= 15
        if RKNE[0] > RANK[0] + 0.05:
            errors.append("Right knee passes toes"); score -= 15

        # 무릎 벌어짐 체크 
        knee_distance = self.calculate_distance(LKNE[:2], RKNE[:2])
        hip_distance  = self.calculate_distance(LHIP[:2], RHIP[:2])
        if knee_distance / (hip_distance + 1e-6) < 0.7:
            errors.append("Avoid knee valgus"); score -= 15

        # 복합 에러 페널티
        if len(errors) >= 3:
            score -= 10

        return max(0, score), errors
    
    def check_pushup(self, landmarks) -> Tuple[float, List[str]]:
        errors, score = [], 100.0
        LSH = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        RSH = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        LEL = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW.value)
        REL = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        LWR = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST.value)
        LHP = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP.value)
        LAK = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE.value)

        elbow_angle = self.calculate_angle(LSH, LEL, LWR)
        # 기준 (85°까지 내려가도록)
        UP_ENTER, DOWN_ENTER = 155.0, 85.0
        if elbow_angle >= UP_ENTER:
            self.exercise_state = "up"
        elif elbow_angle <= DOWN_ENTER:
            self.exercise_state = "down"

        # DOWN 상태에서 깊이 체크
        if self.exercise_state == "down" and elbow_angle > 95:
            errors.append("Lower further"); score -= 15
        # 중간 범위에서도 체크
        elif 110 < elbow_angle < 155:
            errors.append("Lower further"); score -= 10

        # 몸통 일직선 체크
        body_y_coords = [LSH[1], LHP[1], LAK[1]]
        if np.std(body_y_coords) * 10 > 0.3:
            errors.append("Keep body straight"); score -= 20

        # 팔꿈치-어깨 정렬 체크 (어깨 부상 방지)
        avg_elbow_y = (LEL[1] + REL[1]) / 2
        avg_shoulder_y = (LSH[1] + RSH[1]) / 2
        if abs(avg_elbow_y - avg_shoulder_y) > 0.15:
            errors.append("Keep elbows close"); score -= 15

        # 복합 에러 페널티
        if len(errors) >= 3:
            score -= 10

        return max(0, score), errors
    
    def check_plank(self, landmarks) -> Tuple[float, List[str]]:
        errors, score = [], 100.0
        LSH = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        LHP = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP.value)
        LAK = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE.value)
        LEL = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW.value)

        # 몸통 일직선 체크 (가장 중요)
        body_y = [LSH[1], LHP[1], LAK[1]]
        if np.std(body_y) * 10 > 0.2:
            hip_mid = (LSH[1] + LAK[1]) / 2
            if LHP[1] > hip_mid + 0.05:
                errors.append("Hips are sagging"); score -= 25
            else:
                errors.append("Hips too high"); score -= 20

        # 팔꿈치-어깨 정렬 체크
        if abs(LSH[0] - LEL[0]) > 0.1:
            errors.append("Elbows under shoulders"); score -= 15

        return max(0, score), errors
    
    def check_lunge(self, landmarks) -> Tuple[float, List[str]]:
        errors, score = [], 100.0
        LSH = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        LHP = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP.value)
        RHP = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP.value)
        LK = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE.value)
        RK = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE.value)
        LA = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE.value)
        RA = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ANKLE.value)

        # 앞/뒤 무릎 각도 체크 
        front_knee_angle = self.calculate_angle(LHP, LK, LA)
        back_knee_angle  = self.calculate_angle(RHP, RK, RA)
        if front_knee_angle < 75 or front_knee_angle > 105:
            errors.append("Front knee around 90°"); score -= 15
        if back_knee_angle < 75 or back_knee_angle > 105:
            errors.append("Bend back knee"); score -= 15

        # 상체 직립 체크
        mid_hip = [(LHP[i] + RHP[i])/2 for i in range(3)]
        torso_vertical = abs(LSH[0] - mid_hip[0])
        if torso_vertical > 0.1:
            errors.append("Keep torso upright"); score -= 15

        # 복합 에러 페널티
        if len(errors) >= 3:
            score -= 10

        return max(0, score), errors
    
    def check_crunch(self, landmarks) -> Tuple[float, List[str]]:
        errors, score = [], 100.0
        LSH = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        LHP = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP.value)
        LK  = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE.value)
        LA  = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE.value)
        NO  = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.NOSE.value)

        # 무릎 각도 체크
        knee_angle = self.calculate_angle(LHP, LK, LA)
        if knee_angle < 85 or knee_angle > 120:
            errors.append("Bend knees to ~90°"); score -= 15

        # 상체 각도로 상태 전환
        torso_angle = self.calculate_angle(NO, LSH, LHP)
        UP_ENTER, DOWN_ENTER = 110.0, 145.0
        if torso_angle <= UP_ENTER:
            self.exercise_state = "up"
        elif torso_angle >= DOWN_ENTER:
            self.exercise_state = "down"

        # UP 상태에서 충분히 올리지 않으면 감점
        if self.exercise_state == "up" and torso_angle > 115:
            errors.append("Lift torso higher"); score -= 15
        # 중간 범위에서도 체크
        elif 120 < torso_angle < 145:
            errors.append("Lift torso higher"); score -= 10

        return max(0, score), errors
    
    def add_feedback(self, message: str):
        current_time = time.time()
        if current_time - self.last_feedback_time > self.feedback_cooldown:
            self.feedback_messages.append(message)
            self.last_feedback_time = current_time
            if len(self.feedback_messages) > 10:
                self.feedback_messages.pop(0)
    
    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        score = 0
        errors = []
        
        if results.pose_landmarks:
            self.last_pose_detected = True
            smoothed_landmarks = self.smooth_landmarks(results.pose_landmarks)
            self.mp_drawing.draw_landmarks(
                image,
                smoothed_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
            if self.exercise_mode == "squat":
                score, errors = self.check_squat(smoothed_landmarks)
            elif self.exercise_mode == "pushup":
                score, errors = self.check_pushup(smoothed_landmarks)
            elif self.exercise_mode == "plank":
                score, errors = self.check_plank(smoothed_landmarks)
            elif self.exercise_mode == "lunge":
                score, errors = self.check_lunge(smoothed_landmarks)
            elif self.exercise_mode == "crunch":
                score, errors = self.check_crunch(smoothed_landmarks)
            
            for error in errors:
                self.add_feedback(error)
            
            if self.prev_exercise_state == "down" and self.exercise_state == "up":
                self.rep_count += 1
            self.prev_exercise_state = self.exercise_state
        else:
            self.last_pose_detected = False
        
        self.last_score = int(score)
        self.last_errors = list(errors)
        self.draw_ui(image, score, errors)
        self.frame_count += 1
        return image
    
    def draw_ui(self, image, score, errors):
        h, w = image.shape[:2]
        
        # 상단 바
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        
        cv2.putText(image, f"Exercise: {self.exercise_mode.upper()}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        score_color = (0, 255, 0) if score > 80 else (0, 165, 255) if score > 60 else (0, 0, 255)
        cv2.putText(image, f"Score: {int(score)}", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, score_color, 2)
        cv2.putText(image, f"Reps: {self.rep_count}", 
                    (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f"State: {self.exercise_state}", 
                    (w - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 하단 피드백 박스
        overlay = image.copy()
        cv2.rectangle(overlay, (0, h - 320), (w, h - 140), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
        y_offset = h - 300
        cv2.putText(image, "Feedback:", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        items = []
        for msg in self.feedback_messages[-3:]:
            y_offset += 32
            items.append((f"- {msg}", (20, y_offset)))
        if not items:
            y_offset += 32
            default_msg = "Good posture" if self.last_pose_detected else "No person detected"
            items.append((f"- {default_msg}", (20, y_offset)))
        for text, (x, y) in items:
            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(image, "Press: 1=Squat 2=Pushup 3=Plank 4=Lunge 5=Crunch Q=Quit", 
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def analyze_video(input_path: str, output_path: str = None, display: bool = True,
                  initial_mode: str = "squat", max_process_width: int = 960) -> int:
    if not os.path.exists(input_path):
        print(f"[ERROR] 입력 영상이 존재하지 않습니다: {input_path}")
        return 1

    corrector = VideoPoseCorrector()
    if initial_mode in {"squat", "pushup", "plank", "lunge", "crunch"}:
        corrector.exercise_mode = initial_mode

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없습니다: {input_path}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None

    print("=" * 60)
    print("Video-based Pose Correction Analysis")
    print("=" * 60)
    print(f"Input: {input_path}")
    if output_path: print(f"Output: {output_path}")
    print(f"Mode: {corrector.exercise_mode}")
    print(f"Max process width: {max_process_width}")
    print(f"Display: {'ON' if display else 'OFF'}")
    print("=" * 60)

    # 통계 수집
    scored_frames = 0
    score_sum = 0.0
    issue_counts = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            if max_process_width and w > max_process_width:
                scale = max_process_width / float(w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            output = corrector.process_frame(frame)
            
            # 통계 집계
            if corrector.last_pose_detected:
                scored_frames += 1
                score_sum += float(corrector.last_score)
                for msg in corrector.last_errors:
                    issue_counts[msg] = issue_counts.get(msg, 0) + 1

            if output_path and writer is None:
                oh, ow = output.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v") if output_path.lower().endswith(".mp4") \
                         else cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (ow, oh))
                if not writer.isOpened():
                    print(f"[WARN] 출력 파일을 열 수 없어 저장을 비활성화합니다: {output_path}")
                    writer = None

            if writer is not None:
                writer.write(output)

            if display:
                cv2.imshow("AI Pose Corrector (Video)", output)
                key = cv2.waitKey(int(1000 / max(1.0, fps))) & 0xFF
                if key == ord('q'): break
                elif key == ord('1'): corrector.exercise_mode="squat";  corrector.rep_count=0; corrector.exercise_state=corrector.prev_exercise_state="up"
                elif key == ord('2'): corrector.exercise_mode="pushup"; corrector.rep_count=0; corrector.exercise_state=corrector.prev_exercise_state="up"
                elif key == ord('3'): corrector.exercise_mode="plank";  corrector.rep_count=0; corrector.exercise_state=corrector.prev_exercise_state="up"
                elif key == ord('4'): corrector.exercise_mode="lunge";  corrector.rep_count=0; corrector.exercise_state=corrector.prev_exercise_state="up"
                elif key == ord('5'): corrector.exercise_mode="crunch"; corrector.rep_count=0; corrector.exercise_state=corrector.prev_exercise_state="up"

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    # 요약 출력
    avg_score = (score_sum / scored_frames) if scored_frames else 0.0
    print("\n" + "=" * 60)
    print("Video Analysis Summary")
    print("=" * 60)
    print(f"Average Score: {avg_score:.1f}")
    print(f"Total Reps: {corrector.rep_count}")
    if issue_counts:
        print("Top Issues:")
        for k, v in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {k}: {v} times")
    else:
        print("Top Issues: None")
    print("=" * 60)

    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze a video file with pose correction")
    parser.add_argument("input", help="입력 영상 경로 (예: input.mp4)")
    parser.add_argument("--output", "-o", default=None, help="분석 결과 저장 경로 (예: output.mp4)")
    parser.add_argument("--no-display", action="store_true", help="화면 표시 비활성화")
    parser.add_argument("--mode", choices=["squat", "pushup", "plank", "lunge", "crunch"], default="squat", help="초기 운동 모드")
    parser.add_argument("--max-process-width", type=int, default=960, help="처리용 프레임 최대 너비")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    display = not args.no_display
    sys.exit(analyze_video(
        args.input,
        args.output,
        display,
        args.mode,
        max_process_width=args.max_process_width,
    ))


if __name__ == "__main__":
    main()
