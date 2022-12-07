#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import time
import copy
import argparse

import cv2 as cv
import mediapipe as mp
from model.yolox.yolox_onnx import YoloxONNX


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--skip_frame", type=int, default=0)

    parser.add_argument(
        "--model",
        type=str,
        default='model/yolox/yolox_nano.onnx',
    )
    parser.add_argument(
        '--input_shape',
        type=str,
        default="416,416",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.7,
        help='Class confidence',
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.45,
        help='NMS IoU threshold',
    )
    parser.add_argument(
        '--nms_score_th',
        type=float,
        default=0.1,
        help='NMS Score threshold',
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )

    args = parser.parse_args()

    return args


def main():
    # Analisis pengaturan #################################################################
    args = get_args()
    mp_drawing = mp.solutions.drawing_utils 
    mp_holistic = mp.solutions.holistic 
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    fps = args.fps
    skip_frame = args.skip_frame

    model_path = args.model
    input_shape = tuple(map(int, args.input_shape.split(',')))
    score_th = args.score_th
    nms_th = args.nms_th
    nms_score_th = args.nms_score_th
    with_p6 = args.with_p6

    if args.file is not None:
        cap_device = args.file

    frame_count = 0

    # Persiapan kamera ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model run #############################################################
    yolox = YoloxONNX(
        model_path=model_path,
        input_shape=input_shape,
        class_score_th=score_th,
        nms_th=nms_th,
        nms_score_th=nms_score_th,
        with_p6=with_p6,
        # providers=['CPUExecutionProvider'],
    )

    # Baca Label ###########################################################
    with open('setting/labels.csv', encoding='utf8') as f:
        labels = csv.reader(f)
        labels = [row for row in labels]
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            start_time = time.time()
            # Kamera Capture #####################################################
            ret, frame = cap.read()
            if not ret:
                continue
            debug_image = copy.deepcopy(frame)
            # BGR - RGB
            debug_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            debug_image.flags.writeable = False       
            # hasil
            results = holistic.process(debug_image)
            # print(results.face_landmarks)
            # RGB - BGR
            debug_image.flags.writeable = True   
            debug_image = cv.cvtColor(debug_image, cv.COLOR_RGB2BGR)
            # 2. Tangan Kanan
            mp_drawing.draw_landmarks(debug_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                    )
            # 3. Tangan Kiri
            mp_drawing.draw_landmarks(debug_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                    )
            # 4. Pose
            mp_drawing.draw_landmarks(debug_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )
            frame_count += 1
            if (frame_count % (skip_frame + 1)) != 0:
                continue

            # Deteksi #############################################################
            bboxes, scores, class_ids = yolox.inference(frame)

            for bbox, score, class_id in zip(bboxes, scores, class_ids):
                class_id = int(class_id) + 1
                if score < score_th:
                    continue

                # Visualisasi hasil deteksi ###################################################
                x1, y1 = int(bbox[0]), int(bbox[1])
                x2, y2 = int(bbox[2]), int(bbox[3])

                cv.putText(
                    debug_image, 'ID:' + str(class_id) + ' ' +
                    labels[class_id][0] + ' ' + '{:.3f}'.format(score),
                    (x1, y1 - 15), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    cv.LINE_AA)
                cv.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Exit(ESC) #################################################
            key = cv.waitKey(1)
            if key == 27:  # ESC
                break

            # FPS #############################################################
            elapsed_time = time.time() - start_time
            sleep_time = max(0, ((1.0 / fps) - elapsed_time))
            time.sleep(sleep_time)

            cv.putText(
                debug_image,
                "FPS:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
                (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)

            # Show #############################################################
            cv.imshow('Viskom Assignment NARUTO HandSignDetection', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
