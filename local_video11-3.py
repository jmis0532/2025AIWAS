# -*- coding: utf-8 -*-
# 2025/09/25 Summer 版（雙模型＋雙 JSON，D 鍵切換模型/配置）
# - 預載 2 個 .pt 模型 + 2 份 classes_config.json
# - 播放時按 D 切換模型/配置；暫停時按 D 仍為「下一幀」
# - R 切換類別：ALL(None) → 0 → 1 → … → N-1 → ALL(None)
# - per-class 文字大小/粗細/顏色/置信度門檻與框線厚度
# - 影片循環播放、逐幀控制（SPACE/Q/A/D）、W1/W2 分數、距底排序
# - 中文控制說明（PIL + TTF 可用則中文，否則英文）
# - 不支援 .engine 直接載入（會明確報錯）

import os
import cv2
import math
import json
import yaml
import numpy as np
import supervision as sv
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

# ======= 使用者可調參數 =======
MODEL_PATHS  = [
    #"yolo11x.pt",
    #"yolo11x.pt",
    r"C:/Users/User/Desktop/AIWASPT/CBA/1004best.pt",
    r"C:/Users/User/Desktop/AIWASPT/WRS/0909best.pt",
]
CONFIG_PATHS = [
    #"", ""
    r"C:/Users/User/Desktop/AIWASPT/JSON/classes_config_CBA.json",
    r"C:/Users/User/Desktop/AIWASPT/JSON/classes_config_WRS.json",
]

MM_PER_PIXEL = 0.5
W2_WEIGHT = 3
TOP_K = 10
LOOP_PLAYBACK = True
WINDOW_TITLE = "YOLO Video Detection"

# 中文控制說明（未暫停：D 切換模型；暫停：D 下一幀）
TEXT_HINT_ZH = "SPACE 暫停/播放 | Q 第一幀 | A 上一幀 | D 切換模型(播放中)/下一幀(暫停) | R 切換類別 | ESC 離開"
TEXT_HINT_EN = "SPACE Pause/Play | Q First | A Prev | D Switch Model(playing)/Next(paused) | R Class Cycle | ESC Exit"

# Windows 常見中文字型；若不存在，會自動改用英文提示
FONT_PATH = r"C:/Windows/Fonts/msjh.ttc"
# ============================

# ======= 全域狀態 =======
CLASS_FILTER = None          # None = ALL；int = 只顯示該類
NUM_CLASSES = None
LAST_NAMES = []
CFG = None                   # 目前啟用的 config（會跟著模型切換）
CURRENT_MODEL_IDX = 0
CURRENT_MODEL_NAME = ""
# ========================

# 嘗試載入 PIL 以顯示中文（僅用於控制說明，不影響標籤字）
try:
    from PIL import ImageFont, ImageDraw, Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

def file_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False

def hex_to_bgr(hex_color: str):
    if not isinstance(hex_color, str):
        return (0, 255, 0)
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (0, 255, 0)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

def load_config_from(path: str):
    cfg = None
    if file_exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    if not cfg:
        cfg = {
            "names": [], "colors": [], "text_colors": [], "conf": [], "thickness": [],
            "font_scales": [], "font_thicks": [],
            "default_conf": 0.25, "default_thickness": 2,
            "font_scale": 0.8, "font_thick": 2
        }
    cfg["_colors_bgr"] = [hex_to_bgr(c) for c in cfg.get("colors", [])]
    cfg["_text_colors_bgr"] = [hex_to_bgr(c) for c in cfg.get("text_colors", [])]
    return cfg

def load_all_configs(paths):
    return [load_config_from(p) for p in paths]

def try_load_names_from_files():
    for fname in ("data.yaml", "classes.txt"):
        if file_exists(fname):
            if fname.endswith(".yaml"):
                try:
                    with open(fname, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        if "names" in data and isinstance(data["names"], (list, dict)):
                            if isinstance(data["names"], dict):
                                return [data["names"][i] for i in sorted(data["names"].keys(), key=int)]
                            return data["names"]
                        if "class_names" in data and isinstance(data["class_names"], list):
                            return data["class_names"]
                except Exception:
                    pass
            else:
                try:
                    with open(fname, "r", encoding="utf-8") as f:
                        names = [line.strip() for line in f if line.strip()]
                    if names:
                        return names
                except Exception:
                    pass
    return None

def resolve_names(results_obj, model_obj):
    names = getattr(results_obj, "names", None)
    if names and isinstance(names, (list, dict)):
        if isinstance(names, dict):
            try:
                return [names[i] for i in sorted(names.keys(), key=int)]
            except Exception:
                pass
        else:
            return names
    names = getattr(model_obj, "names", None)
    if names and isinstance(names, (list, dict)):
        if isinstance(names, dict):
            try:
                return [names[i] for i in sorted(names.keys(), key=int)]
            except Exception:
                pass
        else:
            return names
    names = try_load_names_from_files()
    if names:
        return names
    # 從模型 nc 退路
    try:
        nc = int(getattr(getattr(model_obj, "model", None), "nc", 0)) or 0
        if nc > 0:
            return [str(i) for i in range(nc)]
    except Exception:
        pass
    return []

def name_of(cid, names_list):
    if isinstance(cid, int) and names_list and 0 <= cid < len(names_list):
        return names_list[cid]
    return str(cid)

def get_conf_threshold(cid: int):
    arr = CFG.get("conf", [])
    if isinstance(cid, int) and 0 <= cid < len(arr):
        return float(arr[cid])
    return float(CFG.get("default_conf", 0.25))

def get_thickness(cid: int):
    arr = CFG.get("thickness", [])
    if isinstance(cid, int) and 0 <= cid < len(arr):
        return int(arr[cid])
    return int(CFG.get("default_thickness", 2))

def get_color_bgr(cid: int):
    arr = CFG.get("_colors_bgr", [])
    if isinstance(cid, int) and 0 <= cid < len(arr):
        return tuple(arr[cid])
    return (0, 255, 0)

def get_text_color_bgr(cid: int):
    arr = CFG.get("_text_colors_bgr", [])
    if isinstance(cid, int) and 0 <= cid < len(arr):
        return tuple(arr[cid])
    return (255, 255, 255)

def get_font_scale(cid: int):
    arr = CFG.get("font_scales", [])
    if isinstance(cid, int) and 0 <= cid < len(arr):
        try:
            return float(arr[cid])
        except Exception:
            pass
    return float(CFG.get("font_scale", 0.8))

def get_font_thick(cid: int):
    arr = CFG.get("font_thicks", [])
    if isinstance(cid, int) and 0 <= cid < len(arr):
        try:
            return int(arr[cid])
        except Exception:
            pass
    return int(CFG.get("font_thick", 2))

FONT = cv2.FONT_HERSHEY_SIMPLEX

def has_valid_font(path: str) -> bool:
    return isinstance(path, str) and len(path) > 0 and file_exists(path)

USE_PIL_CHINESE = False
PIL_FONT = None
if PIL_AVAILABLE and has_valid_font(FONT_PATH):
    try:
        PIL_FONT = ImageFont.truetype(FONT_PATH, 20)  # 控制說明用
        USE_PIL_CHINESE = True
    except Exception:
        PIL_FONT = None
        USE_PIL_CHINESE = False

def draw_text(img_bgr: np.ndarray, text: str, org_xy: tuple, color_bgr=(220, 220, 220), font_scale=0.7, thickness=1):
    if USE_PIL_CHINESE and PIL_FONT is not None:
        img_pil = Image.fromarray(img_bgr)
        draw = ImageDraw.Draw(img_pil)
        try:
            ascent, descent = PIL_FONT.getmetrics()
            y_offset = ascent
        except Exception:
            y_offset = int(20 * font_scale)
        x, y = org_xy
        y = y - y_offset + 2
        color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        draw.text((x, y), text, fill=color_rgb, font=PIL_FONT)
        return np.array(img_pil)
    else:
        cv2.putText(img_bgr, text, org_xy, FONT, font_scale, color_bgr, thickness, cv2.LINE_AA)
        return img_bgr

def load_model(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".engine":
        raise RuntimeError("目前此程式未直接支援 .engine。請改用 .pt 或 .onnx，或改用 TensorRT runtime。")
    try:
        return YOLO(path)  # 支援 .pt / .onnx
    except Exception as e:
        raise RuntimeError(f"載入 YOLO 模型失敗：{e}")

def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("無法開啟影片")
        raise SystemExit
    return cap

def run_inference(model, frame):
    try:
        return model(frame, verbose=False)
    except TypeError:
        # 舊版 Ultralytics 可能需要 task 指定
        return model(frame, task="detect", verbose=False)

def process_frame(frame, frame_index, total_frames, model, class_filter=None):
    global NUM_CLASSES, LAST_NAMES

    results_list = run_inference(model, frame)
    result = results_list[0]
    names = resolve_names(result, model)
    LAST_NAMES = names[:] if isinstance(names, list) else []

    # 更穩的 NUM_CLASSES 推導：names → model.model.nc → result.boxes → CFG
    if NUM_CLASSES is None:
        try:
            if isinstance(names, list) and len(names) > 0:
                NUM_CLASSES = len(names)
            else:
                nc_try = int(getattr(getattr(model, "model", None), "nc", 0)) or 0
                if nc_try > 0:
                    NUM_CLASSES = nc_try
                else:
                    if hasattr(result, "boxes") and hasattr(result.boxes, "cls") and result.boxes.cls is not None and result.boxes.cls.numel() > 0:
                        NUM_CLASSES = int(result.boxes.cls.max().item()) + 1
                    else:
                        NUM_CLASSES = len(CFG.get("names", [])) or 1
        except Exception:
            NUM_CLASSES = len(CFG.get("names", [])) or 1

    detections = sv.Detections.from_ultralytics(result)

    # 依 conf 與 class_filter 篩選
    keep_indices = []
    for i in range(len(detections)):
        cid = int(detections.class_id[i]) if detections.class_id is not None else -1
        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
        if conf >= get_conf_threshold(cid):
            if (class_filter is None) or (cid == class_filter):
                keep_indices.append(i)

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cross_len = 40
    annotated = frame.copy()

    # 距底排序（排除 "WK" 不參與紅色攻擊順序）
    bottom_distances = []
    for i in keep_indices:
        x1, y1, x2, y2 = map(int, detections.xyxy[i])
        bottom_y = max(y1, y2)
        distance_to_bottom = max(0, h - bottom_y)
        cname = name_of(int(detections.class_id[i]), names)
        if cname != "WK":
            bottom_distances.append((i, distance_to_bottom))

    sorted_by_bottom = sorted(bottom_distances, key=lambda x: x[1])
    topk = sorted_by_bottom[:max(0, min(TOP_K, len(sorted_by_bottom)))]
    idx_to_rank = {idx: rank + 1 for rank, (idx, _) in enumerate(topk)}
    idx_to_w1   = {idx: len(topk) - rank for rank, (idx, _) in enumerate(topk)}

    for i in keep_indices:
        x1, y1, x2, y2 = map(int, detections.xyxy[i])
        cid = int(detections.class_id[i]) if detections.class_id is not None else -1
        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
        cname = name_of(cid, names)

        # 框線
        color = get_color_bgr(cid)
        thick = get_thickness(cid)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)

        # 標籤（per-class 字體大小/粗細）
        label = f"{cname} {conf:.2f}"
        label_scale = get_font_scale(cid)
        label_thick = get_font_thick(cid)
        (tw, th), _ = cv2.getTextSize(label, FONT, label_scale, label_thick)
        y_text = max(0, y1 - 8)
        cv2.rectangle(annotated, (x1, y_text - th - 4), (x1 + tw + 4, y_text), color, -1)
        text_color = get_text_color_bgr(cid)
        cv2.putText(annotated, label, (x1 + 2, y_text - 2), FONT, label_scale, text_color, label_thick, cv2.LINE_AA)

        # 目標中心 + 連線 + 十字
        bx = (x1 + x2) // 2
        by = (y1 + y2) // 2
        cv2.line(annotated, (cx, cy), (bx, by), (0, 125, 0), 2)
        cv2.line(annotated, (bx - cross_len, by), (bx + cross_len, by), (255, 0, 0), 1)
        cv2.line(annotated, (bx, by - cross_len), (bx, by + cross_len), (255, 0, 0), 1)

        # 幾何資訊 + W 分數
        pixel_distance = math.hypot(bx - cx, by - cy)
        mm_distance = pixel_distance * MM_PER_PIXEL
        angle_deg = math.degrees(math.atan2(cy - by, bx - cx))
        if angle_deg < 0: angle_deg += 360
        mid_x = (cx + bx) // 2
        mid_y = (cy + by) // 2

        w1 = idx_to_w1.get(i, 0)
        w2 = W2_WEIGHT if cname == "WK" else 0
        total_score = w1 + w2

        cv2.putText(annotated, f"{mm_distance:.1f} mm", (mid_x + 5, mid_y - 20), FONT, 0.5, (200, 200, 200), 1)
        cv2.putText(annotated, f"{angle_deg:.1f} deg", (mid_x + 5, mid_y - 5), FONT, 0.5, (160, 160, 160), 1)
        cv2.putText(annotated, f"W1:{w1} W2:{w2} W:{total_score}", (mid_x + 5, mid_y + 15), FONT, 0.5, (0, 255, 255), 1)

        # 紅色攻擊順序（框右上角）
        rank = idx_to_rank.get(i, None)
        if rank is not None:
            (tw2, th2), _ = cv2.getTextSize(str(rank), FONT, 0.9, 2)
            pos_x = x2 - tw2 - 2
            pos_y = max(y1 + th2 + 2, th2 + 2)
            cv2.putText(annotated, str(rank), (pos_x, pos_y), FONT, 0.9, (0, 0, 255), 2)

    # 畫面中心紅十字
    cv2.line(annotated, (cx - cross_len, cy), (cx + cross_len, cy), (0, 0, 255), 1)
    cv2.line(annotated, (cx, cy - cross_len), (cx, cy + cross_len), (0, 0, 255), 1)

    # 幀資訊
    total_str = f"{total_frames}" if (isinstance(total_frames, int) and total_frames > 0) else "?"
    frame_info = f"Frame: {frame_index} / {total_str}"
    cv2.putText(annotated, frame_info, (10, 22), FONT, 0.65, (255, 255, 255), 1)

    # 目前模型提示
    try:
        model_text = f"Model[{CURRENT_MODEL_IDX}]: {CURRENT_MODEL_NAME}"
    except Exception:
        model_text = f"Model[{CURRENT_MODEL_IDX}]"
    cv2.putText(annotated, model_text, (10, 44), FONT, 0.65, (180, 255, 180), 1)

    # 類別濾鏡提示
    if class_filter is None:
        filt_text = "Class: ALL"
    else:
        cname = name_of(class_filter, names) if names else str(class_filter)
        filt_text = f"Class: {cname} ({class_filter})"
    cv2.putText(annotated, filt_text, (10, 64), FONT, 0.65, (180, 220, 255), 1)

    # 控制說明（中文或英文）
    hint_text = TEXT_HINT_ZH if (USE_PIL_CHINESE and (PIL_FONT is not None)) else TEXT_HINT_EN
    annotated = draw_text(annotated, hint_text, (10, 86), (220, 220, 220), 0.7, 1)

    return annotated

def main():
    global CLASS_FILTER, NUM_CLASSES, LAST_NAMES, CFG
    global CURRENT_MODEL_IDX, CURRENT_MODEL_NAME

    # 檔案對數檢查
    if len(MODEL_PATHS) != 2 or len(CONFIG_PATHS) != 2:
        raise RuntimeError("請提供兩個 MODEL_PATHS 與兩個 CONFIG_PATHS。")

    # 預載兩個模型
    models = []
    for p in MODEL_PATHS:
        models.append(load_model(p))

    # 預載兩份 config
    CFG_LIST = load_all_configs(CONFIG_PATHS)

    # 初始選用第 0 組
    current_model_idx = 0
    model = models[current_model_idx]
    CFG = CFG_LIST[current_model_idx]

    # 初始化 NUM_CLASSES
    try:
        if isinstance(getattr(model, "names", None), dict):
            NUM_CLASSES = len(model.names)
        elif isinstance(getattr(model, "names", None), list):
            NUM_CLASSES = len(model.names)
        else:
            NUM_CLASSES = int(getattr(getattr(model, "model", None), "nc", 0)) or None
    except Exception:
        NUM_CLASSES = None

    # 顯示用名稱
    CURRENT_MODEL_IDX = current_model_idx
    CURRENT_MODEL_NAME = os.path.basename(MODEL_PATHS[current_model_idx])

    # 選檔
    root = tk.Tk(); root.withdraw()
    video_path = filedialog.askopenfilename(
        initialdir=r'C:/Users/User/Desktop/yolotestvedio',
        title="選擇影片",
        filetypes=[("All Files", "*.*")]
    )
    if not video_path:
        print("未選擇影片，結束程式")
        return

    cap = open_video(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = -1

    paused = False
    current_frame = 0
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if LOOP_PLAYBACK:
                    print("== 影片到尾，從頭繼續 ==")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    current_frame = 0
                    continue
                else:
                    print("影片播放完畢")
                    break
            current_frame += 1
            annotated_image = process_frame(frame, current_frame, total_frames, model, CLASS_FILTER)
            cv2.imshow(WINDOW_TITLE, annotated_image)

        k = cv2.waitKey(0 if paused else 1) & 0xFF
        if k == 27:  # ESC
            print("按下 ESC，結束播放")
            break
        elif k == ord(' '):
            paused = not paused
            print("== 暫停播放 ==" if paused else "== 繼續播放 ==")

        # 播放中：D 切換模型/配置
        if not paused and k in (ord('d'), ord('D')):
            current_model_idx = 1 - current_model_idx
            model = models[current_model_idx]
            CFG = CFG_LIST[current_model_idx]

            # 更新 NUM_CLASSES / LAST_NAMES
            try:
                if isinstance(getattr(model, "names", None), dict):
                    NUM_CLASSES = len(model.names)
                    LAST_NAMES[:] = [model.names[i] for i in sorted(model.names.keys(), key=int)]
                elif isinstance(getattr(model, "names", None), list):
                    NUM_CLASSES = len(model.names)
                    LAST_NAMES[:] = model.names[:]
                else:
                    nc_try = int(getattr(getattr(model, "model", None), "nc", 0)) or 0
                    if nc_try > 0:
                        NUM_CLASSES = nc_try
            except Exception:
                pass

            # 顯示用名稱
            CURRENT_MODEL_IDX = current_model_idx
            CURRENT_MODEL_NAME = os.path.basename(MODEL_PATHS[current_model_idx])

            # 若擔心類別索引在新模型越界，可解開下行重設為 ALL：
            # CLASS_FILTER = None

            print(f"== 已切換模型到 [{CURRENT_MODEL_IDX}]: {CURRENT_MODEL_NAME} ==")
            continue

        # 暫停時：逐幀控制（A/Q/D）
        elif paused:
            if k in (ord('a'), ord('A')):  # 上一幀
                target = max(0, current_frame - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, frame = cap.read()
                if ret:
                    current_frame = target + 1
                    annotated_image = process_frame(frame, current_frame, total_frames, model, CLASS_FILTER)
                    cv2.imshow(WINDOW_TITLE, annotated_image)
                    print(f"== 回上一幀: {current_frame} ==")

            elif k in (ord('d'), ord('D')):  # 下一幀
                ret, frame = cap.read()
                if ret:
                    current_frame += 1
                    annotated_image = process_frame(frame, current_frame, total_frames, model, CLASS_FILTER)
                    cv2.imshow(WINDOW_TITLE, annotated_image)
                    print(f"== 下一幀: {current_frame} ==")
                else:
                    if LOOP_PLAYBACK:
                        print("== 下一幀到尾，從頭繼續 ==")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        current_frame = 0
                    else:
                        print("已到影片末尾")

            elif k in (ord('q'), ord('Q')):  # 回到第一幀
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if ret:
                    current_frame = 1
                    annotated_image = process_frame(frame, current_frame, total_frames, model, CLASS_FILTER)
                    cv2.imshow(WINDOW_TITLE, annotated_image)
                    print("== 回到第一幀 ==")

        # 類別濾鏡切換（播放或暫停皆可）
        if k in (ord('r'), ord('R')):
            # 確保 NUM_CLASSES 有值
            if NUM_CLASSES is None:
                try:
                    if isinstance(getattr(model, "names", None), dict):
                        NUM_CLASSES = len(model.names)
                    elif isinstance(getattr(model, "names", None), list):
                        NUM_CLASSES = len(model.names)
                    else:
                        NUM_CLASSES = int(getattr(getattr(model, "model", None), "nc", 0)) or 1
                except Exception:
                    NUM_CLASSES = 1

            # 循環 ALL(None) → 0 → 1 → … → N-1 → ALL(None)
            if CLASS_FILTER is None:
                CLASS_FILTER = 0
            else:
                CLASS_FILTER = CLASS_FILTER + 1
                if CLASS_FILTER >= max(NUM_CLASSES, 1):
                    CLASS_FILTER = None

            cname = name_of(CLASS_FILTER, LAST_NAMES) if CLASS_FILTER is not None else "ALL"
            print(f"== 切換類別 == 現在只顯示：{cname} ({CLASS_FILTER if CLASS_FILTER is not None else 'ALL'}) ==")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
