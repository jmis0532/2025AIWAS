"""
2025/10/10 by Summer Lee 
所有訓練資料固定放在桌面trainingTEST資料夾
訓練內容 YOLOv11m（imgsz=960）
- 自動判斷 pt / yaml 與 pretrained 的合理搭配
- 啟用驗證與早停，穩定輸出 best.pt
"""

import sys
from pathlib import Path

# ====== 可調整超參數區 ======
DATA_YAML = "C:/Users/User/Desktop/trainingTEST/data.yaml"

# 用官方權重： "yolo11m.pt" or "yolo26.pt" -- 2026/1/15
# 從零開始：   "yolo11m.yaml"
MODEL_WEIGHTS = "yolo11m.pt"

IMG_SIZE   = 960 # 約佔3080ti 10.2GB 不會OOM"
EPOCHS     = 1
BATCH      = 8
SEED       = 42
WORKERS    = 0           # Windows 建議 0，穩定最重要
PROJECT    = "runs/detect"
NAME       = "train_test"
DEVICE     = 0           # 單卡 0；沒 GPU 可改 "cpu"
SAVE_PERIOD = -1         # -1=只存 best/last
PATIENCE   = 50          # 早停耐心值（需要 val=True 才生效）

# 是否從中斷點續訓（放入先前的 last.pt 路徑，或設 None）
RESUME_WEIGHTS = None  # 例如 r"runs/detect/train_960/weights/last.pt"

# 記憶體夠→'ram'，不夠→False；SSD 也可 'disk'
CACHE = "ram"

# 是否使用餘弦學習率
COS_LR = True

# 是否開啟混合精度（預設 True，遇到不穩可關掉）
AMP = True
# =====================

def main():
    from ultralytics import YOLO

    is_pt   = MODEL_WEIGHTS.lower().endswith(".pt")
    is_yaml = MODEL_WEIGHTS.lower().endswith(".yaml")

    # 決定是否用預訓練
    # - pt：本身就載入權重，不另外再設定 pretrained
    # - yaml：要預訓練就 True（會載入官方權重），否則 False（隨機初始化）
    if is_pt:
        pretrained_flag = None  # 交給 .pt 直接載入，不再傳 pretrained
    elif is_yaml:
        pretrained_flag = True  # 想要 from scratch -> 改為 False
    else:
        raise ValueError("MODEL_WEIGHTS 請使用 .pt 或 .yaml")

    print("===== YOLOv11m 訓練開始 =====")
    print(f"data.yaml : {DATA_YAML}")
    print(f"model     : {MODEL_WEIGHTS}")
    print(f"imgsz     : {IMG_SIZE}")
    print(f"epochs    : {EPOCHS}")
    print(f"batch     : {BATCH}")
    print(f"device    : {DEVICE}")
    print(f"cache     : {CACHE}")
    print(f"cos_lr    : {COS_LR}")
    print(f"amp       : {AMP}")
    print(f"resume    : {RESUME_WEIGHTS is not None}")
    print("============================\n")

    # 續訓模式：直接用 last.pt 建模再 train(resume=True) 最穩
    if RESUME_WEIGHTS:
        model = YOLO(RESUME_WEIGHTS)
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            device=DEVICE,
            seed=SEED,
            workers=WORKERS,
            project=PROJECT,
            name=NAME,
            exist_ok=True,
            save=True,
            save_period=SAVE_PERIOD,
            verbose=True,
            patience=PATIENCE,
            val=True,          # 要挑 best/早停，一定要驗證
            cache=CACHE,
            cos_lr=COS_LR,
            amp=AMP,
            resume=True        # 這行會接續 optimizer/scheduler 狀態
        )
    else:
        model = YOLO(MODEL_WEIGHTS)
        train_kwargs = dict(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            device=DEVICE,
            seed=SEED,
            workers=WORKERS,
            project=PROJECT,
            name=NAME,
            exist_ok=True,
            save=True,
            save_period=SAVE_PERIOD,
            verbose=True,
            patience=PATIENCE,
            val=True,          # 產生 best.pt / 啟用 early stop
            cache=CACHE,
            cos_lr=COS_LR,
            amp=AMP,
        )
        # 只有 yaml 模型才需要顯式指定 pretrained（pt 檔已經有含預訓練）
        if pretrained_flag is not None:
            train_kwargs["pretrained"] = pretrained_flag

        results = model.train(**train_kwargs)

    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        print(f"\n===== 訓練完成 =====\n輸出目錄：{save_dir}")
        w = Path(save_dir) / "weights"
        print(f"- 最佳權重：{w / 'best.pt'}")
        print(f"- 最後權重：{w / 'last.pt'}")
    else:
        print("（注意）無法取得輸出目錄")

if __name__ == "__main__":
    main()

