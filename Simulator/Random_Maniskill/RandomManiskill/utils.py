import cv2

def center_crop_and_resize(img, target_size):
    H, W = img.shape[:2]
    short = min(H, W)
    y1 = (H - short) // 2
    x1 = (W - short) // 2
    cropped = img[y1:y1 + short, x1:x1 + short]
    if target_size is not None:
        resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return resized
    return cropped