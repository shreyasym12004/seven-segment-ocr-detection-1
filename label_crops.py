"""Review crops: press 0-9 = set correct digit, Space/Enter = keep, d = delete, q = quit."""
import glob, os, shutil
import cv2

files = sorted(glob.glob("dataset/[0-9]/*.png"))
for i, f in enumerate(files):
    if not os.path.exists(f):
        continue
    guess = os.path.basename(os.path.dirname(f))
    img = cv2.resize(cv2.imread(f), (280, 280), interpolation=cv2.INTER_NEAREST)
    cv2.putText(img, f"guess {guess}  {i + 1}/{len(files)}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("crop", img)
    k = cv2.waitKey(0) & 0xFF
    if k == ord("q"):
        break
    if k == ord("d"):
        os.remove(f)
    elif ord("0") <= k <= ord("9") and chr(k) != guess:
        os.makedirs(f"dataset/{chr(k)}", exist_ok=True)
        shutil.move(f, f"dataset/{chr(k)}/{os.path.basename(f)}")
cv2.destroyAllWindows()
