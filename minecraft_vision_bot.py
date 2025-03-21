import numpy as np
import win32gui, win32ui, win32con
import cv2 as cv
from ultralytics import YOLO


class WindowCapture:
    def __init__(self, window_name):
        # Find the specified window by name
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception(f"Window not found: {window_name}")

        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        # Adjust for borders/titlebar if needed
        border_pixels = 8
        titlebar_pixels = 30
        self.w -= (border_pixels * 2)
        self.h -= (border_pixels + titlebar_pixels)
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

    def get_screenshot(self):
        # Capture the screenshot of the window
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h),
                   dcObj, (self.cropped_x, self.cropped_y),
                   win32con.SRCCOPY)

        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        # Cleanup
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # Drop alpha: now we have BGR
        img = img[:, :, :3]

        # Ensure array is contiguous for OpenCV
        img = np.ascontiguousarray(img)

        return img


def main():
    # Find the Minecraft window
    window_name = "Minecraft* 1.21.4 - Singleplayer"  # Example
    wincap = WindowCapture(window_name)

    # Load the YOLO model
    MODEL_PATH = "runs/detect/train35/weights/best.pt"
    model = YOLO(MODEL_PATH)

    while True:
        frame = wincap.get_screenshot()

        # Perform YOLO inference
        results = model.predict(frame, verbose=False)

        # Draw bounding boxes with confidence
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                class_idx = int(box.cls)
                label = results[0].names.get(class_idx, "unknown")
                confidence = float(box.conf[0]) * 100  # Convert to percentage

                # Draw the bounding box
                cv.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                # Display the label and confidence
                label_text = f"{label} ({confidence:.1f}%)"
                cv.putText(frame, label_text, (xyxy[0], xyxy[1] - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the detections
        cv.imshow("Minecraft Detections", frame)

        # Exit on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
