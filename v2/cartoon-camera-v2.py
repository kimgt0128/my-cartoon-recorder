import cv2
import numpy as np
import os

def apply_ghibli_effect(img):
    """
    원본 이미지에 지브리 애니메이션 스타일의 렌더링 효과를 적용합니다.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    s = cv2.add(s, 30) 
    v = cv2.add(v, 15)
    
    hsv_boost = cv2.merge((h, s, v))
    img_boost = cv2.cvtColor(hsv_boost, cv2.COLOR_HSV2BGR)

    color = img_boost.copy()
    for _ in range(5): 
        color = cv2.bilateralFilter(color, 9, 100, 100)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 
        9, 5 
    )

    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def main():
    # ---------------------------------------------------------
    # [경로 자동 추적: v2 폴더 기준]
    # 현재 스크립트가 있는 폴더(v2)를 찾고 그 안에 result 폴더 지정
    # ---------------------------------------------------------
    current_file_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file_path) # v2 폴더 경로
    save_dir = os.path.join(script_dir, "result")   # v2/result
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    print("=====================================")
    print(" 🌸 지브리 스타일 카메라 (v2) 실행!")
    print(f" - 저장 위치: {save_dir}")
    print(" - [Space] 또는 [c] 키: 사진 촬영")
    print(" - [ESC] 또는 [q] 키: 프로그램 종료")
    print("=====================================")

    capture_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Live Webcam', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            print("프로그램을 종료합니다.")
            break
        
        elif key == ord('c') or key == 32:
            print("사진 촬영 중...")
            
            flash_frame = np.full(frame.shape, 255, dtype=np.uint8)
            cv2.imshow('Live Webcam', flash_frame)
            cv2.waitKey(100) 
            
            ghibli_img = apply_ghibli_effect(frame)
            
            capture_count += 1
            filename = os.path.join(save_dir, f"ghibli_result_{capture_count}.jpg")
            cv2.imwrite(filename, ghibli_img)
            
            print(f">>> 성공적으로 저장되었습니다: {filename}")
            cv2.imshow('Ghibli Result', ghibli_img)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()