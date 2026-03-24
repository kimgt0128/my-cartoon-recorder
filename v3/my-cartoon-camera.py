import cv2
import numpy as np
import os

def apply_cell_animation_effect(img):
    """
    짱구, 도라에몽 등 셀 애니메이션 특유의 평면적이고 화사한 파스텔 색감을 구현하며,
    얇고 선명한 펜선을 추출하여 테두리를 뚜렷하게 만듭니다.
    """
    # ---------------------------------------------------------
    # 1. 해상도 축소 (연산 속도 향상 및 수채화 번짐 효과 유도)
    # ---------------------------------------------------------
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)

    # ---------------------------------------------------------
    # 2. 강력한 색감 부스팅 (통통 튀는 애니메이션 파스텔톤)
    # ---------------------------------------------------------
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    
    s_ch = cv2.add(s_ch, 40) 
    v_ch = cv2.add(v_ch, 20) 
    
    hsv_boost = cv2.merge((h_ch, s_ch, v_ch))
    boosted = cv2.cvtColor(hsv_boost, cv2.COLOR_HSV2BGR)

    # ---------------------------------------------------------
    # 3. K-Means 클러스터링 (색상 양자화)
    # ---------------------------------------------------------
    Z = boosted.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    K = 14
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center)
    res = center[label.flatten()]
    quantized = res.reshape((boosted.shape))

    # ---------------------------------------------------------
    # 4. 부드러운 혼합 및 원본 크기 복구
    # ---------------------------------------------------------
    color_small = cv2.bilateralFilter(quantized, 9, 75, 75)
    color_large = cv2.resize(color_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # ---------------------------------------------------------
    # 5. [개선됨] 얇고 선명한 스케치 펜선 추출 (얼굴 윤곽, 테두리 강조)
    # ---------------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1차 필터링: 진짜 테두리는 살리고 미세한 질감만 녹여서 잡선을 억제합니다.
    gray_blur = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 2차 필터링: 자잘한 점 노이즈 제거 (선을 얇게 유지하기 위해 커널 크기를 5로 줄임)
    gray_blur = cv2.medianBlur(gray_blur, 5)
    
    # MEAN_C 대신 GAUSSIAN_C를 사용하여 펜선을 훨씬 얇고 세밀하게 따냅니다.
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # 부드럽고 얇은 선 생성
        cv2.THRESH_BINARY, 
        13, 2 # 여기서 5(상수 C)를 키우면 선이 얇아지고 덜 나타나며, 줄이면 선이 많아집니다.
    )

    # ---------------------------------------------------------
    # 6. 최종 합성
    # ---------------------------------------------------------
    cartoon = cv2.bitwise_and(color_large, color_large, mask=edges)
    return cartoon

def main():
    current_file_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file_path)
    save_dir = os.path.join(script_dir, "result")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    print("=====================================")
    print(" 📺 리얼 셀 애니메이션 카메라 (v2) 실행!")
    print(f" - 정확한 저장 위치: {save_dir}")
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
            print("사진 촬영 중 (연산 진행 중)...")
            
            flash_frame = np.full(frame.shape, 255, dtype=np.uint8)
            cv2.imshow('Live Webcam', flash_frame)
            cv2.waitKey(100) 
            
            anime_img = apply_cell_animation_effect(frame)
            
            capture_count += 1
            filename = os.path.join(save_dir, f"anime_result_{capture_count}.jpg")
            cv2.imwrite(filename, anime_img)
            
            print(f">>> 성공적으로 저장되었습니다: {filename}")
            cv2.imshow('Cell Animation Result', anime_img)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()