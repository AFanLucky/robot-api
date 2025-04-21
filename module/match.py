# 保存图片到本地
import os
import requests
from urllib.parse import urlparse
import cv2
import numpy as np
from PIL import Image,ImageDraw

def save_image(url, save_dir='saved_images'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    try:
        # 下载并保存图片
        if url and isinstance(url, str) and (url.startswith('http://') or url.startswith('https://')):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # 从URL获取文件名，如果没有则使用默认名
            filename = os.path.basename(urlparse(url).path) or 'image.jpg'
            img_path = os.path.join(save_dir, filename)
            
            with open(img_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f'图片已保存为: {img_path}')
            return img_path
    except Exception as e:
        print(f'保存图片时出错: {e}')
        return None

def edge(img):
    # print(img)
    np_data = np.array(img)
    # print(np_data)
    rr = np.where(np_data[:, :, 3] != 0)
    # print(type(rr))
    xmin = min(rr[1])
    ymin = min(rr[0])
    xmax = max(rr[1])
    ymax = max(rr[0])
    return xmin, ymin, xmax, ymax

def get_slide_in(img_path):
    img = Image.open(img_path)
    img = img.convert('RGBA')
    img_edge = edge(img)
    
    # 根据边缘裁剪图像
    cropped_img = img.crop(img_edge)
    
    # 在原图上绘制边框（用于调试）
    draw = ImageDraw.Draw(img)
    draw.rectangle(img_edge, outline="red")
    
    return {
        "slide_y_min": img_edge[1],
        "slide_y_max": img_edge[3],
        "slide_img": cropped_img  # 返回裁剪后的图像
    }

def crop_image_by_y(image_path, y_min, y_max, output_path=None):
    # 打开图像
    img = Image.open(image_path)

    # 获取图像宽度
    width = img.width

    # 裁剪图像 (左, 上, 右, 下)
    cropped_img = img.crop((0, y_min, width, y_max))

    # 保存或返回裁剪后的图像
    if output_path:
        cropped_img.save(output_path)

    return cropped_img

def sobel_edge(image):
    image_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    abs_x = cv2.convertScaleAbs(image_x)
    image_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    abs_y = cv2.convertScaleAbs(image_y)
    dst = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    return np.asarray(dst, dtype=np.uint8)

def process_cv(slide_img, bg_img):
    # 确保输入是numpy数组
    if not isinstance(slide_img, np.ndarray):
        slide_cv = np.array(slide_img)
    else:
        slide_cv = slide_img
        
    if not isinstance(bg_img, np.ndarray):
        bg_cv = np.array(bg_img)
    else:
        bg_cv = bg_img
    
    # 转换为灰度
    if len(slide_cv.shape) == 3:
        slide_gray = cv2.cvtColor(slide_cv, cv2.COLOR_BGR2GRAY if slide_cv.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    else:
        slide_gray = slide_cv
        
    if len(bg_cv.shape) == 3:
        bg_gray = cv2.cvtColor(bg_cv, cv2.COLOR_BGR2GRAY if bg_cv.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    else:
        bg_gray = bg_cv

    # 调试信息
    print(f"slide_gray shape: {slide_gray.shape}, dtype: {slide_gray.dtype}")
    print(f"bg_gray shape: {bg_gray.shape}, dtype: {bg_gray.dtype}")

    # # 应用Sobel边缘检测
    # slide_edges = sobel_edge(slide_gray)
    # bg_edges = sobel_edge(bg_gray)
    
    # 保存边缘图用于调试
    cv2.imwrite(os.path.join('saved_images', 'slide_edges_dilated.jpg'), slide_gray)
    cv2.imwrite(os.path.join('saved_images', 'bg_edges_dilated.jpg'), bg_gray)

    return slide_gray, bg_gray

def handle_calculate(bg_img, slide_img):
    print('执行了')
    try:
        # 保存图片到本地
        bg_path = save_image(bg_img, 'saved_images')
        slide_path = save_image(slide_img, 'saved_images')
        
        if not bg_path or not slide_path:
            print('图片保存失败')
            return 0

        # 获取png内的slide真实y坐标和slide
        handle_slide_res = get_slide_in(slide_path)

        # 根据slide坐标裁剪bg
        cropped_bg_img = crop_image_by_y(bg_path, handle_slide_res['slide_y_min'], handle_slide_res['slide_y_max'])
        
        # 使用cv进行图像处理 - 传入图像对象而不是文件路径
        slide_by_cv, bg_by_cv = process_cv(handle_slide_res['slide_img'], cropped_bg_img)
        
        # 执行模板匹配 - 使用不同的匹配方法并取最佳结果
        match_methods = [cv2.TM_CCOEFF, cv2.TM_CCORR_NORMED]
        best_x = 0
        best_confidence = -float('inf')
        
        for method in match_methods:
            # 对边缘图像进行模板匹配
            res = cv2.matchTemplate(bg_by_cv, slide_by_cv, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            # print(f'匹配方法 {method} - 位置: {max_loc}, 相似度: {max_val}')
            
            if max_val > best_confidence:
                best_confidence = max_val
                best_x = max_loc[0]
                best_method = method
                best_loc = max_loc
        
        print(f'最佳匹配结果 - 方法: {best_method}, x坐标: {best_x}, 相似度: {best_confidence}')

        # 在背景边缘图像上标记匹配位置
        result_edges = cv2.cvtColor(bg_by_cv, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(result_edges, best_loc,
                     (best_loc[0] + slide_by_cv.shape[1], best_loc[1] + slide_by_cv.shape[0]),
                     (0, 0, 255), 2)
        cv2.imwrite(os.path.join('saved_images', 'match_result_edges.jpg'), result_edges)

        return best_x
        
    except Exception as e:
        print(f'计算滑动距离时出错: {e}')
        import traceback
        traceback.print_exc()
        return 0