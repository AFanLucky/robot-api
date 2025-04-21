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
        
        # 转换为OpenCV可用的格式
        slide_cv = np.array(handle_slide_res['slide_img'])  # 从PIL Image转为numpy数组
        
        # 将PIL Image转换为numpy数组
        cropped_bg_cv = np.array(cropped_bg_img)
        
        # 调试信息
        print(f"slide_cv shape: {slide_cv.shape}, dtype: {slide_cv.dtype}")
        print(f"cropped_bg_cv shape: {cropped_bg_cv.shape}, dtype: {cropped_bg_cv.dtype}")
        
        # 图像预处理 - 转为灰度
        if len(slide_cv.shape) == 3:
            slide_gray = cv2.cvtColor(slide_cv, cv2.COLOR_BGR2GRAY if slide_cv.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
        else:
            slide_gray = slide_cv
            
        if len(cropped_bg_cv.shape) == 3:
            bg_gray = cv2.cvtColor(cropped_bg_cv, cv2.COLOR_BGR2GRAY if cropped_bg_cv.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
        else:
            bg_gray = cropped_bg_cv
        
        # 应用高斯滤波去噪
        slide_blur = cv2.GaussianBlur(slide_gray, (5, 5), 0)
        bg_blur = cv2.GaussianBlur(bg_gray, (5, 5), 0)
        
        # 应用边缘检测
        slide_edges = cv2.Canny(slide_blur, 50, 150)
        bg_edges = cv2.Canny(bg_blur, 50, 150)

        # 对边缘图像进行轻微膨胀，使边缘更加明显
        kernel = np.ones((3, 3), np.uint8)
        slide_edges = cv2.dilate(slide_edges, kernel, iterations=1)
        bg_edges = cv2.dilate(bg_edges, kernel, iterations=1)
        
        cv2.imwrite(os.path.join('saved_images', 'slide_edges_dilated.jpg'), slide_edges)
        cv2.imwrite(os.path.join('saved_images', 'bg_edges_dilated.jpg'), bg_edges)

        # 执行模板匹配 - 使用不同的匹配方法并取最佳结果
        match_methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
        best_x = 0
        best_confidence = -float('inf')
        
        for method in match_methods:
            # 对边缘图像进行模板匹配
            res = cv2.matchTemplate(bg_edges, slide_edges, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            print(f'匹配方法 {method} - 位置: {max_loc}, 相似度: {max_val}')
            
            if max_val > best_confidence:
                best_confidence = max_val
                best_x = max_loc[0]
                best_method = method
                best_loc = max_loc
        
        print(f'最佳匹配结果 - 方法: {best_method}, x坐标: {best_x}, 相似度: {best_confidence}')

        # 在背景边缘图像上标记匹配位置
        result_edges = cv2.cvtColor(bg_edges, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(result_edges, best_loc,
                     (best_loc[0] + slide_edges.shape[1], best_loc[1] + slide_edges.shape[0]),
                     (0, 0, 255), 2)
        cv2.imwrite(os.path.join('saved_images', 'match_result_edges.jpg'), result_edges)


        return best_x
        
    except Exception as e:
        print(f'计算滑动距离时出错: {e}')
        import traceback
        traceback.print_exc()
        return 0