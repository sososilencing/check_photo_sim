import torch
import clip
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from skimage.color import rgb2gray
import os

class Photo:
    model = None
    preprocess = None
    def __init__(self, image_path, model=None, preprocess=None, device=None):
        self.image_path = image_path
        self.image = Image.open(image_path).convert("RGB")
        
        # 默认 device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

         # 只加载一次模型
        if Photo.model is None or Photo.preprocess is None:
            Photo.model, Photo.preprocess = clip.load("ViT-B/32", device=self.device)

        self.model = Photo.model
        self.preprocess = Photo.preprocess
        self.feature = None

    
    def get_save_feature(self):
        # 取原图路径同目录
        img_dir ,img_name = self.get_path()
        hidden_name = f".{img_name}.pt"  # 加个点号变成隐藏文件
        feature_path = os.path.join(img_dir, hidden_name)

        if os.path.exists(feature_path):
            self.feature = torch.load(feature_path)
        else:
            # 文件不存在时，提取特征并保存
            self.encode()  # 确保特征提取
            if self.feature is None:
                raise ValueError("Feature extraction failed.")

    def save_feature(self):
        # 取原图路径同目录
        img_dir ,img_name = self.get_path()
        hidden_name = f".{img_name}.pt"  # 加个点号变成隐藏文件
        feature_path = os.path.join(img_dir, hidden_name)
        torch.save(self.feature, feature_path)

    def get_path(self):
        # 取原图路径同目录
        img_dir = os.path.dirname(self.image_path)
        img_name = os.path.basename(self.image_path)
        return img_dir, img_name
    
    def encode(self):
        if self.feature is not None:
            return  # 如果特征已经存在，不做重复处理
        # 使用 CLIP 提取图像特征
        img_tensor = self.preprocess(self.image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.feature = self.model.encode_image(img_tensor).squeeze(0)
        self.save_feature()

    def compare(self, other):
        if self.feature is None:
            self.get_save_feature()
            if self.feature is None:
                raise ValueError("Please call encode() first.")
        if other.feature is None:
            other.get_save_feature()
            if other.feature is None:
                raise ValueError("Please call encode() first.")
        # 1. CLIP 余弦相似度（特征层比较）
        cos_sim = torch.nn.functional.cosine_similarity(
            self.feature.unsqueeze(0), other.feature.unsqueeze(0)
        ).item()

        # 2. 图像尺寸对齐 + 转灰度
        img1 = self.image
        img2 = other.image.resize(img1.size)
        img1_np = np.array(img1)
        img2_np = np.array(img2)
        img1_gray = rgb2gray(img1_np)
        img2_gray = rgb2gray(img2_np)

        # 3. 各类指标
        data_range = img1_gray.max() - img1_gray.min()
        ssim_val = ssim(img1_gray, img2_gray, data_range=data_range)
        mse_val = mean_squared_error(img1_gray, img2_gray)
        psnr_val = peak_signal_noise_ratio(img1_gray, img2_gray, data_range=data_range)
        mae_val = np.mean(np.abs(img1_gray - img2_gray))

        return {
            "cosine": cos_sim,
            "ssim": ssim_val,
            "mse": mse_val,
            "psnr": psnr_val,
            "mae": mae_val
        }

    def final_score(self, other, weights=None):
        # 获取各项相似度
        scores = self.compare(other)

        # 默认权重
        if weights is None:
            weights = {
                "cosine": 0.4,
                "ssim": 0.4,
                "psnr": 0.15,
                "mae": 0.05
            }

        # 归一化处理（根据实际值调整）
        norm_cos = scores["cosine"]  # [0, 1]
        norm_ssim = scores["ssim"]   # [0, 1]
        norm_psnr = min(scores["psnr"] / 50, 1.0)  # 通常 PSNR 会在 30-50
        norm_mae = 1.0 - min(scores["mae"] / 50, 1.0)  # MAE 越小越好，反向归一化

        # 归一化后的加权分数
        final = (
            weights["cosine"] * norm_cos +
            weights["ssim"] * norm_ssim +
            weights["psnr"] * norm_psnr +
            weights["mae"] * norm_mae
        )

        return round(final * 100, 2)