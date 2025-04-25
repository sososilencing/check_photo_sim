import os
import torch
# from fastapi import FastAPI, UploadFile, File,Form
# from fastapi.responses import JSONResponse
from pathlib import Path
# from PIL import Image
# from typing import List
# import shutil

from check_photo_similary import Photo

UPLOAD_DIR = Path("uploaded")
GALLERY_DIR = Path("gallery")
EMBEDDING_EXT = ".pth"
THRESHOLD = 60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# app = FastAPI()

# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(GALLERY_DIR, exist_ok=True)

def get_embedding_path(image_path: Path):
    return image_path.with_suffix(EMBEDDING_EXT)

def load_or_compute_photo(image_path: Path, model, preprocess) -> Photo:
    photo = Photo(str(image_path), model=model, preprocess=preprocess, device=DEVICE)
    embedding_path = get_embedding_path(image_path)

    if embedding_path.exists():
        photo.feature = torch.load(embedding_path, map_location=DEVICE)
    else:
        photo.encode()
        torch.save(photo.feature, embedding_path)

    return photo

# 加载 CLIP 模型一次
from clip import load as clip_load
model, preprocess = clip_load("ViT-B/32", device=DEVICE)

# @app.post("/compare")
# async def compare_uploaded_image(file: UploadFile = File(...), threshold: float = Form(60.0)):
#     # 保存上传图片
#     upload_path = UPLOAD_DIR / file.filename

#     print(f"收到文件: {file.filename}, 大小未知")
#     with open(upload_path, "wb") as f:
#         shutil.copyfileobj(file.file, f)


#     # 构造上传图片的 Photo 实例
#     uploaded_photo = load_or_compute_photo(upload_path, model, preprocess)
#     print(f"上传图片特征向量大小: {uploaded_photo}")
#     results = []
#     for img_path in GALLERY_DIR.rglob("*.jpg"):
#         if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
#             continue

#         gallery_photo = load_or_compute_photo(img_path, model, preprocess)
#         score = uploaded_photo.final_score(gallery_photo)

#         if score >= threshold:
#             results.append({
#                 "match": str(img_path),
#                 "similarity": round(score, 2)
#             })

#     return JSONResponse(content={"matches": results})

import os

def input_file(prompt):
    while True:
        path = input(prompt)
        if os.path.isfile(path):
            return path
        print("❌ 文件路径无效，请重新输入。")

def input_folder(prompt):
    while True:
        path = input(prompt)
        if os.path.isdir(path):
            return path
        print("❌ 文件夹路径无效，请重新输入。")

def input_threshold(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("❌ 阈值必须是数字，请重新输入。")

from tqdm import tqdm
def compare(file_path, folder_path, threshold):
    # TODO: 实现图片对比功能
    image = Photo(file_path)
    paths = collect_all_image_paths(folder_path)
    # print(paths)
    photos = [Photo(p) for p in paths]
    for photo in tqdm(photos, desc="Comparing images", unit="image"):
        score = image.final_score(photo)
        if score >= threshold:
            print(f"{image.image_path} <-> {photo.image_path} 相似度: {score:.2f}%")
            
def collect_all_image_paths(root_dir):
    from pathlib import Path
    return [str(p) for p in Path(root_dir).rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

# ../check_photo_sim_tool/images/img1.jpg
# ../check_photo_sim_tool/images/所有照片
def main():
    print("=== 🚀 欢迎使用图片内容对比工具 ===")

    file_path = input_file("请输入要处理的图片路径：")
    folder_path = input_folder("请输入对比图片文件夹路径：")
    threshold = input_threshold("请输入阈值（浮点数）：")

    print("\n✅ 参数已输入完毕：")
    print(f"📄 文件路径: {file_path}")
    print(f"📁 文件夹路径: {folder_path}")
    print(f"📊 阈值: {threshold}")

    compare(file_path, folder_path, threshold)

if __name__ == '__main__':
    main()