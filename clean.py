
def image_cleaner(root_dir, save_dir):
    from PIL import Image
    import os

    image_paths = os.listdir(root_dir)
    print(len(os.listdir(save_dir)))

    for i, image_path in enumerate(image_paths):
        if not os.path.exists(f"{save_dir}/{image_path}"):
            try:
                img = Image.open(f"{root_dir}/{image_path}")
                img = img.convert("RGB")
                img.save(f"{save_dir}/{image_path}")

                print(f"{i} out of {len(image_paths)}")
            except Exception as e:
                print(f"{image_path} FAILED {e}")

image_cleaner(
    root_dir="./data/data",
    save_dir="./data/data_clean"
)
