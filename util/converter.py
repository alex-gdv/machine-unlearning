from PIL import Image
import os

def rgb_converter(source_dir, save_dir):
    image_paths = os.listdir(source_dir)
    
    for i, image_path in enumerate(image_paths):
        if not os.path.exists(f"{save_dir}/{image_path}"):
            try:
                img = Image.open(f"{source_dir}/{image_path}")
                img = img.convert("RGB")
                img.save(f"{save_dir}/{image_path}")

                print(f"{i} out of {len(image_paths)}")
            except Exception as e:
                print(f"{image_path} FAILED {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True, help="directory containing images to RGB convert")
    parser.add_argument("--save_dir", required=True, help="directory to save RGB converted images")
    args = parser.parse_args()

    assert os.path.isdir(args.source_dir), "source directory does not exist"
    assert os.path.isdir(args.save_dir), "save directory does not exist"

    rgb_converter(
        source_dir=args.source_dir,
        save_dir=args.save_dir
    )
