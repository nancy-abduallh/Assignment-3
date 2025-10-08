import os
import json


def verify_dataset_structure():
    data_path = "./data/lasot"
    classes = ['airplane', 'deer', 'electricfan']

    class_info = {}

    for class_name in classes:
        class_path = os.path.join(data_path, class_name)
        if os.path.exists(class_path):
            # Count sequences (subfolders)
            sequences = [d for d in os.listdir(class_path)
                         if os.path.isdir(os.path.join(class_path, d))]

            # Count total images
            total_images = 0
            for seq in sequences:
                seq_path = os.path.join(class_path, seq)
                img_path = os.path.join(seq_path, 'img')
                if os.path.exists(img_path):
                    images = [f for f in os.listdir(img_path)
                              if f.endswith(('.jpg', '.png', '.jpeg'))]
                    total_images += len(images)

            class_info[class_name] = {
                'sequences': len(sequences),
                'total_images': total_images
            }
            print(f"Class {class_name}: {len(sequences)} sequences, {total_images} images")
        else:
            print(f"Warning: Class {class_name} not found!")

    # Save class information for report
    with open('dataset_info.json', 'w') as f:
        json.dump(class_info, f, indent=4)

    return class_info


if __name__ == "__main__":
    verify_dataset_structure()