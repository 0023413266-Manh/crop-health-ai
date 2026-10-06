import os
import zipfile


def create_kaggle_zip(folder_to_zip, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_to_zip):
            for file in files:
                file_path = os.path.join(root, file)
                # Lấy đường dẫn tương đối
                arcname = os.path.relpath(
                    file_path, start=os.path.dirname(folder_to_zip)
                )
                # CHUYỂN TOÀN BỘ DẤU '\' THÀNH '/' CHUẨN LINUX/KAGGLE
                arcname = arcname.replace('\\', '/')
                zipf.write(file_path, arcname)

    print(f"✅ Đã nén thành công file chuẩn Kaggle: {output_zip}")


# Chạy nén thư mục weights
create_kaggle_zip('weights', 'weights_kaggle.zip')