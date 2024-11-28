import cv2
import os

input_folder = 'raw_data'
output_folder = 'processed_data'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        processed_image = cv2.GaussianBlur(image, (5, 5), 0)
        output_path = os.path.join(output_folder, filename.split('.')[0] + '_processed.' + filename.split('.')[1])
        cv2.imwrite(output_path, processed_image)
