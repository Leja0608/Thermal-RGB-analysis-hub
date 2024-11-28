import cv2
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

class ImageRegistration:
    def __init__(self, fixed_image, moving_image):
        self.fixed_image = fixed_image
        self.moving_image = moving_image
        self.final_transform = None

    def register_images(self):
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.8)
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration_method.SetOptimizerAsOnePlusOneEvolutionary(
            numberOfIterations=500,
            epsilon=1.63e-6,
            initialRadius=0.0028,
            growthFactor=1.1
        )
        initial_transform = sitk.AffineTransform(self.fixed_image.GetDimension())
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        registration_method.SetInterpolator(sitk.sitkLinear)
        self.final_transform = registration_method.Execute(
            sitk.Cast(self.fixed_image, sitk.sitkFloat32),
            sitk.Cast(self.moving_image, sitk.sitkFloat32)
        )
        return self.final_transform

    def resample_image(self, moving_image):
        moving_image_resampled = sitk.Resample(
            moving_image,
            self.fixed_image,
            self.final_transform,
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelID()
        )
        return sitk.GetArrayFromImage(moving_image_resampled)


class ImageProcessor:
    def __init__(self, thermal_path, rgb_path):
        self.thermal_path = thermal_path
        self.rgb_path = rgb_path
        self.thermal = None
        self.rgb = None

    def load_images(self):
        self.thermal = cv2.imread(self.thermal_path)
        self.rgb = cv2.imread(self.rgb_path)
        if self.thermal is None:
            print("Error: Could not load the thermal image.")
        if self.rgb is None:
            print("Error: Could not load the RGB image.")

    def preprocess_images(self):
        if self.thermal is not None and self.rgb is not None:
            offset_image = np.zeros_like(self.rgb)
            offset_image[120:360, 160:480, :] = self.thermal
            self.thermal = offset_image
            self.thermal_gray = cv2.cvtColor(self.thermal, cv2.COLOR_BGR2GRAY)
            self.rgb_gray = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2GRAY)
            self.thermal_gray = cv2.GaussianBlur(self.thermal_gray, (5, 5), 0)
            self.rgb_gray = cv2.GaussianBlur(self.rgb_gray, (5, 5), 0)
            print("Thermal shape:", self.thermal.shape)
            print("RGB shape:", self.rgb.shape)
            print("Thermal grayscale shape:", self.thermal_gray.shape)
            print("RGB grayscale shape:", self.rgb_gray.shape)

    def show_image(self, image, title, grayscale=False):
        if grayscale:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()


def fusion_and_show_results(fixed_image, registered_image):
    fixed_image_np = sitk.GetArrayFromImage(fixed_image)
    resampled_rgb_image_resized = cv2.resize(registered_image, (fixed_image_np.shape[1], fixed_image_np.shape[0]))
    fused_image = cv2.addWeighted(fixed_image_np, 0.5, resampled_rgb_image_resized, 0.5, 0)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(fixed_image_np, cmap='gray')
    plt.title('Fixed RGB Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(resampled_rgb_image_resized, cmap='gray')
    plt.title('Registered RGB Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(fused_image, cmap='gray')
    plt.title('Fused Image (Fixed + Registered RGB)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    thermal_path = 'Database/raw_data/1_thermal.jpg'
    rgb_path = 'Database/raw_data/1_rgb.jpg'

    processor = ImageProcessor(thermal_path, rgb_path)
    processor.load_images()
    processor.preprocess_images()

    fixed_image = sitk.GetImageFromArray(processor.rgb_gray)
    moving_image = sitk.GetImageFromArray(processor.thermal_gray)

    registration = ImageRegistration(fixed_image, moving_image)
    registration.register_images()

    registered_image = registration.resample_image(moving_image)

    fusion_and_show_results(fixed_image, registered_image)


if __name__ == "__main__":
    main()
