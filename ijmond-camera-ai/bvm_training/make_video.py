import cv2
import os

original_folder = 'day'
mask_folder = 'results/day/'
output_video = 'output_video.avi'
original_images = sorted(os.listdir(original_folder))
mask_images = sorted(os.listdir(mask_folder))

fps = 30  # Frames per second
frame_size = (640, 480)  # Adjust as needed
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

for original_img_name, mask_img_name in zip(original_images, mask_images):
    original_img_path = os.path.join(original_folder, original_img_name)
    mask_img_path = os.path.join(mask_folder, mask_img_name)

    original_img = cv2.imread(original_img_path)
    mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

    # Resize the mask to match the size of the original image
    mask_img_resized = cv2.resize(mask_img, (original_img.shape[1], original_img.shape[0]))

    # Apply the mask to the original image
    masked_img = cv2.bitwise_and(original_img, original_img, mask=mask_img_resized)

    # Write the frame to the video
    out.write(masked_img)

    # # Display the frame (optional)
    # cv2.imshow('Masked Image', masked_img)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release the VideoWriter and close any open windows
out.release()
cv2.destroyAllWindows()

