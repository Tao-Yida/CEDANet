import cv2
import os
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=str, default="BVM/data/frames")
    parser.add_argument('--masks', type=str, default="BVM/data/results")
    parser.add_argument('--output', type=str, default="BVM/data/out_videos")
    parser.add_argument('--heatmaps', type=str, default="BVM/data/heatmaps")
    parser.add_argument('--store_video', type=bool, default=True)
    opt = parser.parse_args()

    if not os.path.exists(opt.output):
        os.makedirs(opt.output)

    for vid in os.listdir(opt.frames):
        print(f"Processing video: {vid}")
        original_folder = os.path.join(opt.frames, vid)
        mask_folder = os.path.join(opt.masks, vid)
        heatmaps_folder = os.path.join(opt.heatmaps, vid)
        output_video = os.path.join(opt.output, f"{vid}.avi")

        original_images = sorted(os.listdir(original_folder))
        mask_images = sorted(os.listdir(mask_folder))

        fps = 30
        original_img = cv2.imread(os.path.join(original_folder, original_images[0]))
        frame_size = (original_img.shape[0], original_img.shape[1])  # Adjust as needed
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

        for original_img_name, mask_img_name in zip(original_images, mask_images):
            original_img_path = os.path.join(original_folder, original_img_name)
            mask_img_path = os.path.join(mask_folder, mask_img_name)

            original_img = cv2.imread(original_img_path)
            mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

            # Resize the mask to match the size of the original image
            mask_img_resized = cv2.resize(mask_img, (original_img.shape[1], original_img.shape[0]))
            
            heatmap = cv2.applyColorMap(mask_img_resized, cv2.COLORMAP_JET)
            super_imposed_img = cv2.addWeighted(heatmap, 0.3, original_img, 1.0, 0)
            out.write(super_imposed_img)

            # Write the heatmap to a folder
            if not os.path.exists(heatmaps_folder):
                os.makedirs(heatmaps_folder)
            heatmap_path = os.path.join(heatmaps_folder, mask_img_name)
            cv2.imwrite(heatmap_path, super_imposed_img)

            # Display the frame (optional)
            # cv2.imshow('Masked Image', super_imposed_img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # Release the VideoWriter and close any open windows
        out.release()
        # cv2.destroyAllWindows()
    