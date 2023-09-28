from src.predict import predict_image


# Call the predict_image function with your image data
# Replace 'image_path' with the path to your image file
image_path = 'classification_project\classification_project\Image\gfx100s_sample_04_thum-1.jpg'
predicted_class = predict_image(image_path)
print(f"Predicted class: {predicted_class}")
