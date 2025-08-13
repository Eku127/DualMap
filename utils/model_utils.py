import numpy as np
import torch
from PIL import Image

def compute_clip_features(
    image,
    detections,
    clip_model,
    clip_tokenizer,
    clip_preprocess,
    device,
    classes
):
    # output 1024 feats
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    
    # for image crop
    padding = 20
    
    # outputs
    image_crops = []
    image_feats = []
    text_feats = []
    
    for idx in range(len(detections.xyxy)):
        # get the crop region
        x_min, y_min, x_max, y_max = detections.xyxy[idx]
        # check and adjust the crop region using padding
        image_width, image_height = image.size
        
        left_padding = min(x_min, padding)
        right_padding = min(image_width - x_max, padding)
        top_padding = min(y_min, padding)
        bottom_padding = min(image_height - y_max, padding)
        
        x_min -= left_padding
        x_max += right_padding
        y_min -= top_padding
        y_max += bottom_padding
        
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        # add cropped into the list
        image_crops.append(cropped_image)
        
        # image CLIP feat
        clip_pre_image = clip_preprocess(cropped_image).unsqueeze(0).to(device)
        crop_feat = clip_model.encode_image(clip_pre_image)
        crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
        
        # text CLIP feat
        class_id = detections.class_id[idx]
        token_text_feat = clip_tokenizer(classes[class_id]).to(device)
        
        text_feat = clip_model.encode_text(token_text_feat)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        
        # move to cpu numpy
        crop_feat = crop_feat.detach().cpu().numpy()
        text_feat = text_feat.detach().cpu().numpy()
        
        image_feats.append(crop_feat)
        text_feats.append(text_feat)
        print(idx)

    # turn list of the feats into numpy matrices
    
    print(image_feats.shape)
    print(text_feats.shape)
    
    image_feats = np.concatenate(image_feats, axis=0)
    text_feats = np.concatenate(text_feats, axis=0)
    
    print(len(detections.xyxy))
    print(image_feats.shape)

def compute_clip_features_batched(
    image,
    detections,
    clip_model,
    clip_tokenizer,
    clip_preprocess,
    device,
    classes
):
    """
    This function computes CLIP features for a given image and its detections.
    It performs batch processing for efficiency.

    Parameters:
    - image (np.ndarray): The input image as a numpy array.
    - detections: The detections of objects in the image.
    - clip_model: The CLIP model for encoding image and text features.
    - clip_tokenizer: The CLIP tokenizer for converting text to tokens.
    - clip_preprocess: The CLIP preprocessing function for preparing image data.
    - device (torch.device): The device to run the computations on.
    - classes (list): The list of class names corresponding to the detections.

    Returns:
    - image_crops (list): A list of cropped images corresponding to the detections.
    - image_feats (np.ndarray): A numpy array of image features.
    - text_feats (np.ndarray): A numpy array of text features.
    """

    # Convert the image to a PIL Image
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    
    # Set the padding for cropping
    padding = 20
    
    # Initialize lists to store the cropped images and features
    image_crops = []
    image_feats = []
    text_feats = []
    
    # Initialize lists to store preprocessed images and text tokens for batch processing
    preprocessed_images = []
    text_tokens = []
    
    # Prepare data for batch processing
    for idx in range(len(detections.xyxy)):
        x_min, y_min, x_max, y_max = detections.xyxy[idx]
        image_width, image_height = image.size
        
        # Calculate the padding for each side of the bounding box
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        # Adjust the bounding box coordinates based on the padding
        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        # Crop the image
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        
        # Preprocess the cropped image
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0)
        preprocessed_images.append(preprocessed_image)

        # Get the class id for the detection
        class_id = detections.class_id[idx]
        
        # Append the class name to the text tokens list
        text_tokens.append(classes[class_id])
        
        # Append the cropped image to the image crops list
        image_crops.append(cropped_image)

    # Convert lists to batches
    preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(device)
    text_tokens_batch = clip_tokenizer(text_tokens).to(device)
    
    # Perform batch inference
    with torch.no_grad():
        # Encode the images using the CLIP model
        image_features = clip_model.encode_image(preprocessed_images_batch)
        
        # Normalize the image features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Encode the text tokens using the CLIP model
        text_features = clip_model.encode_text(text_tokens_batch)
        
        # Normalize the text features
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Convert the image and text features to numpy arrays
    image_feats = image_features.cpu().numpy()
    text_feats = text_features.cpu().numpy()
    
    # Return the cropped images, image features, and text features
    return image_crops, image_feats, text_feats