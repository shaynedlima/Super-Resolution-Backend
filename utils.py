from PIL import Image
import torch
import torchvision.transforms.functional as FT
import os
from google.cloud import storage
import tempfile

dirname = os.path.dirname(__file__)
bucket_name = "super_res_bucket"
bucket_prev_results_name = "super-res-results"
bucket_link = "https://storage.googleapis.com/super_res_bucket/"
bucket_results_link = "https://storage.googleapis.com/super-res-results/"
models_path = os.path.join(dirname, 'models')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Some constants
rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)


def convert_image(img, source, target):
    """
    Convert an image from a source format to a target format.

    :param img: image
    :param source: source format, one of 'pil' (PIL image), '[0, 1]' or '[-1, 1]' (pixel value ranges)
    :param target: target format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges),
                   'imagenet-norm' (pixel values standardized by imagenet mean and std.),
                   'y-channel' (luminance channel Y in the YCbCr color format, used to calculate PSNR and SSIM)
    :return: converted image
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % source
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm',
                      'y-channel'}, "Cannot convert to target format %s!" % target

    # Convert from source to [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)

    elif source == '[0, 1]':
        pass  # already in [0, 1]

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # Convert from [0, 1] to target
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # already in [0, 1]

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
        # Based on definitions at https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.

    return img

def allowed_image(filename, app):

    # We only want files with a . in the filename
    if not "." in filename:
        return False

    # Split the extension from the filename
    ext = filename.rsplit(".", 1)[1]

    # Check if the extension is in ALLOWED_IMAGE_EXTENSIONS
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

def scale_image(image, max_pixel_length):    
    image = image.convert('RGB')

    # Want to input an image that has max. pixel side length of 500
    width_scale = int(image.width/max_pixel_length)
    height_scale = int(image.height/max_pixel_length)

    print(width_scale)
    print(height_scale)
    scale = width_scale if (width_scale>height_scale) else height_scale
    scale = 1 if scale==0 else scale

    print("Scale: ", scale)
    lr_img = image.resize((int(image.width / scale), int(image.height / scale)),
                           Image.BILINEAR)

    print("LR Dimensions, W: ", lr_img.width, ", H: ", lr_img.height)
    return lr_img

def app_configs(app):
    app.config["DEBUG"] = True
    app.config["BUCKET_NAME"] = bucket_name
    app.config["BUCKET_LINK"] = bucket_link
    app.config["BUCKET_NAME_PREV"] = bucket_prev_results_name
    app.config["BUCKET_LINK_PREV"] = bucket_results_link
    app.config["MODELS"] = models_path
    app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
    app.config["MAX_IMAGE_FILESIZE"] = 0.5 * 1024 * 1024
    return app

def allowed_image_filesize(filesize, app):
    print(filesize)
    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False
        
def upload_gcp(bucket_name, source_image, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)


    with tempfile.NamedTemporaryFile(suffix='.jpg') as gcs_image:
        source_image.save(gcs_image)
        gcs_image.seek(0)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_file(gcs_image)
        
    print(
        "File {} uploaded to {}.".format(
            destination_blob_name, destination_blob_name
        )
    )

def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)

    blob_names = [blob.name for blob in blobs]
    
    return blob_names