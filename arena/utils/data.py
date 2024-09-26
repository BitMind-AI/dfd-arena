from bitmind.image_dataset import ImageDataset


def load_datasets(datasets):
    """

    Returns:
        real and fake ImageDatasets

    """
    fake_datasets = [
        ImageDataset(
            ds['path'],
            huggingface_dataset_split=ds['split'],
            huggingface_dataset_name=ds.get('name', None),
            create_splits=False,
            download_mode='reuse_cache_if_exists')
        for ds in datasets['fake']
    ]

    real_datasets = [
        ImageDataset(
            ds['path'],
            huggingface_dataset_split=ds['split'],
            huggingface_dataset_name=ds.get('name', None),            
            create_splits=False,
            download_mode='reuse_cache_if_exists')
        for ds in datasets['real']
    ]

    return real_datasets, fake_datasets


def upper_left_quadrant(img):

    width, height = img.size
    left = 0
    top = 0
    right = width // 2
    bottom = height // 2
    upper_left_img = img.crop((left, top, right, bottom))
    return upper_left_img
