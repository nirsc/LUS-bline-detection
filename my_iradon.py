from skimage.transform import iradon

"""A wrapper for the iradon function meant to deal with the fact that often the original image is not square"""
def my_iradon(radon_image, theta=None, output_size=None, filter="ramp", interpolation="linear", circle=True):
    if not (type(output_size) is tuple):
        return iradon(
            radon_image,
            theta=theta,
            output_size=output_size,
            filter=filter,
            interpolation=interpolation,
            circle=circle
        )
    elif (output_size[0] - output_size[1]) % 2 > 0:
        raise Exception('margin size is not an integer')
    elif output_size[0] < output_size[1]:
        margin = int((output_size[1] - output_size[0]) / 2)
        image = iradon(
            radon_image,
            theta=theta,
            output_size=output_size[1],
            filter=filter,
            interpolation=interpolation,
            circle=circle
        )
        return image[margin:-margin, :]
    else:
        margin = int((output_size[0] - output_size[1]) / 2)
        image = iradon(
            radon_image,
            theta=theta,
            output_size=output_size[0],
            filter=filter,
            interpolation=interpolation,
            circle=circle
        )
        return image[:, margin:-margin]

