from uts import ismcuts, iseuts

__all__ = ['image', 'imshow', 'imsave', 'plot']

def plot(s, **kwargs):
    if ismcuts(s):
        from plots import mcuts_subplots
        return mcuts_subplots(s, **kwargs)
    else:
        raise NotImplementedError("'plot' not implemented for {}".format(type(s)))

def image(s, **kwargs):
    if iseuts(s):
         from plots import euts_image
         return euts_image(s, **kwargs)
    else:
        raise NotImplementedError("'implot' not implemented for {}".format(type(s)))

def imshow(s, image_kwargs=None, imshow_kwargs=None):
    if image_kwargs is None:
        image_kwargs = {}
    if imshow_kwargs is None:
        imshow_kwargs = {}
    if iseuts(s):
         from plots import euts_image
         from matplotlib.pyplot import imshow
         print imshow_kwargs
         return imshow(image(s, **image_kwargs), **imshow_kwargs)
    else:
        raise NotImplementedError("'imshow' not implemented for {}".format(type(s)))


def imsave(s, **kwargs):
    if iseuts(s):
         from plots import save_euts_image
         return save_euts_image(s, **kwargs)
    else:
        raise NotImplementedError("'imsave' not implemented for {}".format(type(s)))




