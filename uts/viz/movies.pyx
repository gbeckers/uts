from __future__ import print_function

import numpy as np
import uts
from sys import stdout
import os
from matplotlib import cm


__all__ = ['create_movie_hmultipanel', 'create_testmovie', 'ffmpegcodecs', 'FramesTimeAligned', 'interpolations']

interpolations = ["nearest", "bilinear", "bicubic", "spline16",
                  "spline36", "hanning", "hamming", "hermite",
                  "kaiser", "quadric", "catrom", "gaussian",
                  "bessel", "mitchell", "sinc", "lanczos"]


# FIXME it would be nice to remove matplotlib dependence and have a default colormap function
# FIXME rename clips to scaling points oid
class FramesTimeAligned(object):
    def __init__(self, mcutss, fps=25, slowdown=10., chunksize=None,
                 show_progress=None, hflips=None,
                 colormaps=None, clips=None):

        self.mcutss = self.check_mcutss(mcutss)
        self.npannels = len(self.mcutss)
        self.fps = fps
        self.slowdown = slowdown
        self.fs = self.mcutss[0].fs
        self.ntimesamples = mcutss[0].ntimesamples
        self.decimate = int(np.round(self.fs / (slowdown * fps)))
        self.nframes = self.ntimesamples // self.decimate
        self.realslowdown = self.fs / float(self.decimate * fps)
        self.show_progress = show_progress
        if chunksize is None:
            chunksize = int(np.round(self.fs * 1.0)) * self.decimate
        self.chunksize = chunksize
        self.chunkindices = [ind for ind in
                             self.mcutss[0].movingtimeframeindices(
                                 framesize=self.chunksize,
                                 include_remainder=True)]
        self.hflips = self.check_hflips(hflips, self.mcutss)
        self.colormaps = self.check_colormaps(colormaps, self.mcutss)
        self.clips = self.check_clips(clips, self.mcutss)

    def check_mcutss(self, mcutss):
        for i in range(1, len(mcutss)):
            if not uts.istimealigned(mcutss[0], mcutss[i]):
                raise ValueError('not all signals are time-aligned')
        return mcutss

    def check_hflips(self, hflips, mcutss):
        if hflips is None:
            hflips = [False] * len(mcutss)
        else:
            for hflip, mcuts in zip(hflips, mcutss):
                if not isinstance(hflip, bool):
                    raise TypeError(
                        'hflip parameter is not a bool but %s' % type(hflip))
        return hflips

    def check_colormaps(self, colormaps, mcutss):
        if colormaps is None:
            return [cm.jet for mcuts in mcutss]
        else:
            return [cmap if hasattr(cmap, '__call__') else cm.get_cmap(cmap)
                    for cmap in colormaps]

    def check_clips(self, clips, mcutss):
        if clips is None:
            return [None for mcuts in mcutss]
        else:
            return clips

    def generate_frames(self):
        ars = []
        gmapindices = []
        for i, mcuts in enumerate(self.mcutss):
            gmapindices.append(
                mcuts.geometry.indexfrom1Dto2D(mcuts.channelnames,
                                               hflip=self.hflips[i]))
            x, y = mcuts.geometry.shape
            ars.append(np.empty((y, x, 4), dtype=np.float64))
            # print(gmapindices[-1])
            # print (ars[-1])
        k = -1
        for i, (chunkstart, chunkend) in enumerate(self.chunkindices):
            chunks = []
            for mcuts in self.mcutss:
                chunks.append(
                    uts.decimate(mcuts[chunkstart:chunkend], self.decimate))
            nchunkframes = chunks[0].ntimesamples
            for j in range(nchunkframes):
                time = (chunks[0][j].timerange[0] + chunks[0][j].timerange[
                    1]) / 2.0
                for chunkn, chunk in enumerate(chunks):
                    # print(gmapindices[chunkn])
                    # print(np.squeeze(chunk[j].samples.get(copy=False)))
                    # print(len(ars))
                    samples = np.squeeze(chunk[j].samples.get(copy=False))
                    if not self.clips[chunkn] is None:
                        # samples[samples < self.clips[chunkn][0]] = self.clips[chunkn][0]
                        # samples[samples > self.clips[chunkn][1]] = self.clips[chunkn][1]
                        samples -= self.clips[chunkn][0]
                        samples /= (
                        self.clips[chunkn][1] - self.clips[chunkn][0])
                    ars[chunkn][gmapindices[chunkn]] = self.colormaps[chunkn](
                        samples)
                k += 1
                if self.show_progress and (((k % self.show_progress) == 0) or (
                    k == (self.nframes - 1))):
                    print('<%d>' % k, end='')
                    stdout.flush()
                # this yield the frame array, the total number of frames,
                # the frame number, and the mean time of the frame
                yield ars, k, time


# def create_framegenerator(mcutss, fps=25, slowdown=10., show_progress=None):
#     for i in range(1,len(mcutss)):
#         if not uts.is_timealigned(mcutss[0],mcutss[i]):
#             raise ValueError('not all signals are time-aligned')
#     fs = mcutss[0].fs
#     ntimesamples = mcutss[0].ntimesamples
#     gmapindices = []
#     ars = []
#     for mcuts in mcutss:
#         gmapindices.append(mcuts.geometry.indexfrom1Dto2D(mcuts.channelnames))
#         ars.append(np.empty(mcuts.geometry.shape, dtype=mcuts.dtype))
#     decimate = int(np.round(fs/(slowdown*fps)))
#     print("Decimate factor for movie: %d" % decimate)
#     nframes = ntimesamples//decimate
#     # we will do this per +/- decimate seconds so as not to make things too big
#     chunksize = int(np.round(fs*1.0))*decimate # make sure we can decimate every frame
#     k = -1
#     for i, (starti,endi) in enumerate(mcutss[0].movingtimeframeindices(
#                                 framesize=chunksize, include_remainder=True)):
#         chunks = []
#         for mcuts in mcutss:
#             chunks.append(uts.decimate(mcuts[starti:endi], decimate))
#         nchunkframes = chunks[0].ntimesamples
#         for j in range(nchunkframes):
#             time = (chunks[0][j].timerange[0]+chunks[0][j].timerange[1])/2.0
#             for chunkn, chunk in enumerate(chunks):
#                 ars[chunkn][gmapindices[chunkn]] = np.squeeze(chunk[j].samples.get(copy=False))
#             k += 1
#             if show_progress and (((k%show_progress)==0) or (k==(nframes-1))):
#                 print('<%d>'%k, end='')
#                 stdout.flush()
#             # this yield the frame array, the total number of frames,
#             # the frame number, and the mean time of the frame
#             yield ars, nframes, k, time
#
# def concat_framegenerators(framegenerators):
#     for framegenerator in framegenerators:
#         for frame in framegenerator:
#             yield frame


ffmpegcodecs = {'png': (['-an', '-vcodec', 'png'], '%06d.png'),
                'h264': (['-vcodec', 'libx264', '-crf', '5'], '.mp4'),
                'avi_effectivelylossless1': (
                ['-vcodec', 'huffyuv', '-pix_fmt', 'rgb24'], '.avi'),
                'avi_effectivelylossless2': (
                ['-vcodec', 'huffyuv', '-pix_fmt', 'yuv422p'], '.avi'),
                'avi_perceptuallylossless': (
                ['-vcodec', 'mjpeg', '-qscale:v', '1'], '.avi')}

# def create_movie_hmultipanel(frames, clips, cmaps, basefilename, extension='.mp4',
#                              path='.', shape=(252,570), fps=25, dpi=100, figwidth=8,
#                              imheightfactor=6, textlines=None, interpolation='nearest',
#                              vextraargs=['-vcodec', 'libx264', '-crf','5'],
#                              savepng=False, pngnostart=0, fontcolor='white',
#                              fontsize=12):

#     # for compatibility with mac use ['-pix_fmt','yuv420p'] as vextraargs
#     import matplotlib
#     matplotlib.use('Agg') # needed to run on wakari
#     import matplotlib.animation as animation
#     from matplotlib.pylab import plt, tight_layout
#     import matplotlib.gridspec as gridspec

#     savefig_kwargs = {'facecolor': 'k', 'edgecolor': 'none'}

#     def update_img(n):
#         ars, k, time = framegenerator.next()
#         for im,ar in zip(ims,ars):
#             im.set_data(ar)
#         textline = 'Time: %.3f s    Frame: %.5d    dt: %.3f ms    Slowed down: %.2fx' % (time,k,dt, slowdown)
#         timebar.set_text(textline)
#         if savepng:
#             plt.savefig(os.path.join(pngfullpath,'frame%.3d.png'%(n+pngnostart)),**savefig_kwargs)

#     fullpath = os.path.join(path, basefilename + extension)
#     if savepng:
#         pngfullpath = os.path.join(path, '%s_pngs'%basefilename)
#         if not os.path.exists(pngfullpath):
#             os.makedirs(pngfullpath)

#     framegenerator = frames.generate_frames()
#     nframes = frames.nframes
#     dt = 1000*frames.decimate/frames.fs
#     slowdown = frames.realslowdown
#     fig = plt.figure(facecolor='k', edgecolor='k', frameon=False)
#     fig.set_facecolor('k')

#     nrows = int(imheightfactor) + 1
#     imrows = slice(0,-1)
#     #gs1.update(left=0.05, right=0.95, wspace=0.05)
#     # first arrays
#     ars, k, time = framegenerator.next()
#     nimplots = len(ars)
#     gs = gridspec.GridSpec(nrows, nimplots, wspace=0.0, hspace=0.00)
#     figsize = (figwidth, figwidth/float(nimplots)*((nrows+1)/float(nrows)))
#     fig.set_size_inches(figsize)
#     axes = []
#     ims = []
#     # startposx = 0.01
#     # posy = 0.01 + textbarheight
#     # yheight = 1.0 - posy - 0.01
#     # ximwidth = (1.0 - 2*0.01 - (nimplots-1)*0.01) / nimplots
#     for i in range(nimplots):
#         # posx = (startposx) + i*(ximwidth+0.01)
#         # position = (posx, posy, ximwidth, posy+yheight)
#         #print(position)
#         axes.append(fig.add_subplot(gs[imrows,i]))#, position=position))
#         #axes[-1].set_aspect('equal')
#         axes[-1].get_xaxis().set_visible(False)
#         axes[-1].get_yaxis().set_visible(False)
#         axes[-1].set_frame_on(False)

#         ims.append(axes[-1].imshow(ars[i], cmap=cmaps[i],
#                                    interpolation=interpolation))
#         ims[-1].set_clim(clips[i])

#     if textlines:
#         nlines = len(textlines) + 1
#     else:
#         nlines = 1
#     textax = fig.add_subplot(gs[nrows-1,:])
#     textax.get_xaxis().set_visible(False)
#     textax.get_yaxis().set_visible(False)
#     textax.set_frame_on(False)
#     hshift=1/float(nlines+1)
#     ystart=hshift
#     if not (textlines is None):
#         for textline in textlines:
#             textax.text(0.0, ystart, textline, horizontalalignment='left',
#                  verticalalignment='center', color=fontcolor, fontsize=fontsize)
#             ystart += hshift
#     textline = 'Time: %.3f s    Frame: %.5d    dt: %.3f ms    Slowed down: %.2fx' % (time,k,dt, slowdown)
#     timebar = textax.text(0.0, ystart, textline,
#                          horizontalalignment='left',
#                          verticalalignment='center', color=fontcolor, fontsize=fontsize)

#     tight_layout()
#     ani = animation.FuncAnimation(fig, update_img, nframes-2,
#                                   interval=25,
#                                   blit=False)
#     writer = animation.writers['ffmpeg'](fps=25, bitrate=-1,
#                                 extra_args=vextraargs)

#     ani.save(fullpath, writer=writer, dpi=dpi, savefig_kwargs=savefig_kwargs)
#     plt.close(fig)


def create_movie_hmultipanel(frames, basefilename, extension='.mp4', path='.',
                             shape=(252, 570), fps=25, dpi=100, figwidth=8,
                             imheightfactor=6, textlines=None,
                             interpolation='nearest',
                             vextraargs=['-vcodec', 'libx264', '-crf', '5'],
                             savepng=False, pngnostart=0, fontcolor='white',
                             fontsize=12, overwrite=False):
    # for compatibility with mac use ['-pix_fmt','yuv420p'] as vextraargs
    import matplotlib

    #matplotlib.use('Agg')  # needed to run on wakari
    import matplotlib.animation as animation
    from matplotlib.pylab import plt, tight_layout
    import matplotlib.gridspec as gridspec

    savefig_kwargs = {'facecolor': 'k', 'edgecolor': 'none'}

    if not os.path.exists(path):
        raise IOError("Path '{}' does not exist".format(path))

    def update_img(n):
        ars, k, time = framegenerator.next()
        for i, (im, ar) in enumerate(zip(ims, ars)):
            im.set_data(ar)
        textline = 'Time: %.3f s    Frame: %.5d    dt: %.3f ms    Slowed down: %.2fx' % (
        time, k, dt, slowdown)
        timebar.set_text(textline)
        if savepng:
            plt.savefig(
                os.path.join(pngfullpath, 'frame%.3d.png' % (n + pngnostart)),
                **savefig_kwargs)

    fullpath = os.path.join(path, basefilename + extension)
    if os.path.exists(fullpath):
        if overwrite:
            os.remove(fullpath)
        else:
            raise IOError("File '{}' exists, use overwrite parameter to overwrite".format(fullpath))

    if savepng:
        pngfullpath = os.path.join(path, '%s_pngs' % basefilename)
        if not os.path.exists(pngfullpath):
            os.makedirs(pngfullpath)

    # make sure we have mapping functions, not the names of the mapping functions

    framegenerator = frames.generate_frames()
    nframes = frames.nframes
    dt = 1000 * frames.decimate / frames.fs
    slowdown = frames.realslowdown
    fig = plt.figure(facecolor='k', edgecolor='k', frameon=False)
    fig.set_facecolor('k')

    nrows = int(imheightfactor) + 1
    imrows = slice(0, -1)
    #gs1.update(left=0.05, right=0.95, wspace=0.05)
    # first arrays
    #ars, k, time = framegenerator.next()
    nimplots = frames.npannels
    gs = gridspec.GridSpec(nrows, nimplots, wspace=0.0, hspace=0.00)
    figsize = (
    figwidth, figwidth / float(nimplots) * ((nrows + 1) / float(nrows)))
    fig.set_size_inches(figsize)
    axes = []
    ims = []
    # startposx = 0.01
    # posy = 0.01 + textbarheight
    # yheight = 1.0 - posy - 0.01
    # ximwidth = (1.0 - 2*0.01 - (nimplots-1)*0.01) / nimplots
    for i in range(nimplots):
        # posx = (startposx) + i*(ximwidth+0.01)
        # position = (posx, posy, ximwidth, posy+yheight)
        #print(position)
        axes.append(fig.add_subplot(gs[imrows, i]))  #, position=position))
        #axes[-1].set_aspect('equal')
        axes[-1].get_xaxis().set_visible(False)
        axes[-1].get_yaxis().set_visible(False)
        axes[-1].set_frame_on(False)

        ims.append(
            axes[-1].imshow([[0, 0], [0, 0]], interpolation=interpolation))

    if textlines:
        nlines = len(textlines) + 1
    else:
        nlines = 1
    textax = fig.add_subplot(gs[nrows - 1, :])
    textax.get_xaxis().set_visible(False)
    textax.get_yaxis().set_visible(False)
    textax.set_frame_on(False)
    hshift = 1 / float(nlines + 1)
    ystart = hshift
    if not (textlines is None):
        for textline in textlines:
            textax.text(0.0, ystart, textline, horizontalalignment='left',
                        verticalalignment='center', color=fontcolor,
                        fontsize=fontsize)
            ystart += hshift
    #textline = 'Time: %.3f s    Frame: %.5d    dt: %.3f ms    Slowed down: %.2fx' % (time,k,dt, slowdown)
    timebar = textax.text(0.0, ystart, "",
                          horizontalalignment='left',
                          verticalalignment='center', color=fontcolor,
                          fontsize=fontsize)

    tight_layout()
    def f():
        pass
    ani = animation.FuncAnimation(fig, update_img, nframes - 1, init_func=f,
                                  interval=25,
                                  blit=False)
    writer = animation.writers['ffmpeg'](fps=25, bitrate=-1,
                                         extra_args=vextraargs)

    ani.save(fullpath, writer=writer, dpi=dpi, savefig_kwargs=savefig_kwargs)
    plt.close(fig)

def create_testmovie(filename='testmovie.mp4', imheightfactor=6,
                     textlines=None,
                     interpolation='nearest',
                     vextraargs=['-vcodec', 'libx264', '-crf', '5']):
    import os.path

    datafilepath = os.path.join(os.path.dirname(uts.__file__),
                                'example_data', 'examples.h5')
    lfp = uts.load('doughnut_lfp', datafilepath, inmemory=True)
    amua = uts.load('doughnut_amua', datafilepath, inmemory=True)
    return create_movie_hmultipanel(FramesTimeAligned([lfp, amua],
                                                      slowdown=25,
                                                      show_progress=25),
                                    clips=(
                                    (-0.5e-3, 0.5e-3), (0.5e-5, 2.0e-5)),
                                    cmaps=('jet_r', 'bone'),
                                    filename=filename,
                                    imheightfactor=imheightfactor,
                                    textlines=textlines,
                                    interpolation=interpolation,
                                    vextraargs=vextraargs)