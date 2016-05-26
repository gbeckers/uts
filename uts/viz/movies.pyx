from __future__ import print_function

import numpy as np
import uts
from sys import stdout
from matplotlib import cm
import os


__all__ = ['create_movie_hmultipanel', 'create_testmovie', 'codecs',
           'FramesTimeAligned', 'interpolations']

interpolations = ("nearest", "bilinear", "bicubic", "spline16",
                  "spline36", "hanning", "hamming", "hermite",
                  "kaiser", "quadric", "catrom", "gaussian",
                  "bessel", "mitchell", "sinc", "lanczos")

codecs = {'png': (('-an', '-vcodec', 'png'), '%06d.png'),
          'h264_yuv444': (('-vcodec', 'libx264', '-crf', '5'), '.mp4'),
          'h264_yuv422': (('-vcodec', 'libx264', '-crf', '5', '-pix_fmt', 'yuv422p' ), '.mp4'),
          'h264_yuv420': (('-vcodec', 'libx264', '-crf', '5', '-pix_fmt', 'yuv420p'), '.mp4'),
          'avi_effectivelylossless1': (('-vcodec', 'huffyuv', '-pix_fmt', 'rgb24'), '.avi'),
          'avi_effectivelylossless2': (('-vcodec', 'huffyuv', '-pix_fmt', 'yuv422p'), '.avi'),
          'avi_perceptuallylossless': (('-vcodec', 'mjpeg', '-qscale:v', '1'), '.avi')}



# FIXME it would be nice to remove matplotlib dependence and have a default colormap function
# FIXME rename clips to scaling points oid
class FramesTimeAligned(object):
    def __init__(self, mcutss, fps=25, slowdown=10., chunksize=None,
                 show_progress=None, hflips=None,
                 colormaps=None, clips=None, units=None, scales=None):

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
        if units is None:
            self.units = ['' for i in range(self.npannels)]
        else:
            self.units = units
        self.scales = scales

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
        if self.show_progress:
            print('Generating {} frames: '.format(self.nframes), end='')
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
                    if not (self.scales is None):
                        samples = samples * self.scales[chunkn]
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

def create_movie_hmultipanel(frames, basefilename, path='.',
                             shape=(252, 570), fps=25, dpi=100, figwidth=8,
                             imheightfactor=6, textlines=None,
                             interpolation='nearest',
                             codec=codecs['h264_yuv420'],
                             vextraargs=['-vcodec', 'libx264', '-crf', '5'],
                             extension='.mp4',
                             savepng=False, pngnostart=0, fontcolor='white',
                             fontsize=12, labels=None, overwrite=False):
    # for compatibility with mac use ['-pix_fmt','yuv420p'] as vextraargs
    import matplotlib

    #matplotlib.use('Agg')  # needed to run on wakari
    import matplotlib.animation as animation
    from matplotlib.pylab import plt, tight_layout
    import matplotlib.gridspec as gridspec

    savefig_kwargs = {'facecolor': 'k', 'edgecolor': 'none'}
    if not codec is None:
        vextraargs = codec[0]
        extension = codec[1]

    if not os.path.exists(path):
        raise IOError("Path '{}' does not exist".format(path))

    def update_img(n):
        ars, k, time = framegenerator.next()
        for i, (im, ar) in enumerate(zip(ims, ars)):
            im.set_data(ar)
        textline = 'Time: %.3f s    Frame: %.5d    dt: %.3f ms    Slowed down: %.2fx' % (
        time, k, dt, slowdown)
        timebar.set_text(textline)
        if not labels is None:
            labelrow = labels[((labels['starttime']<=time) & (labels['endtime']>time))]
            if len(labelrow) == 0:
                label = ''
                labelcolor = fontcolor
            elif len(labelrow) == 1:
                label = labelrow[0]['label']
                if 'color' in labelrow.dtype.fields:
                    labelcolor = labelrow[0]['color']
                else:
                    labelcolor = fontcolor
            elif len(labelrow) > 1:
                raise ValueError('More than one label at time {} s.'.format(time))
            labelbar.set_text(label)
            labelbar.set_color(labelcolor)

        if savepng:
            plt.savefig(
                os.path.join(pngfullpath, 'frame%.3d.png' % (n + pngnostart)),
                       dpi=dpi, **savefig_kwargs)

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

    width_even_pixels = uts.round_nearesteven(figsize[0]*dpi)
    height_even_pixels = uts.round_nearesteven(figsize[1]*dpi)
    figwidth = width_even_pixels/float(dpi)
    figheight = height_even_pixels/float(dpi)
    fig.set_size_inches((figwidth, figheight))
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
    if not labels is None:
        nlines +=1

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
    if not labels is None:
        labelbar = textax.text(0.0, ystart, "",
                              horizontalalignment='left',
                              verticalalignment='center', color=fontcolor,
                              fontsize=fontsize)
        ystart += hshift

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

def create_testmovie(basefilename='testmovie', imheightfactor=6,
                     textlines=None,
                     interpolation='nearest',
                     codec=codecs['h264_yuv420'],
                     vextraargs=['-vcodec', 'libx264', '-crf', '5'],
                     extension='.mp4',
                     colormaps=('jet_r', 'bone'),
                     clips=((-0.5e-3, 0.5e-3), (0.5e-5, 2.0e-5)),
                     savepng=False
                     ):
    import os.path
    from ..signalfactory.examplesignals import load_doughnutexamplelfp, load_doughnutexampleamua

    datafilepath = os.path.join(os.path.dirname(uts.__file__),
                                'exampledata', 'examples.h5')
    lfp = load_doughnutexamplelfp()
    amua = load_doughnutexampleamua()

    fta = FramesTimeAligned([lfp, amua], clips=clips,
                            colormaps=colormaps, slowdown=25, show_progress=25)
    return create_movie_hmultipanel(frames=fta, basefilename=basefilename,
                                    imheightfactor=imheightfactor,
                                    textlines=textlines,
                                    interpolation=interpolation,
                                    codec=codec,
                                    vextraargs=vextraargs,
                                    extension=extension, savepng=savepng)


# class HPanelMovie(object):
#
#     def __init__(self, fta, staticimlabels=None, dynamicimlabels=None,
#                  pannelwidth=2.5, pannelheight=2.5, interpolation='nearest',
#                  hmargin=0.15, vmargin=0.0, hspace=0.15,
#                  timebarheight=0.4, legendbarheight=0.8,
#                  staticimlabelbarheight=0.4, dynamicimlabelbarheight=0.4,
#                  labelbarfontsize=12, timebarfontsize=11, dpi=100,
#                  facecolor='#505050', colorbarshrink=0.8, colorbarlabels=None,
#                  include_frameno=False, include_dt=False, sound=None,
#                  topaxesheight=0.0):
#
#         self.fta = fta
#         self.npannels = fta.npannels
#         self.framegenerator = None
#         self.dt = 1000 * fta.decimate / fta.fs
#         self.staticimlabels = staticimlabels
#         self.dynamicimlabels = dynamicimlabels
#         if staticimlabels is None:
#             staticimlabelbarheight = 0.0
#         if dynamicimlabels is None:
#             dynamicimlabelbarheight = 0.0
#         self.staticimlabelbarheight = staticimlabelbarheight
#         self.dynamicimlabelbarheight = dynamicimlabelbarheight
#         self.colorbarlabels = colorbarlabels
#         if colorbarlabels is None:
#             self.colorbarlabels = ['' for i in range(self.npannels)]
#         self.pannelwidth = pannelwidth
#         self.pannelheight = pannelheight
#         self.timebarheight = timebarheight
#         self.legendbarheight = legendbarheight
#         self.hmargin = hmargin
#         self.vmargin = vmargin
#         self.hspace = hspace
#         self.dpi = dpi
#         self.labelbarfontsize = labelbarfontsize
#         self.timebarfontsize = timebarfontsize
#         self.colorbarshrink = colorbarshrink
#         self.include_frameno = include_frameno
#         self.include_dt = include_dt
#         self.sound = sound
#
#         self.topaxess = []
#         self.topaxesheight = topaxesheight
#         self.topaxesupdatefuncs = []
#         self.topaxesinitfuncs = []
#
#         self.dynamicimlabeltext = []
#         self.facecolor = facecolor
#         self.savepng = None
#         self.fig = None
#         self.axes = None
#         self.ims = None
#         self.pngfullpath = None
#         self.savefig_kwargs = savefig_kwargs = {'facecolor': self.facecolor,
#                                                 'edgecolor': 'none'}
#         self.npannels = fta.npannels
#         self.interpolation = interpolation
#         self.setup_fig()
#
#     def calc_sizes(self):
#
#         figheight = self.vmargin * 2 \
#                     + self.topaxesheight \
#                     + self.staticimlabelbarheight \
#                     + self.dynamicimlabelbarheight \
#                     + self.pannelheight \
#                     + self.legendbarheight \
#                     + self.timebarheight
#         figwidth = self.hmargin * 2 \
#                     + self.pannelwidth * self.npannels \
#                     + self.hspace * (self.npannels - 1 )
#         # the next is necessary because we want even number of pixels in
#         # both width and height; some compress ion codecs won't work otherwise
#         width_even_pixels = uts.round_nearesteven(figwidth*self.dpi)
#         height_even_pixels = uts.round_nearesteven(figheight*self.dpi)
#         figwidth = width_even_pixels/float(self.dpi)
#         figheight = height_even_pixels/float(self.dpi)
#
#         # calc coordinates
#         self.pannelheight_c = self.pannelheight / figheight
#         self.vmargin_c = self.vmargin / figheight
#         self.topaxesheight_c = self.topaxesheight / figheight
#         self.staticimlabelbarheight_c = self.staticimlabelbarheight / figheight
#         self.dynamicimlabelbarheight_c = self.dynamicimlabelbarheight / figheight
#         self.legendbarheight_c = self.legendbarheight / figheight
#         self.timebarheight_c = self.timebarheight / figheight
#
#         self.pannelwidth_c = self.pannelwidth / figwidth
#         self.hmargin_c = self.hmargin / figwidth
#         self.hspace_c = self.hspace / figwidth
#
#         self.figwidth = figwidth
#         self.figheight = figheight
#
#     def setup_fig(self):
#
#         def clear_axes(ax):
#             ax.get_xaxis().set_visible(False)
#             ax.get_yaxis().set_visible(False)
#             ax.set_frame_on(False)
#
#         self.calc_sizes()
#         self.fig = fig = plt.figure(figsize=(self.figwidth, self.figheight),
#                                     facecolor='white', edgecolor=None, frameon=False)
#         self.gs = gs = gridspec.GridSpec(4, self.npannels, wspace=0.0, hspace=0.00)
#
#         axes = []
#         ims = []
#         cbs = []
#         hpos = self.hmargin_c
#
#         for i in range(self.npannels):
#             # panel labels
#             x = hpos + self.pannelwidth_c / 2.0
#             y = 1 - self.vmargin_c - self.topaxesheight_c - self.staticimlabelbarheight_c/2.0
#             if not self.staticimlabels is None:
#                 self.fig.text(x, y, self.staticimlabels[i],
#                               horizontalalignment='center',
#                               verticalalignment='center', color='white',
#                               fontsize=self.labelbarfontsize)
#             y = 1 - self.vmargin_c - self.topaxesheight_c - self.staticimlabelbarheight_c \
#                  - self.dynamicimlabelbarheight_c/2.0
#             if not self.dynamicimlabels is None:
#                 txt = fig.text(x, y, 'dynamic label',
#                               horizontalalignment='center',
#                               verticalalignment='center', color='black',
#                               fontsize=self.labelbarfontsize)
# #                 txt.set_animated(True)
#                 self.dynamicimlabeltext.append(txt)
#
#             y = 1 - self.vmargin_c - self.topaxesheight_c - self.staticimlabelbarheight_c \
#                 - self.dynamicimlabelbarheight_c - self.pannelheight_c
#             # image plots, own axes
#             ax_im = fig.add_axes([hpos, y, self.pannelwidth_c, self.pannelheight_c]) # [left, bottom, width, height] values in 0-1 relative figure coordinates
#             #print hpos, y, pannelwidth_c, pannelheight_c
#             clear_axes(ax_im)
#             #print 'panels', ax_im.get_position()
#             ims.append(ax_im.imshow([[0, 1], [1, 0]], interpolation=self.interpolation,
#                                     cmap=self.fta.colormaps[i], aspect='auto'))
#
#             # color bars, own axes
#             y = self.vmargin_c + self.timebarheight_c
#             ax_cb = fig.add_axes([hpos, y, self.pannelwidth_c, self.legendbarheight_c])
#             clear_axes(ax_cb)
#             #print 'cb', ax_cb.get_position()
#             cb = fig.colorbar(ims[-1], ax=ax_cb, orientation='horizontal', fraction=1, ticks=(0.,1), aspect=30,
#                              shrink=self.colorbarshrink, anchor=(0.5,0.5))
#             cb.set_label(self.colorbarlabels[i], labelpad=-30, color='#ffffff')
#             cb.ax.set_xticklabels(['{} {}'.format(v, self.fta.units[i]) for v in self.fta.clips[i]])
#             cbs.append(cb)
#             # topaxis
#             if not self.topaxesheight == 0.0:
#                 y = 1 - self.vmargin_c - self.topaxesheight_c
#                 ax = self.fig.add_axes([hpos, y, self.pannelwidth_c, self.topaxesheight_c])
#                 ax.set_axis_off()
#                 self.topaxess.append(ax)
#                 self.topaxesupdatefuncs.append(None)
#                 self.topaxesinitfuncs.append(None)
#             hpos += self.pannelwidth_c + self.hspace_c
#
#         y = self.vmargin_c + self.timebarheight_c/2.0
#         self.timebar = self.fig.text(self.hmargin_c, y, "",
#                           horizontalalignment='left',
#                           verticalalignment='center', color='white',
#                           fontsize=self.timebarfontsize)
#
#         self.ims = ims
#         self.cbs = cbs
#
#     def update_img(self, n):
#
#         ars, frameno, time = self.framegenerator.next()
#
#         changed = []
#         textline = 'Time: {:.3f} s'.format(time)
#         if self.include_frameno:
#             textline = '{}    Frame: {:d}'.format(textline, frameno)
#         if self.include_dt:
#             textline = '{}    dt: {:.3f} ms'.format(textline, 1000*self.fta.decimate/float(self.fta.fs))
#         if self.fta.slowdown != 1.0:
#             textline = '{}    Slowdown: {:.2f}x'.format(textline, self.fta.slowdown)
#         if not self.dynamicimlabels is None:
#             for i, dl in enumerate(self.dynamicimlabels):
#                 labelrow = dl[((dl['starttime']<=time) & (dl['endtime']>time))]
#                 if len(labelrow) == 0:
#                     self.dynamicimlabeltext[i].set_text('')
#                     self.dynamicimlabeltext[i].set_color('#ffffff')
#                 elif len(labelrow) == 1:
#                     fields = set(labelrow.dtype.fields) - set(['starttime', 'endtime'])
#                     for field in fields:
#                         self.dynamicimlabeltext[i].set(**{field: labelrow[0][field]})
#                 elif len(labelrow) > 1:
#                     raise ValueError('More than one label at time {} s.'.format(time))
#                 changed.append(self.dynamicimlabeltext[i])
# #                 print self.dynamicimlabeltext[i].get_animated(), self.dynamicimlabeltext[i].get_text(),\
# #                       self.dynamicimlabeltext[i].get_position(), self.dynamicimlabeltext[i].get_color()
#         self.timebar.set_text(textline)
#         changed.append(self.timebar)
#         for i, (im, ar) in enumerate(zip(self.ims, ars)):
#             im.set_data(ar)
#             changed.append(im)
#         if self.savepng:
#             self.fig.savefig(os.path.join(self.pngfullpath, 'frame%.3d.png' % (n)),
#                         dpi=self.dpi, **self.savefig_kwargs)
#         for topaxesupdatefunc in self.topaxesupdatefuncs:
#             if not topaxesupdatefunc is None:
#                 for change in topaxesupdatefunc(time, frameno):
#                     changed.append(change)
#         return tuple(changed)
#
#
#     def add_sound(self, filename, path='.', keepwav=False, ffmpegbin='ffmpeg',
#                   vextraargs=('-vcodec', 'libx264', '-crf', '5'), soxbin='sox'):
#
#         slowdown = self.fta.slowdown
#         with uts.io.utils.tempdir(path, keep=keepwav) as tempwavdir:
#             wavpath = os.path.join(tempwavdir, 'even.wav')
#             uts.io.audio.save_wav(self.sound, wavpath)
#             if slowdown != 1.0:
#                 args = [soxbin, wavpath, os.path.join(tempwavdir, 'even_slow.wav'),
#                         "speed", "%f"%(1/slowdown), "dither", "-s"]
#                 subprocess.call(args)
#                 wavpath = os.path.abspath(os.path.join(tempwavdir, 'even_slow.wav'))
#             moviewithsoundname = os.path.splitext(filename)[0] + '_snd.mp4'
#             args = [ffmpegbin, '-y', '-i', filename, '-i', wavpath]
#             args.extend(vextraargs)
#             args.extend(moviewithsoundname)
#             subprocess.call(args)
#
#     def create(self, basefilename, path='.',
#                codec=codecs['h264_yuv420'],
#                vextraargs=('-vcodec', 'libx264', '-crf', '5'),
#                extension='.mp4', blit=True,
#                overwrite=False, savepng=False):
#
#         def initanim():
#             changed = []
#             for im in self.ims:
#                 im.set_data([[],[]])
#                 changed.append(im)
#             for dl in self.dynamicimlabeltext:
#                 dl.set_text('Init')
#                 changed.append(dl)
#             self.timebar.set_text('Timebar')
#             changed.append(self.timebar)
#             for topaxesinitfunc in self.topaxesinitfuncs:
#                 if not topaxesinitfunc is None:
#                     for change in topaxesinitfunc():
#                         changed.append(change)
#             return tuple(changed)
#
#         self.framegenerator = self.fta.generate_frames()
#         self.savepng = savepng
#         if not codec is None:
#             vextraargs = codec[0]
#             extension = codec[1]
#         filename = basefilename + extension
#         fullpath = os.path.join(path, filename)
#         if os.path.exists(fullpath):
#             if overwrite:
#                 os.remove(fullpath)
#             else:
#                 raise IOError("File '{}' exists, use overwrite parameter to overwrite".format(fullpath))
#         if self.savepng:
#             self.pngfullpath = os.path.join(path, '%s_pngs' % basefilename)
#             if not os.path.exists(self.pngfullpath):
#                 os.makedirs(self.pngfullpath)
#         nframes = self.fta.nframes
#         ani = animation.FuncAnimation(self.fig, self.update_img, nframes - 1,
#                                       init_func=initanim,
#                                       interval=25,
#                                       blit=blit)
#
#         writer = animation.writers['ffmpeg'](fps=25, bitrate=-1, extra_args=vextraargs)
#         ani.save(fullpath, writer=writer, dpi=self.dpi, savefig_kwargs=self.savefig_kwargs)
#         plt.close(self.fig)
#         if not (self.sound is None):
#             self.add_sound(filename=filename, path=path)
#
#
# class LetterAxes(object):
#
#     def __init__(self, letters=('AAAAAAA', 'BBBBBBB'), fontsize=11, fontcolor='black',
#                  bgcolor='white', axes=None):
#         self.letters = letters
#         self._calc_positions()
#         self.fontsize = fontsize
#         self.bgcolor = bgcolor
#         self.fontcolor = fontcolor
#         self.lettertexts = None
#         if not axes is None:
#             self.axes = axes
#         else:
#             self.axes = plt.gca()
#         self.axes.set_axis_off()
#         self._draw_letters()
#
#     def _calc_positions(self):
#         self.nlines = len(self.letters)
#         self.lineheight_c = 1/float(self.nlines)
#         self.maxnletters = max([len(line) for line in self.letters])
#         self.letterwidth_c = 1/float(self.maxnletters)
#
#     def _draw_letters(self):
#
#         self.axes.add_patch(patches.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor=self.bgcolor, edgecolor='white'))
#         lettertexts = []
#         for lineno, lineletters in enumerate(self.letters[::-1]):
#             lettertexts.append([])
#             for letterno, letter in enumerate(lineletters):
#                 x = letterno * self.letterwidth_c + self.letterwidth_c * 0.5
#                 y = lineno * self.lineheight_c + self.lineheight_c * 0.5
#                 letter = self.axes.text(x,y, letter, horizontalalignment='center',
#                           verticalalignment='center', color='black',
#                           fontsize=self.fontsize)
#                 lettertexts[-1].append(letter)
#         self.lettertexts = lettertexts
#
#
# class TimeIndicatedLetterAxes(LetterAxes):
#
#     def __init__(self, starttime=0., endtime=10.,
#                  timeindicators=(5.0,), letters=('AAAAAAA', 'BBBBBBB'),
#                  fontsize=11, fontcolor='black', bgcolor='white', linewidth=1, axes=None):
#
#         self.starttime = starttime
#         self.endtime = endtime
#         self.duration = float(endtime - starttime)
#         self.timeindicators = timeindicators
#         self.linewidth = linewidth
#         super(TimeIndicatedLetterAxes, self).__init__(letters=letters,
#                                                       fontsize=fontsize,
#                                                       bgcolor=bgcolor,
#                                                       fontcolor=fontcolor,
#                                                       axes=axes)
#         self._draw_timeindicators()
#
#     def _time_to_x(self, time):
#         return (time - self.starttime) / self.duration
#
#     def _draw_timeindicators(self):
#         timeindicatorlines = []
#         for timeindicator in self.timeindicators:
#             x = self._time_to_x(timeindicator)
#             l = self.axes.add_line(lines.Line2D((x,x), (0.,1.), linewidth=self.linewidth, color='blue'))
#             timeindicatorlines.append(l)
#         self.timeindicatorlines = timeindicatorlines
#
#     def update(self, time, frameno):
#         for til in self.timeindicatorlines:
#             x = self._time_to_x(time)
#             til.set_data((x,x),(0.,1))
#         return tuple(self.timeindicatorlines)
#
#
#     def init(self):
#         for til in self.timeindicatorlines:
#             x = self._time_to_x(self.starttime)
#             til.set_data((x,x),(0.,1))
#         return tuple(self.timeindicatorlines)
