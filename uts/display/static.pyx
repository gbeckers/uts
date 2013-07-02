# coding=utf-8

import sys
from pylab import cm
import numpy as np
from scipy.stats import scoreatpercentile
from scipy.misc import toimage
from scipy.misc import imresize
from os.path import abspath, join
import os, os.path
from uts.core.utils import iminterpolate
import uts
from matplotlib.colors import LinearSegmentedColormap
import Image, ImageFont, ImageDraw



class VoltageMap2(LinearSegmentedColormap):
    
    cdict = {'red':   ((0.0, 0.0, 0.0),
                       (0.5, 0.0, 0.0),
                       (0.75, 1.0, 1.0),
                       (1.0, 1.0, 1.0)),
             'green': ((0.0, 1.0, 1.0),
                       (0.25, 0.0, 0.0),
                       (0.5, 0.0, 0.0),
                       (0.75, 0.0, 0.0),
                       (1.0, 1.0, 1.0)),
             'blue': ((0.0, 1.0, 1.0),
                      (0.5, 0.0, 0.0),
                      (1.0, 0.0, 0.0))}
    
    def __init__(self):
        LinearSegmentedColormap.__init__(self, 'voltage_colormap', self.cdict, 256)

class VoltageMap3(LinearSegmentedColormap):
    
    cdict = {'green':   ((0.0, 0.0, 0.0),
                       (0.5, 0.0, 0.0),
                       (0.75, 1.0, 1.0),
                       (1.0, 1.0, 1.0)),
             'red': ((0.0, 1.0, 1.0),
                       (0.25, 0.0, 0.0),
                       (0.5, 0.0, 0.0),
                       (0.75, 0.0, 0.0),
                       (1.0, 1.0, 1.0)),
             'blue': ((0.0, 1.0, 1.0),
                      (0.5, 0.0, 0.0),
                      (1.0, 0.0, 0.0))}
    
    def __init__(self):
        LinearSegmentedColormap.__init__(self, 'voltage_colormap', self.cdict, 256)


# def iter_image_series(mcuts, starttime, endtime, ips=30.,ranges=None, clip=None, 
#                       zeroincenter=False, invert=False, msize=None, 
#                       interpolate='nearest'):
    
#     if (not(clip is None)) and (not(ranges is None)):
#         raise ValueError, \
#               "You cannot set ranges and clip parameters at the same time"
#     #samples = mcuts.samples.get(False)
#     #low = samples.min()
#     #high = samples.max()
#     dt = 1/float(ips)
#     if not clip is None:
#         low,high = clip
#     else:
#         low = np.inf
#         high = -np.inf
#         nframes = int(np.floor((endtime-starttime) / float(dt)))
#         for framen in range(nframes):
#             fstarttime = framen * dt + starttime
#             fendtime = framen * dt + dt + starttime
#             framemean = mcuts[fstarttime:fendtime].samples.get(copy=False).mean()
#             if framemean > high:
#                 high = framemean
#             if framemean < low:
#                 low = framemean
#     if zeroincenter:
#         high = max(abs(low),abs(high))
#         low = -high
#     #print "low,high", low, high      
            
#     if not ranges is None:
#         low = scoreatpercentile(mcuts.samples.flatten(),ranges[0])
#         high = scoreatpercentile(mcuts.samples.flatten(),ranges[1])
#     g = mcuts.geometry
#     nframes = int(np.floor((endtime-starttime) / float(dt)))
#     for framen in range(nframes):
#         fstarttime = framen * dt + starttime
#         fendtime = framen * dt + dt + starttime
#         print framen, dt, starttime
#         print 'chopping', fstarttime, fendtime
#         frame = mcuts[fstarttime:fendtime].transpose(['time','channel']).samples.get(copy=True)
#         print frame.shape
#         np.clip(frame, low, high, frame)  
#         frame -= low
#         frame /= (high - low)
#         if invert:
#             frame -= 1.0
#             frame[:] = np.absolute(frame)
#         mat = g.mapfrom1Dto2D(frame.mean(0), mcuts.channelnames)
#         if not msize is None:
#             mat = imresize(mat,msize,interpolate)
#             yield mat
            
def iter_image_series_noresample(mcuts, ranges=None, clip=None, 
                      zeroincenter=False, invert=False, msize=None, 
                      interpolate='nearest'):
    
    if (not(clip is None)) and (not(ranges is None)):
        raise ValueError(
            "You cannot set ranges and clip parameters at the same time")
    if not clip is None:
        low,high = clip
    else:
        low = uts.min(mcuts)
        high = uts.max(mcuts)
    if zeroincenter:
        high = max(abs(low),abs(high))
        low = -high
    if not ranges is None:
        samples = mcuts.samples.get(False)
        low = scoreatpercentile(samples.flatten(),ranges[0])
        high = scoreatpercentile(samples.flatten(),ranges[1])
    g = mcuts.geometry
    #for frame in mcuts.iter_frames(1):
    #    frame = frame.transpose(['time','channel']).samples.get(copy=True)[0]
    for i in range(mcuts.ntimesamples):
        if mcuts.channelaxis == 0:
          frame = mcuts.samples[:,i]
        else:
          frame = mcuts.samples[i]
        frame = np.clip(frame, low, high)
        mat = g.mapfrom1Dto2D(frame, mcuts.channelnames)
        if not msize is None:
            cliprange = high-low
            lsc = (mat.min()-low) / cliprange
            hsc = (mat.max()-low) / cliprange
            mat = imresize(mat,msize,interpolate).astype(np.float32) # this function scales everything from 0 to 255
            # we scale it back
            mat *= (hsc-lsc)/255.
            mat += lsc
        else:
            mat -= low
            mat /= (high - low)
            #print 2, frame.max(), frame.min(),frame.mean()
        if invert:
            mat -= 1.0
            mat[:] = np.absolute(mat)
        yield mat
        
def save_image_series(mcutss, basefilename, path='.', starttime=None, 
                      endtime=None, ranges=None, clips=None, 
                      zeroincenters=None, inverts=None, colormaps=None, 
                      msize=(141,141), hspace=10, hcolor=(0,0,0,1),
                      adddiff=False, colormapdiff=None, diffmag=10., 
                      interpolate='nearest', hflip=False, 
                      copyrighttext="2013 G.J.L. Beckers et al.",
                      label="", textwindowheight=70, startnumber=0, imgext='png', badchannels=None,
                      badcolors=None, debug=True):
    
    if (type(mcutss) != type([])):
        raise TypeError("mcutss must be a list")
    if (len(mcutss)%2 == 0) & (hspace%2 != 0):
        hspace += 1 # otherwise we can't make movies
    if badcolors is None:
        badcolors = []
        for _ in mcutss:
          badcolors.append((0.5,0.5,0.5,1.0))
    # use sensible defaults if parameters were not specified
    if starttime is None:
        starttime = -np.inf
        endtime = np.inf
        for mcuts in mcutss:
            cstarttime,cendtime = mcuts.timerange
            if cstarttime > starttime:
                starttime = cstarttime
            if cendtime < endtime:
                endtime = cendtime
    if colormaps is None:
        colormaps = [cm.hot]*len(mcutss)
    if clips is None:
        clips = [None]*len(mcutss)
    if zeroincenters is None:
        zeroincenters = [None]*len(mcutss)
    if inverts is None:
        inverts = [False]*len(mcutss)
    if colormapdiff is None:
        colormapdiff = cm.PRGn
    path = abspath(path)
    
    # if not copyrighttext is None:
    #     copyrighttext = unicode("© ",'utf-8') + copyrighttext
    # bad channels
    
    #shape = mcutss[0].geometry.shape
    badmask = np.zeros(mcutss[0].geometry.shape[::-1], np.int)
    if not badchannels is None:
        chns = mcutss[0].channelnames[mcutss[0]._to_channelsslice(badchannels)]
        if not isinstance(chns, np.ndarray):
          chns = [chns]
        for chn in chns:
          badmask[mcutss[0].geometry[chn]] = 1
    if not msize is None:
        badmask = imresize(badmask,msize,'nearest').astype('?')
    dirname = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]
    # print os.path.join(dirname,"uts", "fonts","Ubuntu-R.ttf")
    # FIXME doe nots use uts hard
    font = ImageFont.truetype(os.path.join(dirname,"uts","fonts","Ubuntu-R.ttf"), 16)
        
    ## now calculate the images
    imiters = []
    # first get the iterators that yield the images for each panel
    for i,mcuts in enumerate(mcutss):
        startindex, endindex = mcuts.time_to_index([starttime, endtime]) # for cython
        imiters.append(iter_image_series_noresample(mcuts=mcuts[startindex:endindex], 
                                     ranges=ranges,
                                     clip=clips[i], 
                                     zeroincenter=zeroincenters[i], 
                                     invert=inverts[i], msize=msize, 
                                     interpolate=interpolate))
    # now loop over time bins, and yield the images in mats
    startindex, endindex = mcutss[0].time_to_index([starttime, endtime]) # for cython    
    samplingtimes = mcutss[0][startindex:endindex].samplingtimes()     
    nloops = endindex - startindex
    if debug:
        print "fs: %f" % mcutss[0].fs
        print "starttime: %f, endtime: %f" % (starttime, endtime)
        print "startindex: %d, endindex: %d" % (startindex, endindex)
        print "nloops: %d" % nloops
    #for i, mats in enumerate(zip(*imiters)):
    print "producing %d images" % nloops
    for i in range(nloops):
        mats = []
        for j in range(len(mcutss)):
            # if hflip:
            #     mats.append(imiters[j].next()[:,::-1])
            # else:
            mats.append(imiters[j].next())
        print "<%d>"%(i+startnumber), #len(mats), mats[0].shape
        sys.stdout.flush()
        w = mats[0].shape[1]
        h = mats[0].shape[0]
        nmats = len(mats)
        
        # note colormap() adds a last dimension of length 4
        if nmats>1:
            if not adddiff:
                tempar = colormaps[0](np.zeros((h,w*nmats+hspace*(nmats-1))))
                d = colormaps[0](mats[0])
                d[badmask,:] = badcolors[0]
                if hflip:
                    d = d[:,::-1]
                tempar[:,:w] = d  
                for j in range(1,nmats):
                    d = colormaps[j](mats[j])
                    d[badmask] = badcolors[j]
                    if hflip:
                        d = d[:,::-1]
                    tempar[:,hspace*j+j*w:hspace*j+(j+1)*w] = d
                    
            else:
                tempar = colormaps[0](np.zeros((h,w*(nmats+1)+hspace*(nmats))))
                d = colormaps[0](mats[0])
                d[badmask,:] = badcolors[0]
                tempar[:,:w] =  d
                for j in range(1,nmats):
                    d =  colormaps[j](mats[j])
                    d[badmask] = badcolors[j]
                    tempar[:,hspace*j+j*w:hspace*j+(j+1)*w] = d
                j += 1
                d = colormapdiff((((mats[j-1]-mats[0])*diffmag)+2.0)/4.0)
                d[badmask] = (0.5,0.5,0.5,1.0)
                if hflip:
                    d = d[:,::-1]
                tempar[:,hspace*j+j*w:hspace*j+(j+1)*w] = d
                # if hflip:
                #     tempar = tempar[:,::-1]
        else:
            tempar = colormaps[0](mats[0])
            tempar[badmask,:] = badcolors[0]
            if hflip:
                tempar = tempar[:,::-1]
        car = tempar
        if adddiff:
            nmats += 1
        if nmats>1:
            for j in range(1,nmats):
                car[:,hspace*(j-1)+j*w:hspace*(j-1)+hspace+j*w] = hcolor
        # print car.min(), car.max(), car.dtype, car.shape
        height, width, depth = car.shape
        newheight = height + textwindowheight
        if (height % 2) == 1: # codecs want an even height 
            newheight += 1
        newcar = np.empty((newheight, width, depth), dtype=np.uint8)
        newcar[:height,:] = np.uint(car*255)
        newcar[height:] = (0,0,0,255)
        # im = Image.fromarray(np.uint8(car*255), mode='RGBA')
        im = Image.fromarray(newcar, mode='RGBA')
        # if not (copyrighttext is None):
        draw = ImageDraw.Draw(im)
        text = 'Time: %.3f s    Frame: %.5d    dt: %.2f ms' % (samplingtimes[i], (i+startnumber), mcutss[0].dt*1000)
        textwidth, textheight = font.getsize(text)
        draw.text((5, height+5), text, font=font)
        textwidth, textheight = font.getsize(label)    
        newtextheight = height + textheight
        draw.text((5, newtextheight+5), label, font=font)
        if copyrighttext:
            textwidth, textheight = font.getsize(copyrighttext)
            newtextheight += textheight
            draw.text((5, newtextheight+5), copyrighttext, font=font)
        absfilename = join(path, '%s_%.5d.%s' % (basefilename,i+startnumber,imgext))
        im.save(absfilename)
    return i+startnumber

def create_movie(moviename, moviepath, moviedir=None, fs=30., fps=30., audio=None, 
                 remove_frames=True, *args,**kwargs):
    import subprocess
    import tempfile
    import shutil
    if moviedir is None:
        kwargs['path'] = tempfile.mkdtemp(dir=moviepath)
    else:
        kwargs['path'] = moviedir        
    kwargs['basefilename'] = 'frame'
    kwargs['fs'] = fs
    
    # create a series of png files 
    save_image_series(*args,**kwargs)
    
    # compile them to a movie
    pngfiles = os.path.join(kwargs['path'], 'frame_%5d.png')
    moviename = os.path.join(moviepath, moviename)
    subprocess.call(['ffmpeg', '-r', '%f'%fps, '-b', '9600',
                     '-i', pngfiles, moviename+'.webm'])
    if remove_frames:
        shutil.rmtree(kwargs['path'])
        
    # and add audio if requested
    if not audio is None:
        audiofilename = os.path.join(moviepath,'audio.wav')
        uts.io.write_audiofile(audio,audiofilename,fmt=('wav','pcm16','file'),
                               overwrite=True)
        subprocess.call(['ffmpeg', '-i', moviename+'.webm', '-i', audiofilename,
                         '-ar', '22050', '-acodec', 'libvorbis', 
                         '-vcodec', 'copy', moviename+'_audio.webm'])
        os.remove(audiofilename)

def create_movie_from_rec(r, starttime, endtime, fs=30., fps=30., 
        clips=[(-8e-4,8e-4),(-8e-2,8e-2),(1e-5,2e-5)], audiochannel=None, 
        moviename=None, moviepath='.', msize=(141,141), 
                           remove_frames=True, 
                           copyrighttext=unicode("© G.J.L. Beckers",'utf-8')):
    import ds
    import tempfile
    
    # get the signals
    mcutsa = r.get_signal('amua')
    mcutsl = r.get_signal('lfp')
    mcutsc = r.get_signal('csd')
    if moviename is None:
        moviename = r.name + '_%.3f_%.3f_%.2ffs_%.2ffps' \
                             % (starttime,endtime, fs, fps)
    if (not audiochannel is None) and (fs == fps):
        sound = ds.bp(r.samples[starttime:endtime,[audiochannel]])
        audio = ds.scale(sound,1/ds.min(sound))
    else:
        audio = None
    # now create a movie
    create_movie(moviename=moviename,moviepath=moviepath, fs=fs, fps=fps, 
                 audio=audio, remove_frames=remove_frames, 
                 mcutss=[mcutsl,mcutsc,mcutsa], starttime=starttime, 
                 endtime=endtime, 
                 colormaps=[VoltageMap2(),VoltageMap2(), cm.bone],
                 zeroincenters=[True,True,False], 
                 clips=clips, hflip=True, hcolor=(1,1,1,1), msize=msize,
                 hspace=5, copyrighttext=copyrighttext)
    
    
    

    
    
        
# code for generating movie:
# mencoder mf://sleep_d20*.png -mf fps=40.95 -o amovie_sleep_d20.avi -ovc x264