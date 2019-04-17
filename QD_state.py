'''
Arthur McCray
arthurmccray95@gmail.com
Summer 2016/2017

File containing the state (image) class and QD class
'''

'''
other thoughts/to do: 
-- the way it's currently setup it makes no sense for the FFT SL orientations 
to be in each individual dot.orientation. It should be a attribute of the state 
class. This isn't particularly important thought, and would be a pain to fix I reckon. 
'''


import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import skimage
import sklearn
# blob detection used for segmentation and bragg spot finding
from skimage.feature import blob_log, blob_dog, blob_doh
# for watershed implementation 
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import watershed
# for segmentation and making masks
from scipy.ndimage.morphology import binary_closing, binary_opening, binary_dilation, binary_erosion

from time import time
from os.path import exists
from os import makedirs

from scipy import ndimage as ndi
# could be switched in the code, used once
from scipy.ndimage.measurements import center_of_mass

# KD tree
from sklearn import neighbors
# for pretty colormaps
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
# for voronoi cells
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull

from helpers_orig import *
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button 
import time



class QD(object): 
    '''
    The quantum dot object - one per dot in the image. 
    Instantiate with center of mass of the dot and image name
    Class for each individual dot as identified by segmentation and such. 
    QD objects for an image are stored in a list (QD_list) particular to that 
    image, and therefore have a value (QD_index) so they can be easily referenced. 
    Has attributes:
    ## roughly up to date i think ##
        cm -- center of mass
        image -- image its from
        QD_index -- index in the final QD_list (useful for 
            identifying/returning to a particular dot)
        psi4KD -- vector of psi4 -- neighbors defined in radius
        psi4vor -- vector of psi4 -- neighbors defined via Voronoi diagram and 
            weighted by edge length
        psi6V -- psi6 value based on voronoi diagram
        neighborsKD -- QD_index of neighbors within a radius, originally 
            specified by psi4rad(), can be changed via def_neighbors()
        bragg_spots -- the location (y,x) of the bragg spotts in the 
            300x300 FFT of just the dot. 
        orientation -- ['100',angle(s), global superlattice angles from fft(DEGREES two angles))]
        local_orientation_KD -- int. DEGREES orientation of arg(psi4KD)
        local_orientation_vor -- int. DEGREES orientation of arg(psi4vor)
        neighbor_alphas -- list of [[neighbor1, alpha^alpha1],[neighbor2,alpha^alpha2],etc.]
        ave_neighbor_alpha -- float: average misalignment from neighbors
        neighborsvor -- list of QD_index of each other dot that shares an edge with the QD 
        vor_cell -- list of lists, [QD_index, (vertex shared with that QD y,x)] 
            ---- its redundant to have neighborsvor as well, but whatever. 
        vor_area -- area of the voronoi cell containing the QD 
        strain -- for each neighbor sharing edge of voronoi cell, gets distance 
            between the two centers multiplied by edge length (summed over all 
            neighbors), currently not normalized by cell perimeter. (nothing came of this)

    '''
    
    def __init__(self, cm, image_name): 
        self.cm = cm
        self.image = image_name
        self.windowsize = None #radius that i use to make the hann window
        self.psi4KD = None
        self.psi4vor = None
        self.neighborsKD = np.array([])
        self.neighborsvor = np.array([])  
        self.bragg_spots = np.array([])     # should really be deleted 
        self.orientation = np.array([])     # will be a string 
        self.local_orientation_KD = None
        self.local_orientation_vor = None
        self.QD_index = None
        self.neighbor_alphas = None
        self.ave_neighbor_alpha = None
        
    def scale_fft(self):
        '''This method takes the absolute value of self.fft and scales it between 0 and 1. 
        fft_scaled should be used when trying to show the fft or find its peaks.'''
        if np.any(self.fft):
            log_fft = np.where( np.abs(self.fft) !=0,np.log( np.abs(self.fft) ),0)
            self.fft_scaled = (log_fft - np.min(log_fft))  /np.max(log_fft - np.min(log_fft))


'''
state object. This is the object for the image file. 
'''
class state(object):
    """
    state(image, image_name, filesize (tuple), fov (optional) )

    has attributes: 
    filename - string
    orig_image - 2D np array
    imsize - (# pixels)
    fov - field of view
    QD_list - stores QD objects
    SL_orient - overall SL direction if relevant
    QD_list_crop - ** need to decide, but probably array indices of QD_list
    ave_psi4_KD - average of all QD psi4 values with nearest neighbors (NNs) 
                  as calculated by cutoff distance 
    ave_psi4_vor - average of all psi4 values with NNs calculated by a voronoi
                   diagram

    This is the object that holds all the info of a given image. 
    Initialize with an image, filename, filesize (tuple), and field of view (nm, optional and it makes a state object. 

    Run self.segment_dots() and it will (with some user help) segment the 
    image, making a whole bunch of Quantum Dot (QD) objects, and put them in 
    state.QD_list. 

    ANALYSIS functions that i've done before, i have others too, just trying to catalogue them here a little: 
    sep_QD = num_dots(QD_list)
    sep_QD_crop = num_dots(QD_list_crop)
    """  
    def __init__(self, orig_image, filename, filesize, fov = None):
        self.filename = filename
        self.orig_image = orig_image
        self.seg_image = None 
            # will be segmented image once segment_dots is run, 
            # it's nice to have this as an instance variable because it's used 
            # for getting the ffts of dots 
        self.imsize = filesize 
            # int (assumes square image cuz that seems 
            # reasonable) number pixels per side
        self.fov = fov 
            # int, nanometers
        self.QD_list = None 
        self.SL_orient = None 
            # the overall orientation of the superlattice, 
            # picked by the user. (doesnt work well unless large grain)
        self.QD_list_crop = None # the cropped list, dots that have a cm more than 200 pixels from an edge
            # (the analysis can only really be done on QDs that have a full set of neighbors)
        self.ave_psi4_KD = None # the global SL orientation that is the average 
            # of all the psi4KD orientation
        self.ave_psi4_vor = None # same as above, but psi4Vor

    def segment_dots(self):
        '''
        The function that, with some user helps, attempts to segment the state
        into individual QDs and get their orientations. 
        '''

        # first up: pick a threshhold value for applying first 
        # order binary segmentation. 
        # I'd like to have this in its own function, but it screws with 
        # matplotlib and isn't worth it for me. 

        seg_array = [-1]
        im = self.orig_image

        sm = np.min(im)
        big = np.max(im)
        num_bins = np.size(im) / 1000

        fig, ax = plt.subplots()
        init_val = 0

        plt.hist(np.ravel(im),bins=np.linspace(sm,big,num_bins))
        vert = plt.axvline(sm, color = 'k')

        slider_ax = plt.axes([0.25, .03, 0.50, 0.02])
        slid = Slider(slider_ax, 'threshhold', sm, big, valinit=sm)

        def update(val):
            loc = slid.val
            # update curve
            vert.set_xdata(loc)
            # redraw canvas while idle
            fig.canvas.draw_idle()

        # call update function on slider value change
        slid.on_changed(update)

        done_but = plt.axes([0.8, 0.85, 0.2, 0.15])
        button = Button(done_but, 'Done', color='y', hovercolor='0.975')

        def done(event):
            seg_array[:] = list([slid.val])
            plt.close()

        button.on_clicked(done)
        done_but._button = button # needed to prevent garbage collection in a function ? 
        plt.show()

        while True:
            plt.pause(1)

            if seg_array[0] != -1:
                seg_thresh = seg_array[0]
                break


        print('seg threshhold: ', seg_thresh)
        
        ### End of getting first threshhold value. 




        print('this part takes a minute')
        
        # segment the image
        img_segmented = self.segment_im(self.orig_image, seg_thresh)
        self.seg_image = img_segmented

        # uncomment the following lines if you like: 
        print('heres the segmented image... not necessary but nice to see it didnt screw up.')
        show_im(img_segmented[0])
        print("A RuntimeWarning is thrown. Don't worry about it too much... \n also this part takes several minutes.")
        
        # get center of masses and make initial QD_list
        # this throws a runtime warning -- not sure why and as it works I haven't looked into it too much. 
        #self.QD_list = self.center_masses(self.seg_image[0], self.seg_image[1], self.filename)

        print('''about to engage clickable_im so that you can make corrections as necessary.
            \nFor the most part the dots near the edge are cropped so they dont matter as much,
            \nbut they are still helpful to add so the ones not on the edge have a more accurate psi4 value.
            \nRight click to add a dot, delete key near the cm of a dot to delete it.\n''')
        check = 'correct'
        while check != 'done':
            self.clickable_im_cm(self.orig_image, self.QD_list)
            check = str(input('say "done" if done, or anything else to make more edits: '))

        #this corrects for changes made in clickable_im, somehow gets weirded out sometimes
        i = 0
        for dot in self.QD_list:
            dot.QD_index = i
            i += 1

        # so at this point the QD_list is made with all the dots in it. Next steps
        # are defining the neighbors of each QD.  

        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Poor form, I know, but it threw a warning every time it made a tree.
        
        # calculate psi4 KD for all the dots: 
        # normally uses 184 pixels as cutoff radius (this would be filesize / fov * 7.1875 if i want to make this generic ***) 
        # if you want to see what the cutoff looks like at this radius you can use
        # the function show_neighbors_rad(QD_list,184, orig_image) -- maybe not true, but I had that written at one point. unsure if it still works
        self.psi4rad(self.QD_list, 184)

        #psi4vor 
        self.psi4V(self.QD_list)
        
        # gets superlattice orientation with help of user input.
        # I decided it wasn't worth the time to try to pick out the points that corresponded to the 
        # SL after how hard it had been with the bragg spots.
        print('''Okay, time to get the SL orientation from the fft. I have this 
            done by user cuz it seemed awful hard to code it.\nSo how it works, 
            youll select the two corners of a rectangle around the bragg spot, 
            and then itll weight the pixels in that rectangle to find the true 
            spot.\nObviously it therefore works better if you're a little more 
            accurate...\nDon't forget the 'point' selected is the upper left 
            corner of the pixel you click on.''')

        check2 = 'blah'
        while check2 != 'done':
            half_windows = []
            orig_fft = self.get_fft(self.orig_image)
            orig_fft_scaled = np.where(np.abs(orig_fft)!=0,np.log(np.abs(orig_fft)),0)
            self.clickable_im(orig_fft_scaled, half_windows, self.QD_list)
            windows = []

            for window in half_windows: 
                mirror = [[self.imsize - window[1][0], self.imsize - window[1][1]],[self.imsize - window[0][0], self.imsize - window[0][1]]] 
                windows.append(window)
                windows.append(mirror)
            super_pointslst = super_points(self.orig_image, windows)
            # check
            show_im_fits(orig_fft_scaled,super_pointslst)
            check2 = str(input('\nsay "done" if done, or anything else to try again (SL point): '))
        
        self.SL_orient = get_theta12_super(super_pointslst)
        
        # makes fft and getting bragg spots and assigning orientation for each dot 
        # *** this stuff would all have to be made more general. It assumes 300x300 big enough for each dot. 
        print('Okay the rest is automatic, but the getting of orientations takes a while. sorry. \n it tells you every hundred dots')
        counter = 0
        for dot in self.QD_list:
            # .get_orientation is the function that does stuff
            self.get_orientation(dot, self.SL_orient, self.seg_image) 
            if counter % 100 == 0:
                print('getting orientation for dot ', counter, '/',np.shape(self.QD_list)[0])
            counter += 1

        # get the misorientations from neighbors
        for dot in self.QD_list:
            self.get_alpha_neighbors(dot)
            if dot.psi4KD != None:
                dot.local_orientation_KD = get_theta(dot.psi4KD)
            if dot.psi4vor != None:
                dot.local_orientation_vor = get_theta(dot.psi4vor)

        # Specify the cropped qd_list, helpful because most of the analysis i don't want done on the full image
        # normally buffer set to 250 pixels, could be changed
        self.QD_list_crop = self.crop_QD_list(self.QD_list,200,self.orig_image)
        #get global SL orientation from average of local psi4 orientations
        self.ave_psi4_vor = self.get_mean_beta(self.QD_list_crop,'vor')
        self.ave_psi4_KD = self.get_mean_beta(self.QD_list_crop,'KD')
        voronoi_neighbors(self.QD_list)
        psi6V(self.QD_list)
        get_strain(self.QD_list)

        print('\n\nFinished segmenting the dots yayyy :)')

    def get_mean_beta(self,QD_list,kd_vor):
        '''
        gets <betaKD> or <betaVOR> for a QD_list. 
        '''
        thetalist = []
        if kd_vor == 'KD':
            for dot in QD_list:
                theta = dot.local_orientation_KD
                if theta != None:
                    thetalist.append(theta)
        elif kd_vor == 'vor':
            for dot in QD_list:
                theta = dot.local_orientation_vor
                if theta != None:
                    thetalist.append(theta)
        else:
            return("Please input 'KD' or 'vor' for second argument")
        
        return(np.mean(thetalist))

    def get_alpha_neighbors(self,tdot):
        '''
        For a dot looks at neighbors and gets the misalignment between the dot and its neighbors. 
        Assigns to dot.neighbor_alphas a list of misalignment for the dot (referenced by QD_index of other dot)
        and for dot.ave_neighbor_alpha the average absolute value misalignment form neighbors. 
        '''
        a1 = tdot.orientation[1]
        tdot.neighbor_alphas = []
        neighborlist = []
        for index in tdot.neighborsKD:
            neighborlist.append(self.QD_list[index])

        diflist = []
        
        # if the dot doesnt have an orientation it won't have relative misorientations
        if a1:
            for ndot in neighborlist:    
                append = True
                n1 = ndot.orientation[1]
                # if n1 doesnt have an orientation cant get relative misorientations 
                if not n1:
                    append = False
                else: 
                    # if ndot is 100 get average orientation and compare to a1 (or average if its 100 too)
                    if np.shape(n1)[0] == 2:
                        ntheta = ((n1[0]%90) + (n1[1]%90)) / 2
                        if np.shape(a1)[0] == 2:
                            atheta = ((a1[0]%90) + (a1[1]%90)) / 2
                            dif = atheta - ntheta
                        else:
                            dif = a1[0] % 90 - ntheta
                    # here if ndot isn't 100 its the same thing but no average
                    elif np.shape(n1)[0] == 1:
                        if np.shape(a1)[0] == 2:
                            atheta = ((a1[0]%90) + (a1[1]%90))/2
                            dif = atheta - (n1[0] % 90)
                        else: 
                            dif = (a1[0] % 90) - (n1[0] % 90)
                    else:
                        # just a silly check
                        print('in get alpha neighbors, n1 has the wrong number of orientations')
                # because a misorientation of -60 is the same as +30 cuz everything is square lattice
                if append == True:
                    if dif < -45:
                        fdif = 90 + dif
                    elif dif > 45:
                        fdif = dif - 90
                    else:
                        fdif = dif  
                        
                    tdot.neighbor_alphas.append([ndot.QD_index,fdif])
                    diflist.append(fdif)
            # get average value and assign
            if np.shape(diflist)[0] != 0: 
                alpha3 = np.average(diflist)
                
                if alpha3 < -45:
                    avealpha = 90 + alpha3
                elif alpha3 > 45:
                    avealpha = alpha3 - 90
                else:
                    avealpha = alpha3  
                tdot.ave_neighbor_alpha = avealpha 
            
    def crop_QD_list(self,QD_list,buffer,orig_image):
        '''Because the Voronoi implementation has problems with edges, this crops the image
        so only QD objects in the interior (defined by "buffer" number of pixels) are included. 
        Buffer should be at minimum 1 QD diameter. '''
        tlist = []
        for dot in QD_list:
            y, x = dot.cm[0], dot.cm[1]
            ymin = xmin = 0
            ymax, xmax = np.shape(orig_image)
            if y > ymin + buffer and y < ymax - buffer and x > xmin + buffer and x < xmax - buffer:
                tlist.append(dot)
        # if you want to see the image with only the relevant dots highlighted:
        # show_im_circs_dots(orig_image,tlist,'no_plot')
        return(tlist)

    def hann(self, im, cp):
        '''
        takes in a masked image (from the mask function), that is one dot with everything else masked. 
        First gets average diameter of not masked dot, then uses that to make a subwindow, applies the 
        hann curve to the window and then takes fft of that window
        '''
        # get radius of window and use to make hann window
        # this could be used instead of d... just substitute it in s1...s4
        A = np.nonzero(im)
        h = np.max(A[0])-np.min(A[0]) #maximum height of window
        w = np.max(A[1])-np.min(A[1]) 
        #r is really a diameter
        r = int((h+w-(h+w)/10)/2)
        r = min(r, 300)

        # I tried doing something a little fancier and making the window based 
        # off the average diameter of the mask, but it didnt seem to work as well for the fft. 
        # I'm leaving it in in case it's helpful for the future. 
        # d = average_diam(im)

        # gets boundaries of new image border, cant be outside of original image, and centered on centerpoint.
        s1 = min(max(int(cp[0]-r/2),0), 4095)
        s2 = min(max(int(cp[0]+r/2),0), 4095)
        s3 = min(max(int(cp[1]-r/2),0), 4095)
        s4 = min(max(int(cp[1]+r/2),0), 4095)

        # make subimage of just the area around the not black part
        subimg1 = im[s1 : s2] 
        subimg =[row[s3 : s4] for row in subimg1]

        # makes the hann windows of the horizontal and vertical height
        # might be different if dot is on a border
        hanw = np.hanning(abs(s2-s1))
        hanh = np.hanning(abs(s4-s3))

        # apply hann to columns
        h1 = np.array(np.copy(subimg))
        for i in range(np.shape(subimg[0])[0]):
            h1[:,i] = np.array(subimg)[:,i] * hanw

        # apply hann to rows
        hf = np.array([row * hanh for row in h1]) 

        # pad image to 300x300 so all are same shape, and image is centered
        wid = np.shape(hf)[0]
        ht = np.shape(hf)[1]       
        w1 = int(np.abs(np.floor((300 - ht)/2)))
        w2 = int(np.abs(np.ceil((300 - ht)/2)))
        t1 = int(np.abs(np.floor((300 - wid)/2)))
        t2 = int(np.abs(np.ceil((300 - wid)/2)))

        hfp = np.pad(hf,((t1,t2),(w1,w2)),'constant')  
        # hfp is final hanned image, can show_im(hfp) to see effect of window

        ffth = get_fft(hfp)


        return(ffth,r)

    def hann2(self, im, diam, cp):
        '''
        takes in a masked image (from the mask function), that is one dot with everything else masked. 
        First gets average diameter of not masked dot, then uses that to make a subwindow, applies the 
        hann curve to the window and then takes fft of that window
        '''
        
        r = diam 

        # gets boundaries of new image border, cant be outside of original image, and centered on centerpoint.
        s1 = min(max(int(cp[0]-r/2),0), 4095)
        s2 = min(max(int(cp[0]+r/2),0), 4095)
        s3 = min(max(int(cp[1]-r/2),0), 4095)
        s4 = min(max(int(cp[1]+r/2),0), 4095)

        # make subimage of just the area around the not black part
        subimg1 = im[s1 : s2] 
        subimg =[row[s3 : s4] for row in subimg1]

        # makes the hann windows of the horizontal and vertical height
        # might be different if dot is on a border
        hanw = np.hanning(abs(s2-s1))
        hanh = np.hanning(abs(s4-s3))

        # apply hann to columns
        h1 = np.array(np.copy(subimg))
        for i in range(np.shape(subimg[0])[0]):
            h1[:,i] = np.array(subimg)[:,i] * hanw

        # apply hann to rows
        hf = np.array([row * hanh for row in h1]) 

        # pad image to 300x300 so all are same shape, and image is centered
        wid = np.shape(hf)[0]
        ht = np.shape(hf)[1]       
        w1 = int(np.abs(np.floor((300 - ht)/2)))
        w2 = int(np.abs(np.ceil((300 - ht)/2)))
        t1 = int(np.abs(np.floor((300 - wid)/2)))
        t2 = int(np.abs(np.ceil((300 - wid)/2)))

        hfp = np.pad(hf,((t1,t2),(w1,w2)),'constant')  
        # hfp is final hanned image, can show_im(hfp) to see effect of window

        ffth = get_fft(hfp)
        return(ffth,r)
    

    def segment_im(self, im, threshhold):
        '''This function takes in the original image and segments it using a watershed algorithm and the threshhold value
         Returns the segmented image and number of marked dots
         Also has the option of saving the image to file
         Time to run: a minute or two

         ### this function used to work fine, just recentl I've found it started doing some odd things and sometimes making a dot
         super small in the middle, not sure why. I'm looking into that or will be soon'''

         ### Okay! So it's definitely an issue with the actual watershed implemenation, and I'm not sure why!
         ### this is currently in progress hence the comments and such. 

        # threshholds the image and runs some openings and closings to get it into a good format to use a blob finder on
        threshholded_im = im > threshhold
        opened_im = binary_opening(threshholded_im,iterations = 30)
        distanceT = ndi.distance_transform_edt(opened_im)
        open3 = binary_opening(distanceT, iterations = 30)
        distance2 = ndi.distance_transform_edt(open3)

        # blob finder finds an approximate center for each dot
        blobs = blob_dog(distance2, max_sigma=5, min_sigma = 1, threshold=.01)

        # uncomment everything about "points" if you want it to display the sources for the watershed algorithm
        points = []
        
        points = np.array(points)


        # fine through here
        # creates an array of 0's the same size as original image
        centers = np.zeros(shape = np.shape(im), dtype=bool)

        # for each centerpoint found from the blob finder marks that point as 1 in the 0s array
        for blob in blobs: 
            y, x, r = blob
            # a border is created around the images, this moves the sources off the border. 
            if y > 4091: 
                y = 4090
            if y < 5:
                y = 5
            if x > 4091: 
                x = 4090
            if x < 5:
                x = 5
            centers[int(y),int(x)] = True
            # points.append([int(y),int(x)])

        # uses the ndi.label function so you get the centers in the format the watershed algorithm wants
        markers = ndi.label(centers) 

        # does morphology stuff to get a good mask 
        open1 = binary_opening(threshholded_im, iterations = 2)
        close1 = binary_closing(open1, iterations = 4)
        mask = close1 
        # overlay of markers on mask
        # show_im_fits(mask, np.array(points))temp

        labels_ws = watershed(-distance2, markers[0], mask= mask)

        return(labels_ws, markers[1])
    
    def center_masses(self, im, num, img_name):
        '''takes in the output of segment_im and gives you the centers of masses for all dat ish'''
        temp = []
        pixlist = []

        for i in range(1,int(num)):
            dot_im = np.where(im == i, 1, 0)
            npix = np.count_nonzero(dot_im)
            # pixlist.append(npix)

            if npix > 15000:
                cm = center_of_mass(dot_im)
                temp.append(cm)

        # get_histo(pixlist, 0, np.max(pixlist), 200)
        ret = []    
        ind = 0
        for i in temp:
            if not np.isnan(i[0]):
                ret.append(QD(i,img_name))
                ret[-1].QD_index = ind
            ind += 1 

        return(ret)
    
    def clickable_im(self, image, retlist, QD_list):
        '''
        Okay, so this is an example of code I got to work but is really bad. 
        I really wasn't sure how to get all this stuff to do what i wanted it to, 
        so this is what i managed to patch together.

        Use
        ----------
        rightclick to add point to retlist
        centerclick to exit.
        To define a window [[y1,x1],[y2,x2]] 'w' to define [y1,x1], 'e' to 
        define [y2,x2], 't' to confirm and assign the window. '''
        im = image

        def OnClick(event):
            if event.button == 3:
                y = event.ydata
                x = event.xdata
                print('adding [',y,', ',x,']')
                retlist = np.append(retlist,[y,x])
            if event.button == 2:
                print('closing')
                plt.close()

        def on_key(event):

            if event.key == 'w':
                global start
                global scheck
                scheck = True
                start = [event.xdata, event.ydata]
                event.start = 5

            if event.key == 'e':
                global end
                global echeck 
                echeck = True
                end = [event.xdata, event.ydata]
                event.end = 29

            if event.key == 't':
                if scheck and echeck: 
                    add = [start, end]
                    print('Window is :', add)
                    retlist.append(add)            
                    start = None
                    end = None
                else: 
                    print('yo! you didnt define the corners yet ("w" for upper left, "e" for lower right)')

            if event.key == 'm':
                y = event.ydata
                x = event.xdata
                for dot in QD_list:
                    yD = dot.cm[0]
                    xD = dot.cm[1]
                    if y - 20 < yD and y + 20 > yD and x - 20 < xD and x + 20 > xD:
                        print('QD_index: ',dot.QD_index, 'orientation: ',dot.orientation)

        fig, ax = plt.subplots()
        implot = ax.matshow(im, cmap = 'gray')
        connection_id = fig.canvas.mpl_connect('button_press_event', OnClick)
        cid = fig.canvas.mpl_connect('key_press_event', on_key)

        #set the size of the matplotlib figure in data units, so that it doesn't
        #auto-resize (which it will be default on the first drawn item)
        # fig.tight_layout()
        fig.set_tight_layout(True)
        plt.show()
        plt.pause(1)

    def clickable_im_cm(self, image, QD_list):
        '''a variation of clickable_im()
        this takes in a QD_list and image to help the dot finder.'''
        im = image
        peaks = []
        for dot in QD_list:
            peaks.append(dot.cm)
        peaks = np.array(peaks)

        def OnClick(event):
            if event.button == 3:
                y = event.ydata
                x = event.xdata

                print('adding [',y,', ',x,']')
                ndot = QD([y,x],QD_list[0].image)

                QD_list.append(ndot)
                print(QD_list[-1].cm)

            if event.button == 2:
                print('closing')
                plt.close()
                return(QD_list)

        def on_key(event):     
            if event.key == 'delete':
                y = event.ydata
                x = event.xdata
                i = 0
                print('delete pressed')
                for dot in QD_list:
                    yD = dot.cm[0]
                    xD = dot.cm[1]
                    if y - 20 < yD and y + 20 > yD and x - 20 < xD and x + 20 > xD:
                        print('delete ',dot.cm)
                        QD_list.pop(i)
                    i += 1

            if event.key == 'm':
                y = event.ydata
                x = event.xdata
                for dot in QD_list:
                    yD = dot.cm[0]
                    xD = dot.cm[1]
                    if y - 20 < yD and y + 20 > yD and x - 20 < xD and x + 20 > xD:
                        print('QD_index: ',dot.QD_index, 'orientation: ',dot.orientation)

        print('new click \n')
        fig, ax = plt.subplots()
        
        implot = ax.matshow(im, cmap = 'gray')
        ax.plot(peaks[:,1],peaks[:,0],
            linestyle='None',marker='o',color='r',fillstyle='none')

        connection_id = fig.canvas.mpl_connect('button_press_event', OnClick)
        cid = fig.canvas.mpl_connect('key_press_event', on_key)

        #set the size of the matplotlib figure in data units, so that it doesn't
        #auto-resize (which it will be default on the first drawn item)
        # plt.tight_layout()
        fig.set_tight_layout(True)
        plt.show()

    def psi4rad(self, QD_list, rad=184):
        '''
        defining neighbors as all points within 184 pixels (unless otherwise specified, 
        and this I determined by looking for a maximum value that doesnt give dots 
        having 5 neighbors when they should have 4 ***(see below for the code that 
        produces those images)).

        this function calculates the psi-4 value for each point and assigns the 
        psi4KD value to each dot for which it can be calculated (must have minimum 
        2 neighbors), as well as the list of QD_index for neighboring dots to dot.neighborsKD

        raises a shit ton of deppreciation warnings, one every time it makes a 
        KD tree. I'm not sure why... so i turn the warning off before i run this....
        '''
        cmlist = np.array([dot.cm for dot in QD_list])


        tree = sklearn.neighbors.KDTree(cmlist, leaf_size=6)
        retlist = []
        
        for dot in QD_list:
            neighbors = findNeighborsKD(tree, dot.cm, rad,cmlist)

            neighborlist = []
            for cm in neighbors:
                for dot2 in QD_list:
                    if np.all(dot2.cm == cm):
                        neighborlist.append(dot2.QD_index)

            dot.neighborsKD = neighborlist
            psi4 = 0

            if np.shape(neighbors)[0] > 1:

                for neighbor in neighbors:
                    theta = np.arctan2((neighbor[1]-dot.cm[1]),(neighbor[0]-dot.cm[0]))
                    num = np.exp(4j*theta)
                    psi4 += num

                try:
                    psi4n = psi4 / np.shape(neighbors)[0]
                except ZeroDivisionError:
                    psi4n = 0

                dot.psi4KD = psi4n

    def psi4V(self, QD_list):
        '''
        The Voronoi implementation of getting psi_4. This uses a lot of functions I put in the Utility Functions cell.
        since i just realized this might not be written anywhere else in this code: 
        It uses the normal psi4 equation where the neighbor cells are defined as points that share a border of the 
        Voronoi diagram with the point of interest. Then for that part of the sum it is weighted by the edge length
        so larger edges correlating with more important neighbors bare more weight, and then rather than dividing by 
        total neighbors you divide by total border length so everything normalizes to 1. 
        '''

        cmlist = np.array([dot.cm for dot in QD_list])

        retlist = []

        # fucking magic. 
        vor = Voronoi(cmlist)

        # Cuz the way Voronoi works it keeps track of points by index in the original list so I need to keep track too.
        i = 0

        for dot in QD_list: 
            point = dot.cm
            psi4 = 0
            A = get_vertices(i,cmlist[i], vor)
            i += 1 

            # needs to be bordering more than two cells to be considered legit
            if np.shape(A)[0] > 2:
                edges = get_edges(A)
                totL = 0

                # need to calculate total border length first
                for edge in edges: 
                    length = plength(edge)
                    totL += length

                # this loops over each edge and calculates its component of the psi4 total
                for edge in edges: 
                    length = plength(edge)

                    # the slope between the two points is the inverse of the slope of the edge
                    dx = edge[0,0]-edge[1,0]
                    dy = edge[0,1]-edge[1,1]
                    theta = np.arctan2(dy,dx)

                    # component of psi4 for this point due to this individual edge
                    psip = length/totL * np.exp(4j*theta)
                    psi4 += psip

                # assign dot to have psi4vor value 
                dot.psi4vor = psi4
        
    def get_fft(self,im):
        '''get fast forier transform of an image.'''
        return np.fft.fftshift(np.fft.fft2(im))
    
    def dot_fft(self, dot, orig_image):
        '''
        returns a tuple of (window around dot, processed window turned into fft, r (hann window diameter)) for a dot. 
        i think its doing something unnecessary/twice, v redundant with hann() ... but its not run all that often so doesnt matter too much
        ''' 
        cm = dot.cm
        # r = mask2(self.orig_image, cm)
        # fft, windowsize = hann2(self.orig_image, 2*r, cm) 
        mask_im = mask(self.orig_image, self.seg_image, cm)
        fft, windowsize = self.hann(mask_im, cm) # might be something to change (r vs d) 
        
        # get radius of window and use to make hann window
        # this could be used instead of d... just substitute it in s1...s4
        d = windowsize
        r = d/2
        cp = cm
        # gets boundaries of new image border, cant be outside of original image, and centered on centerpoint.
        s1 = min(max(int(cp[0]-r),0), 4095)
        s2 = min(max(int(cp[0]+r),0), 4095)
        s3 = min(max(int(cp[1]-r),0), 4095)
        s4 = min(max(int(cp[1]+r),0), 4095)

        # centerpoint in new subimage coordinates 
        cpnew = ['y', 'x']
        if s1 > 0:
            cpnew[0] = r
        else:
            cpnew[0] = s2 - r
        if s3 > 0: 
            cpnew[1] = r 
        else: 
            cpnew[1] = s4 - r

        # print('h: ',h,'w: ',w,'r: ',r,'s1: ',s1,'s2: ',s2,'s3: ',s3,'s4: ',s4)

        # make subimage of just the area around the not black part
        
        subimg1 = mask_im[s1 : s2] 
        subimg =[row[s3 : s4] for row in subimg1]

        cpnew_wide = np.copy(cpnew)
        if s1 >= 75 and s2 <= 4020:
            subimg_wide = orig_image[s1 -75 : s2 + 75]
            cpnew_wide[0] += 75 
        elif s1 >= 75: 
            subimg_wide = orig_image[s1 -75 : 4095]
            cpnew_wide[0] += 75 
        else: 
            subimg_wide = orig_image[0 : s2 + 75]
            cpnew_wide[0] += s1
      
        if s3 >= 75 and s4 <= 4020:
            subimg_wide = [row[s3 - 75 : s4 + 75] for row in subimg_wide]
            cpnew_wide[1] += 75 
        elif s3 >= 75: 
            subimg_wide = [row[s3 - 75 : 4095] for row in subimg_wide]
            cpnew_wide[1] += 75 
        else: 
            subimg_wide = [row[0 : s4 + 75] for row in subimg_wide]
            cpnew_wide[1] += s3


        # makes the hann windows of the horizontal and vertical height
        # might be different if dot is on a border
        hanw = np.hanning(abs(s2-s1))
        hanh = np.hanning(abs(s4-s3))

        # apply hann to columns
        h1 = np.array(np.copy(subimg))
        for i in range(np.shape(subimg[0])[0]):
            h1[:,i] = np.array(subimg)[:,i] * hanw

        # apply hann to rows
        hf = np.array([row * hanh for row in h1]) 

        # pad image to 300x300 so all are same shape, and image is centered
        wid = np.shape(hf)[0]
        ht = np.shape(hf)[1]       
        w1 = int(np.abs(np.floor((300 - ht)/2)))
        w2 = int(np.abs(np.ceil((300 - ht)/2)))
        t1 = int(np.abs(np.floor((300 - wid)/2)))
        t2 = int(np.abs(np.ceil((300 - wid)/2)))

        hfp = np.pad(hf,((t1,t2),(w1,w2)),'constant')  
        # show_im(hfp)
        # hfp is final hanned image, can show_im(hfp) to see effect of window

        ffth = get_fft(hfp)

        subimg = (subimg, cpnew)
        subimg_wide = (subimg_wide, cpnew_wide)

        return(subimg, ffth, d, subimg_wide)
        
    def show_dot_fft(self, dot, subimg, fft, hann_diam,subimg_wide, save = False):
        '''I like this one. It plots together the dot with the hann window overlaid, a little larger area for context, then the fft.'''
        fig=plt.figure()
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(133)
        ax3 = plt.subplot(132)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        fig.subplots_adjust(wspace=0.05)

        # Plot ffts
        cmap2 = 'gray'
        display_fft = np.where(np.abs(fft)!=0,np.log(np.abs(fft)),0)
        ax1.matshow(subimg[0],cmap='gray')
        ax2.matshow(display_fft,cmap='gray')
        ax3.matshow(subimg_wide[0], cmap='gray')
        
        hann_win = plt.Circle((subimg[1][1], subimg[1][0]), radius = hann_diam/2, color='r',fill=False)
        # hann_win = plt.Circle((hann_diam/2, hann_diam/2), radius = hann_diam/2, color='r',fill=False)
        # hann_win_wide = plt.Circle((75 + hann_diam/2, 75 + hann_diam/2), radius = hann_diam/2, color='r',fill=False)
        hann_win_wide = plt.Circle((subimg_wide[1][1], subimg_wide[1][0]), radius = hann_diam/2, color='r',fill=False)
 
        ax1.set_title('image: ' + dot.image + "\nQD_index: " + str(dot.QD_index) + '\norientation: ' + str(dot.orientation[0]) + '\n location:\n' + str('%.2f' % dot.cm[0]) +' , ' + str('%.2f' % dot.cm[1]) + ')' )
        ax3.set_title("Wider image w/\nHann Diam")
        ax2.set_title('resulting FFT')
        
        ax1.add_artist(hann_win)
        ax3.add_artist(hann_win_wide)
        ### to save for figure
        if save:
            # fig2, axx = plt.subplots()
            # axx.matshow(subimg_wide[0], cmap='gray')
            # axx.axis('off')
            plt.savefig('/Users/arthur/Desktop/Arthur/presentables/m&m_2017/poster/images/triple_window.png', dpi = 300, bbox_inches = 'tight')
            # plt.savefig("../../rename_plz.tif",dpi=200,bbox_inches='tight')
        plt.show()

    def get_orientation(self, dot, SL_orient, seg_im):
        ''' gets the orientation of a dot and assigns it. 
        In this function are the big ones get_points and def_orientation, both of which are a lot...
        '''
        cm = dot.cm
        mask_im = mask(self.orig_image, seg_im, cm)
        fft, windowsize = self.hann(mask_im, cm) # might be something to change (r vs d) 
        dot.windowsize = windowsize
        if np.shape(fft)[0] == 0:
            return('\n\n\n why is fft 0?')
        self.get_points_v2(dot, fft, SL_orient)

    def get_points_v2(self, dot, fft, super_thetas, thresh = 8, recur = 0, force = 0):
        '''
        Going on the basis that the frequency for 100 dots puts their bragg peaks 37 pixels
        from the center. The angles between the dots will be 90, but due to scan coil 
        misalignment there might be a couple degrees of error (for all the orientations).
            1n0 (large n): dots will have the same spacing, 1nn dots will be at sqrt[2]*37 = 53.3 
        pixels, and because only 2 bragg peaks 180 degrees between. 
            1nn (large or small n) : spacing sqrt[2]*37, 180 degrees. For these the AL orientation is 45 off of the 
        lattice orientation    
            110:  at 37/sqrt[2] for 4/6 dots and (it's not a perfect hexagon) 37 for the other 2. Four 
        of the internal angles will be 54.7 degrees (plus minus error) and the other two 70.6
            1,n+1,n (large n): 
            111: spacing (uncomfirmed as I've yet to see one) Sqrt[3/8]*37 for all dots (true
        hexagon). Internal angles all 60.
            
        So that's the info I'm going on. The problem is many of the dots are in between orientations,
        and my solution to that is pick whatever dots are the brightest and select those. The one 
        exception perhaps is between 100 and 1nn, as there are some that seem like they should be
        100 but might be a little dimmer in one direction. 
        
        Plan is to apply blob finder and keep reducing intensities until plausible dots are found. 
        '''
              
        def two_points(blobs1, distlist, force = 0):
            # differentiate between 1n0, 1nn, (1,n+1,n)
            dot.bragg_spots = weight_spots(fft, blobs1, 6)
            if distlist[0] < 34.6:
                # you can't get the orientation of a 1, n+1, n
                # point0y, point0x = dot.bragg_spots[0][0]-150, dot.bragg_spots[0][1]-150
                # theta = (np.arctan2(point0y, point0x) / np.pi * 180) - 45 
                # theta1 = theta % 90
                dot.orientation = ['1,n+1,n', [], super_thetas]

            elif distlist[0] < 41: # this boundary is a little high, but otherwise it was excluding some 
                # dots that looked in real space to be 1n0s. I don't know of any other 
                # orientations that would have a lattice spacing that it'd be bumping into. 
                # check if maybe 100
                if force: 
                    point0y, point0x = dot.bragg_spots[0][0]-150, dot.bragg_spots[0][1]-150
                    theta = np.arctan2(point0y, point0x) / np.pi * 180 
                    theta1 = theta % 90
                    dot.orientation = ['1n0', [theta1], super_thetas]
                else: 
                    blobs2 = blob_log(fft, max_sigma= 10, min_sigma = 1, threshold = thresh*2/3, overlap = 0.001)
                    blobs100 = []
                    distlist100 = []
                    for blob in blobs2:
                        dist = np.sqrt((blob[0] - 150)**2 + (blob[1] - 150)**2)
                        if dist > 22 and dist < 60: 
                            blobs100.append(blob)
                            distlist100.append(dist)

                    num_dots_100 = np.shape(blobs100)[0]
                    distlist100 = np.array(distlist100) 

                    if num_dots_100 == 2: 
                        # its really a 1n0
                        point0y, point0x = dot.bragg_spots[0][0]-150, dot.bragg_spots[0][1]-150
                        theta = np.arctan2(point0y, point0x) / np.pi * 180 
                        theta1 = theta % 90
                        dot.orientation = ['1n0', [theta1], super_thetas]

                    elif num_dots_100 == 4: 
                        # check if they're the same rad
                        t1 = ((distlist100 >= distlist100[0]-3).sum() == distlist100.size).astype(np.int)
                        t2 = ((distlist100 <= distlist100[0]+3).sum() == distlist100.size).astype(np.int)
                        if t1 and t2: #they're the same radius, 
                            four_points_same_dist(blobs100, distlist100, 1)
                        else: 
                            # really a 1n0
                            point0y, point0x = dot.bragg_spots[0][0]-150, dot.bragg_spots[0][1]-150
                            theta = np.arctan2(point0y, point0x) / np.pi * 180 
                            theta1 = theta % 90
                            dot.orientation = ['1n0', [theta1], super_thetas]
                    else: 
                        #get only the ones within 100/1n0 range and do 2/four dots
                        blobs3 = []
                        distlist2 = []
                        k = np.where(distlist100 < 40)[0]
                        k1 = np.where(distlist100[k] > 35)[0]
                        keeplist = []
                        for i in k1:
                            keeplist.append(k[i])

                        for index in keeplist:
                            blobs3.append(blobs100[index])
                            distlist2.append(distlist100[index])

                        if np.shape(blobs3)[0] == 4:
                            if four_points_same_dist(blobs3, distlist2, 1):
                                # its really a 1n0
                                point0y, point0x = dot.bragg_spots[0][0]-150, dot.bragg_spots[0][1]-150
                                theta = np.arctan2(point0y, point0x) / np.pi * 180 
                                theta1 = theta % 90
                                dot.orientation = ['1n0', [theta1], super_thetas]

                        else:
                             # its really a 1n0
                            point0y, point0x = dot.bragg_spots[0][0]-150, dot.bragg_spots[0][1]-150
                            theta = np.arctan2(point0y, point0x) / np.pi * 180 
                            theta1 = theta % 90
                            dot.orientation = ['1n0', [theta1], super_thetas]

            elif distlist[0] > 49 and distlist[0] < 56.5: 
                point0y, point0x = dot.bragg_spots[0][0]-150, dot.bragg_spots[0][1]-150
                theta = (np.arctan2(point0y, point0x) / np.pi * 180) - 45
                theta1 = theta % 90
                dot.orientation = ['1nn', [theta1], super_thetas]
                
            else: 
                print('two dot problem. radius doesnt fit with any. QD_index', dot.QD_index)
                dot.bragg_spots = weight_spots(fft, blobs1, 6)
                dot.orientation = ['fail_2_spots',[], super_thetas]
            
        def four_points_same_dist(blobs1, distlist, force = 0):
            distlist = np.array(distlist)
            t1 = ((distlist >= distlist[0]-3).sum() == distlist.size).astype(np.int)
            t2 = ((distlist <= distlist[0]+3).sum() == distlist.size).astype(np.int)
            if t1 and t2: #they're the same radius, 
                t1 = ((distlist >= 34.5).sum() == distlist.size).astype(np.int)
                t2 = ((distlist <= 41).sum() == distlist.size).astype(np.int)
                if t1 and t2:
                    # print('its a 100')
                    dot.bragg_spots = weight_spots(fft, blobs1, 6)
                   
                    tr, lr = 0, 0
                    for spot in dot.bragg_spots:
                        if spot[0] <= 150 and spot[1] >=150:
                            tr = (spot[0], spot[1])
                        if spot[0] >= 150 and spot[1] >= 150:
                            lr = (spot[0], spot[1])
                    if lr == 0 or tr == 0: # happens when it picks very wrong dots for a 100, really a 1n0
                        self.get_points_v2(dot, fft, super_thetas, thresh + 1, recur + 1, 1)
                    else:    
                        theta1 = np.arctan2(tr[0] - 150, tr[1] - 150) / np.pi * 180
                        theta2 = np.arctan2(lr[0] - 150, lr[1] - 150) / np.pi * 180
                        
                        dot.orientation = ['100', [theta2, theta1], super_thetas]
                else: #so theyre the same radius but not 100 -> probably amorphous run higher thresh
                    self.get_points_v2(dot, fft, super_thetas, thresh + 1, recur + 1)
                
            else: #not a 100 so assign to 1n0
                t1 = ((distlist <= 26).sum() == distlist.size).astype(np.int)
                if t1: 
                    print('holy shit its a 111 with only 4 dots?')
                    dot.orientation = ['111', [], super_thetas]
                    dot.bragg_spots = weight_spots(fft, blobs1, 6)
                else:
                    if force:
                        return True

                    else: 
                        self.get_points_v2(dot, fft, super_thetas, thresh + 1, recur + 1)

        if recur > 35: 
            print('recur = ', recur, ', Double check that this is amorphous please. Dot QD_index: ', dot.QD_index)
            dot.orientation = ['amorphous', [],super_thetas]
            return

        # all the bragg spots should be about the same size, so I should be able to narrow in on what the appropriate
        # min and max sigma values are which won't change 
        # threshhold should be the one I decrease slowly over time, overlap should be zero
        fft = np.abs(fft)
        distlist = []
        
        blobs0 = blob_log(fft, max_sigma= 10, min_sigma = 1, threshold = thresh, overlap = 0.001)
        blobs1 = []
        for blob in blobs0:
            dist = np.sqrt((blob[0] - 150)**2 + (blob[1] - 150)**2)
            if dist > 22 and dist < 60: 
                blobs1.append(blob)
                distlist.append(dist)
        # #helpful debug
        # print('\n recur = ', recur, ' thresh = ', thresh)
        # for i in range(np.shape(blobs1)[0]): 
        #     print(blobs1[i], ' dist: ', distlist[i])

        ### If there aren't any found points
        if not blobs1:
            if recur <= 4:
                # print('recurring cuz it might not actually be amorphous')
                self.get_points_v2(dot, fft, super_thetas, thresh - 1, recur + 1)
            elif recur == 5:
                # print('amorphous')
                # show_fft(fft)
                dot.orientation = ['amorphous', [], super_thetas]
            else:
                # ill explain this in a moment. but basically you sometimes get a situation where its confused
                # between 1nn and 1n0, in these cases in realspace it looks like 1n0, so i want it to 
                # define it as such. 
                # If it gets here again, then it's because it's amorphous. 
                if force: 
                    dot.orientation = ['amorphous', [], super_thetas]
                else: 
                    self.get_points_v2(dot, fft, super_thetas, thresh - 1, recur + 1, 1)
                
        ### So there are found points  
        else: 
            distlist = np.array(distlist)
            # checks to see if all of the distances are within three of the first distance in the list
            t1 = ((distlist >= distlist[0]-3).sum() == distlist.size).astype(np.int)
            t2 = ((distlist <= distlist[0]+3).sum() == distlist.size).astype(np.int)
            if t1 and t2:
                # The dots are all at the same radius
                if np.shape(blobs1)[0] == 2:
                    two_points(blobs1, distlist, force)
                    
                elif np.shape(blobs1)[0] == 4: 
                   four_points_same_dist(blobs1, distlist, force)
                        
                elif np.shape(blobs1)[0] == 6:
                    print('hot damn is this a 111?')
                    # maybe this should check radius 
                    dot.orientation = ['111', [], super_thetas]
                    dot.bragg_spots = weight_spots(fft, blobs1, 6) 
                else:
                    print('wrong number of dots, equal radius. QD_index: ', dot.QD_index)
                    dor.orientation = ['amorphous', [], super_thetas]
            else:
                # So they aren't at equal radius spacing. Either its a 110 or there are too many points. 
                if np.shape(blobs1)[0] == 6:
                    #  Check if it's a 110
                    t1 = ((distlist <= 34.5).sum() == 4).astype(np.int)
                    t2 = ((distlist >= 34.5).sum() == 2).astype(np.int)
                    if t1 and t2: 
                        # The spacing is right, it is
                        if np.min(distlist) > 29 and np.max(distlist) < 40:
                            dot.orientation = ['110', [], super_thetas]
                            dot.bragg_spots = weight_spots(fft, blobs1, 6)
                            
                        else:
                            # The spacing isn't right for a 110 so run it again 
                            self.get_points_v2(dot, fft, super_thetas, thresh = thresh +1, recur = recur + 1)
                    else:
                        # The distribution isn't right for a 110 so run it again
                        self.get_points_v2(dot, fft, super_thetas, thresh = thresh +1, recur = recur + 1)
                        
                else: 
                    # check radius to see if they fit with 110 (sometimes it would only find 4/6 points)
                    if np.shape(blobs1)[0] == 4:
                        minn = np.min(distlist)
                        maxx = np.max(distlist)
                        if minn > 29 and minn < 35 and maxx < 40 and maxx > 34:
                            # So they pass the first test, but sometimes they should really be a 1n0 or 1 n+1 n
                            # If, after increaseing thresh, there are still 4 dots then probably 110,
                            # if two disapear then take the brighter ones and send to two dots
                            # if they all disapear then they're all dim, and given the orientation should be 
                            # classified as amorphous
                            blobs3 = blob_log(fft, max_sigma= 10, min_sigma = 1, threshold = thresh + 1.5, overlap = 0.001)
                            blobs110 = []
                            distlist110 = []
                            for blob in blobs3:
                                dist = np.sqrt((blob[0] - 150)**2 + (blob[1] - 150)**2)
                                if dist > 22 and dist < 60: 
                                    blobs110.append(blob)
                                    distlist110.append(dist)
                            num_dots_110 = np.shape(blobs110)[0]          

                            if num_dots_110 == 2: 
                                # its not a 110
                                two_points(blobs110, distlist110, force)
                            elif num_dots_110 == 4: 
                                dot.orientation = ['110', [], super_thetas]
                                dot.bragg_spots = weight_spots(fft, blobs1, 6)
                            else: 
                                print('Double check this amorphous please (and not 110). QD_index: ', dot.QD_index)
                                dot.orientation = ['amorphous', [], super_thetas]

                        elif force:
                            # this is the tricky part. Force being on means they're about the same 
                            # brightness but in a bad orientation. 

                            blobs2 = []
                            distlist2 = np.copy(distlist)
                            min_index = np.where(distlist == np.min(distlist))[0]
                            max_index = np.where(distlist == np.max(distlist))[0]
                            for index in min_index:
                                blobs2.append(blobs1[index])
                
                            distlist2 = np.delete(distlist,max_index)    
                            two_points(blobs2, distlist2, 1)

                        else: 
                            self.get_points_v2(dot, fft, super_thetas, thresh + 1, recur + 1)
                    
                    else: 
                        # if not six or four points and at different spacing, there's too many
                        self.get_points_v2(dot, fft, super_thetas, thresh + 1, recur + 1)

        if recur == 0:
            dot.bragg_spots = np.array(dot.bragg_spots)
            # Overlay of points 
            # show_fft(fft, dot.bragg_spots)
            
        return

