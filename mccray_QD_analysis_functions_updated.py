'''
Arthur McCray
arthurmccray95@gmail.com
Summer 2016/2017

This is a bunch of functions used in my quantum dot analysis. 

All the input images I had were 160nm field of view 4096x4096 HAADF STEM images 
of PbSe QDs in an epitaxially fused lattice. My code is really meant for those 
images, but could presumably be modified fairly easily. The main type of thing 
that would have to be modified is due to number of pixels per dot, e.g. all the 
ffts of individual dots had to be of the same size window so I padded them to 
300x300 pixels as the largest dots were ~275 pixels across. Where I think of it 
I'm marking things that would have to be changed/generalized with a ***, but I 
probably dont get all of them. 
-- i quickly forgot about that and didn't mark much. 

I define most of the angles/terms I use at the top of the jupyter notebook.
Also you should ctrl-f "save" and rename all your file paths. 

Oh a couple other things, it takes a while (no doubt because of my inefficient code)
to actually make the state objects (there's one for each image you're working with), 
so I highly reccomend saving them. That's all done in the jupyter notebooks where 
I did my daily workflow stuff, so if there are questions please email me. 
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
    Soooo, this is the object that holds all the info of a given image. 
    Initialize with an image, filename, filesize (#pixels per size, this assumes 
    square image), and field of view (nm) and it makes a state object.
    Run self.segment_dots() and it will (with some user help) segment the image, 
    making a whole bunch of Quantum Dot (QD) objects, and put em in a QD_list.

    That gets the initial analysis out of the way (gives values to all the variables 
    each QD object has as discussed above), and after that you can run more detailed 
    analysis or make plots or histograms or whatever. I have some (awful and messy) 
    code for that, but i havent cleaned it up too much. 

    ANALYSIS functions that i've done before, i have others too, just trying to catalogue them here a little: 
    sep_QD = num_dots(QD_list)
    sep_QD_crop = num_dots(QD_list_crop)
    % that are each orientation dot

    histogram of alpha values - more misaligned means 1nn 
    psi4 for 
    """  
    def __init__(self, orig_image, filename, filesize, fov):
        self.filename = filename
        self.orig_image = orig_image
        self.seg_image = None # will be segmented image once segment_dots is run, 
        	# it's nice to have this as an instance variable because it's used 
        	# for getting the ffts of dots 
        self.imsize = filesize # int (assumes square image cuz that seems 
        	# reasonable) number pixels per side
        self.fov = fov # int, nanometers
        self.QD_list = None # yes i know i don't have to do this, but it helps 
        	# me keep track of things so unless there's a reason not to i'll continue to do so. 
        self.SL_orient = None # the overall orientation of the superlattice, 
        	# picked by the user. (doesnt work well for grain boundaries)
        self.QD_list_crop = None # the cropped list, dots that have a cm more than 200 pixels from an edge
        	# (the analysis can only really be done on QDs that have a full set of neighbors)
        self.ave_psi4_KD = None # the global SL orientation that is the average 
        	# of all the psi4KD orientation
        self.ave_psi4_vor = None # same as above, but psi4Vor

    # the big one
    def segment_dots(self):
        '''big deal'''
        #pick a threshhold value for applying first order binary segmentation. 
        seg_thresh = self.get_seg_thresh()
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
    
    def get_seg_thresh(self):
        get_histo(self.orig_image,0,1,500)
        plt.pause(0.001)
        seg_thresh = float(input('Please input a threshhold value (number between 0 and 1: '))
                
        # show it overlaid 
        fig,ax=plt.subplots()
        ax.hist(np.ravel(self.orig_image),bins=np.linspace(0,1,500))
        ax.vlines(seg_thresh,0,300000,color='k')
        plt.show()
        plt.pause(.001)
        
        check = input('does this work? (y/n) ')
        if check =='y':
            return(seg_thresh)
        elif check == 'n': 
            return(self.get_seg_thresh())
        else: 
            print("you didnt choose 'y' or 'n' so we're doing this again.")
            return(self.get_seg_thresh())

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


'''
General use functions. I tried to get rid of the ones I moved inside the state class.
No guarantees that all are used/helpful, I just included all of my in progress stuff as well. 
Not at all sorted. Not even a little bit. 

plenty of these things got developed at different times and are therefore a little redundant.
i decided to keep all of them that are currently used anywhere, or seem likely to be used or helpful.
(in no particular order)
'''

def get_histo(im, minn, maxx, numbins,tag=None):
    '''
    gets a histogram of a list of datapoints (im), specify minimum value, maximum value, and number of bins
    '''
    fig,ax = plt.subplots()
    ax.hist(np.ravel(im),bins=np.linspace(minn,maxx,numbins))   
    if tag != None:     
        plt.savefig("./outputs/compilation/"+ tag + '.tif' ,dpi=200,bbox_inches='tight')
    plt.show() 
    # plt.pause(0.001)
    
def get_histo2(im):
    '''
    histogram from min to max, bins is number of items/10
    '''
    fig,ax = plt.subplots()
    ax.hist(np.ravel(im),bins=np.linspace(np.min(im),np.max(im),int(np.size(im)/10)))   
    plt.show()  

def get_histo_dists(distss):
    '''
    makes a histogram with one bin per value... usefull when doing angles or something sometimes. 
    '''
    dists = []
    for bit in distss:
        if bit != None:
            dists.append(bit)
    minn = np.floor(np.min(dists)) - .5
    maxx = np.ceil(np.max(dists)) + .5
    get_histo(dists, minn, maxx, (maxx - minn + 1)*2)

def flip(im):
    '''flips image over line y=x 
    I have this cuz my segmentation function flips the image somehow...'''
    shape = np.shape(im)
    copy = np.copy(im)
    
    for i in range (shape[0]):
        for j in range (shape[1]):
            copy[i,j] = im[j, i]
    return(copy)

def get_fft(im):
    '''get fast forier transform of an image.'''
    return np.fft.fftshift(np.fft.fft2(im))

def get_fft_2(im):
    '''get a scaled fast fourier transform of an image--to better display graphically'''
    fft = np.fft.fftshift(np.fft.fft2(im))
    return np.where(np.abs(fft)!=0,np.log(np.abs(fft)),0)

def get_ifft(fft):
    '''gets inverse of an fft'''
    return np.real(np.fft.ifft2(np.fft.ifftshift(fft)))

def show_im(im):
    '''shows an image with matplotlib'''
    fig,ax=plt.subplots()
    ax.matshow(im,cmap='gray')
    plt.show()

def show_fft(fft, peaks=None):
    '''given an fft this displays the log of that fft using matplot lib,
    has the option of also highlighting points.'''
    fig, ax = plt.subplots()
    display = np.where(np.abs(fft)!=0,np.log(np.abs(fft)),0)
    ax.matshow(display,cmap='gray')
    # ax.set_xlim([0,300])
    # ax.set_ylim([300,0])
    try:
        ax.plot(peaks[:,1],peaks[:,0],
                linestyle='None',marker='o',color='r',fillstyle='none')
        for i in range(len(peaks)):
            ax.annotate(str(i),(peaks[i,1],peaks[i,0]))  
    except (TypeError, IndexError):
        pass
    plt.show()

def get_dist(dot_list):
    '''returns a list of all the radial distances of all the bragg spots for all dots in the input list of dots.'''
    retlist = []
    for dot in dot_list: 
        points = dot.bragg_spots
        ave = 0
        count = 0
        # print("qd ind: ", dot.QD_index)
        for point in points:
            dist = np.sqrt((point[0] - 150)**2 + (point[1] - 150)**2)
            # print(dist, count)
            # ave += dist
            # count += 1
            retlist.append(dist)
        # print(ave/count)
            
    return(np.array(retlist))

def show_im_fits(im, peaks):
    '''show an image with circls overlaid on points. peaks must be numpy array'''
    fig,ax=plt.subplots()
    ax.matshow(im,cmap='gray')
    try:
        ax.plot(peaks[:,1],peaks[:,0],
            linestyle='None',marker='o',color='r',fillstyle='none')
    except IndexError:
        print('no points to overlay')
    plt.show()

def save_im_fits(im, peaks, outputpath, tag,dot): 
    '''plots an image with an overlay and saves it.'''
    plt.ioff()
    plt.close('all')
    fig,ax=plt.subplots()
    ax.matshow(im,cmap='gray')
    ax.plot(peaks[:,1],peaks[:,0],
        linestyle='None',marker='o',color='r',fillstyle='none')
    ax.axis('off')
    ax.set_ylim(0,300)
    ax.set_xlim(0,300)
    
    img_name = dot.image
    file_name = img_name + tag
    plt.savefig(outputpath + file_name + '.tif' ,dpi=200,bbox_inches='tight')
    plt.ion()
    plt.close()
    
def show_im_circs(im, peaks,tag):
    '''same as show im fits but uses patches so the dots scale as you zoom on the image. 
    Quite nice if you want to save a cmap.'''
    fig,ax=plt.subplots()
    ax.matshow(im,cmap='gray')
    y,x = np.transpose(peaks)

    for x, y in zip(x, y):
        ax.add_artist(Circle(xy = (x, y),radius= 10,fill = 'red', edgecolor = 'blue' ))

#    plt.axis('off')
#    plt.savefig(output_path+ '/' + img_name + '_' + tag + '.tif', dpi = 600,bbox = 'tight')
    plt.show()
    
def show_im_circs_dots(im, QD_list,tag):
    '''
    For all the dots in a QD list it uses patches to mark with a circle their center of mass,
    and overlays that on a given image, then saves the image at output path.
    set tag to "no_plot" to not save a plot'''
    fig,ax=plt.subplots()
    ax.matshow(im,cmap='gray')

    for dot in QD_list:
        y, x = dot.cm
        try:
            ax.add_artist(Circle(xy = (x, y),radius= 10,fill = None, edgecolor = 'blue'))
        except TypeError:
            pass
        
    if tag != 'no_plot':
        plt.axis('off')
        img_name = QD_list[0].image
        output_path = "./outputs/compilation/" + img_name + "/"
        if not exists(output_path):
            makedirs(output_path)
        plt.savefig(output_path+ '/' + img_name + '_' + tag + '.tif', dpi = 600,bbox = 'tight')

    plt.show()

def show_im_fits_two_color(im, peaks_A, peaks_B):
    '''shows image with two colors of highlighted points'''
    fig,ax=plt.subplots()
    ax.matshow(im,cmap='gray')
    ax.plot(peaks_A[:,1],peaks_A[:,0],
        linestyle='None',marker='o',color='r',fillstyle='none')
    ax.plot(peaks_B[:,1],peaks_B[:,0],
        linestyle='None',marker='o',color='b')
    plt.show()

def check_angle(theta, thetalist, buff):
    '''for an angle theta, and a list of other angles thetalist,
    this checks to see if theta is within buffer range of any of those angles.
    if so, returns true, otherwise false.'''
    for angle in thetalist:
        if angle - buff < theta and theta < angle + buff:
            return(True)
    return(False)

def check_dist_points(point1, point2, buff):
    '''returns true if two points are more than buffer pixels away from each other'''
    if np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2) < buff:
        return(False)
    return(True)

def up_right_dot(points): 
    ''' takes in a list of points (generally of an fft) and returns the rightmost point *** Fthat is above the line y = 150 
    this cuz the ffts are all 300x300 images, so (150,150) is centerpoint
    (would have to be changed if diff size fft)
    remember cs is stupid and so tuples are (y,x), and y goes down so "above" y = 150 is y < 150'''
    uplist = []
    for point in points: 
        if point[0] < 150:
            uplist.append(point)
    maxr = 0
    # big number and 0
    retpoint = [1000,0]
    for point in uplist:
        # **** there shouldnt be a case where theyd be both above and close enough for that to matter, i think its cuz of a misordering of things, i should be checking for bragg spots too close to each other before checking the angles between them. 
        if point[1] - 10 > maxr:
            maxr = point[1]
            retpoint = point
        elif point[1] + 10 > maxr:
            if point[0] > retpoint[0]:
                maxr = point[1]
                retpoint = point
    
    theta = np.arctan2(retpoint[1] - 150, retpoint[0] - 150)
    return(theta)

def bot_right_dot(points):
    '''see up_right_dot, same sort of thing cept bottom right'''
    dlist = []
    for point in points: 
        if point[0] > 150:
            dlist.append(point)
    maxr = 0
    retpoint = [0,1000]
    # **** same thing here 
    for point in dlist:
        if point[1] - 10 > maxr:
            maxr = point[1]
            retpoint = point
        elif point[1] + 10 > maxr:
            if point[0] < retpoint[0]:
                maxr = point[1]
                retpoint = point
    
    theta = np.arctan2(retpoint[1] - 150,retpoint[0] - 150)
    return(theta)


def bot_left_dot(points):
    '''see up_right_dot, same sort of thing cept bottom left'''
    dlist = []
    for point in points: 
        if point[0] > 150:
            dlist.append(point)
    maxr = 300
    retpoint = [1000,1000]
    for point in dlist:
        if point[1] + 10 < maxr:
            maxr = point[1]
            retpoint = point
        elif point[1] - 10 < maxr: 
            if point[0] < retpoint[0]:
                maxr = point[1]
                retpoint = point
    
    theta = np.arctan2(retpoint[1] - 150,retpoint[0] - 150)
    return(theta)

def bottom_dot(points):
    '''assumes all the points are from the 300x300 image, gets bottom most one'''
    dlist = []
    maxy = 0
    for point in points: 
        if point[0] > maxy:
            maxy = point[0]
            retpoint = point
    
    theta = np.arctan2(retpoint[1] - 150,retpoint[0] - 150)
    return(theta)

def findNeighborsKD(tree, point, rad,cm_list): 
    ''' using a kd tree finds all points within a specified radius of a point. '''
    #print("dot.cm is of type {}".type(dot.cm))
    #print("shape of point is {}".format(point.shape))
    point = np.expand_dims(np.array(point), axis=0)
    ind1 = tree.query_radius(point, r = rad, return_distance = True, sort_results = True)
    ind=ind1[0][0]
    #print("ind = {}".format(ind))
    dist = ind1[1][0]
    #print("dist = {}".format(dist))
    #ind = ind1[0][0][1:]
    points = cm_list[ind]
    #print(points)
    return(points)

def psi4Kneighbors(cmlist,rad):
    '''tells you the number of neighbors for each dot as defined by a kd tree and points within a certain radius
    I use it to make a colormap of number of neighbors so its easier to see if the cutoff radius is right. '''
    tree = sklearn.neighbors.KDTree(cmlist, leaf_size=6)
    retlist = []
    
    
    for point in cmlist: 
        neighbors = findNeighborsKD(tree, point, rad, cmlist)
        numn = np.shape(neighbors)[0]
        rpoint = np.append(point,numn)
        retlist.append(rpoint)
        
    return(retlist)

def get_vertices(cm_ind, cm, vor): 
    '''Part of the Voronoi implementation of psi4.
    Given the voronoi object, the cm point you care about, and its index in the original cm_list
    this gives you the local edges that surround the point.
    sometimes vertices get put thousands of pixels off the screen, so this doesnt count those edges'''
    ind = vor.point_region[cm_ind]
    retlist = []
    for i in vor.regions[ind]:
        p = vor.vertices[i]
        if abs(p[0] - cm[0]) < 300 and abs(p[1] - cm[1]) < 300:
            retlist.append(p)
    return(np.array(retlist))

def get_edges(verts):
    '''Part of the Voronoi implementation of psi4.
    for a list of vertices, this gives (vertex, vertex) pairs of each segment of the convex polygon. 
    Relies on the fact that voronoi gives you these points in a nice order going around the polygon (which it does). This would actually be a bit of a pain otherwise'''
    retlist = []
    # toss out cases on the edge w
    if np.shape(verts)[0] > 2:
        for i in range(np.shape(verts)[0]): 
                ends = (verts[i-1], verts[i])
                retlist.append(ends)
    return(np.array(retlist))

def plength(points):
    '''length of line between two points'''
    dist = np.sqrt((points[0][0]-points[1][0])**2 + (points[0][1]-points[1][1])**2)
    return(dist)

def mask(img, segmented, cm):
    ''' returns image masked so that only the dot specified is its original color, everything else black
    Currently this is straight out of the watershed segmentation, could be "softened" to be more round on the cm, 
    this is possibly something to do based on how much of the "bridge" or connection is wanting to be included, 
    as sometimes you'll see nicer definition (square grid) in those areas. 
    *** '''
    value = segmented[0][int(cm[0]),int(cm[1])]
    oneDot = img * np.where(segmented[0] == value , 1, 0)
    return(oneDot)

def mask2(img, cm):
    cp = cm
    rmin = 60
    rmax = 200 #average dot radius/max radius it'll scan over. 
    r = rmax

    val_list = []
    
    for r in range(rmin, rmax):
        ri = r-1
        ro = r+1
        ave = 0
        count = 0
        for j in range(r+1):
            y = j + cp[0] 
            for i in range(r+1):
                x = i + cp[1]
                rr = j**2 + i**2

                if rr <= ro**2 and rr > ri**2:
                    # need to add handling for dots near the edges
                    subcount = 0
                    y = int(y)
                    x = int(x)

                    try:
                        val1 = img[y,x]
                        ave += val1
                        subcount += 1 
                    except IndexError:
                        pass
                    try:
                        val2 = img[-y,x]
                        ave += val2
                        subcount += 1 
                    except IndexError:
                        pass
                    try:
                        val3 = img[y,-x]
                        ave += val3
                        subcount += 1 
                    except IndexError:
                        pass
                    try:
                        val4 = img[-y,-x]
                        ave += val4
                        subcount += 1 
                    except IndexError:
                        pass            

                    count += subcount

        ave = ave / count
        val_list.append(ave)
    
    # smooth the val_list because otherwise its oscillatory
    # (as an effect of the periodic crystal structure)
    smoothed = smooth(np.array(val_list), window_len = 20, window = 'flat') 
    rad = np.argmin(smoothed) + 20 # +30 to account for starting at 60 in range
    return(rad)

def average_diam(im):
    '''gives average diameter of masked dot image
    This is the type of thing that could be applied to mask() '''
    threshhold = im > 1/255
    labeled = skimage.measure.label(threshhold)
    region = skimage.measure.regionprops(labeled)
    return(region[0].equivalent_diameter)

def weight_spots(fft_raw, points, radius):
    '''finds the center of mass localized around all the points in points.
    used to weight the bragg spots in an fft.'''
    fft = np.abs(fft_raw)
    buf = radius
    retlist = np.copy(points)
    ind = 0
    for point in points: 
        ymin = int(np.floor(point[0] - buf))
        ymax = int(np.ceil(point[0] + buf))
        xmin = int(np.floor(point[1] - buf))
        xmax = int(np.ceil(point[1] + buf))
        
        window = fft[ymin:ymax+1,xmin:xmax+1]
        
        npoint = center_of_mass(window)
        retpoint = point[:2] + npoint - buf
        retlist[ind][0] = retpoint[0]
        retlist[ind][1] = retpoint[1]
        
        ind += 1 
    return(retlist)

def get_theta(psi_4):
    # Accepts: psi4
    # Returns: theta IN DEGREES
    theta = np.degrees(np.angle(psi_4))
    return((theta / 4)%90*-1)

''' 
Larger functions: 

these are a bit more involved and do a bit more than the previous ones.
Again I removed the redundancy of functions that are included in the class... 
I dont really have a good system yet for whats in the class and not, and its problematic 
because some of the functions in the class call on ones outside, so theres no real point
to have them in there at all. 

I tried to organize them into three main sections:
* further analysis tools
* plots and histograms
* colormaps

'''


'''

Further Analysis tools
(and things referenced in the state class)

# these are especially important to doublecheck, as many were written before I had finalized how i was doing/organizing things
'''

def sort_QD_list(QD_list):
    '''returns a list with 4 sublists. First is dots that have 0 bragg spots, then 2, then 4, then 6.
    [0] amorphous
    [1] 1n0 
    [2] 1nn
    [3] 100
    [4] 110
    [5] 1 n+1, n
    [6] 111
    [7] other/failed
    '''
    dot_null = []
    dot_1n0 = []
    dot_1nn = []
    dot_100 = []
    dot_110 = []
    dot_1n1n = []
    dot_111 = []
    dot_dif = []
    
    for dot in QD_list:
        orientation = dot.orientation[0]

        if np.all(orientation == 'amorphous'):
            dot_null.append(dot)
            
        elif np.all(orientation == '1n0'):
            dot_1n0.append(dot)
        
        elif np.all(orientation == '1nn'):
            dot_1nn.append(dot)
            
        elif np.all(orientation == '100'):
            dot_100.append(dot)
            
        elif np.all(orientation == '110'):
            dot_110.append(dot)

        elif np.all(orientation == '1,n+1,n'):
            dot_1n1n.append(dot)

        elif np.all(orientation == '111'):
            dot_110.append(dot)
            # print('holy shit a 111! QD_index: ',dot.QD_index)
        
        else: 
            print('houston we have a problem in num_dots, has abnormal .orientation[0], QDindex: ', dot.QD_index,'orientation: ',dot.orientation)
            dot_dif.append(dot)
    
    return(np.array([dot_null,dot_1n0,dot_1nn,dot_100,dot_110,dot_1n1n, dot_111,dot_dif]))
    
def all_angles(sep_QD_list):
    ''' 
    i dont think i use this so im not going to update it until i need to...

    Gives you an idea of error due to image artifacts or drift by looking at angles between 100 dots.
    returns a list of the spread of the angles
    the spread of the (100) angles from 90 (there will be two humps) shows theres error for everything done with the image
    because its a result of offset of the scan coils (or drift in the sample).
    there might be ways to correct for this but itd be hard. 
    '''
    four_dot_ret = []
    six_dot_ret = []
    for dot in sep_QD_list[2]:
        points = dot.bragg_spots
        theta1 = up_right_dot(points)/np.pi * 180 - bot_right_dot(points)/np.pi * 180
        theta2 = bot_right_dot(points)/np.pi * 180 - bot_left_dot(points)/np.pi * 180
        four_dot_ret.append(theta1)
        four_dot_ret.append(theta2)
    
    for dot in sep_QD_list[3]:
        points = dot.bragg_spots
        theta1 = np.abs(up_right_dot(points)/np.pi * 180 - bot_right_dot(points)/np.pi * 180)
        theta2 = np.abs(bot_right_dot(points)/np.pi * 180 - bottom_dot(points)/np.pi * 180)
        theta3 = np.abs(bottom_dot(points)/np.pi * 180 - bot_left_dot(points)/np.pi * 180)
        six_dot_ret.append(theta1)
        six_dot_ret.append(theta2)
        six_dot_ret.append(theta3)
    
    return[four_dot_ret,six_dot_ret]

def super_points(orig_image, windows):
    '''
    given windows from clickable_image,
    this takes those windows and finds the center of mass.
    useful for getting the orientation of the superlattice from the fft
    '''

    fft = get_fft(orig_image)
    mag = np.abs(fft)
    
    windows = np.rint(windows).astype(int)
    points = []

    for window in windows:
        windowed_fft = mag[window[0,1]:window[1,1],window[0,0]:window[1,0]]
        mask = windowed_fft > 300
        windowed_fit = windowed_fft * mask
        point = center_of_mass(windowed_fit)
        retpoint = [point[0] + window[0,1], point[1] + window[0,0]]
        points.append(retpoint)
    return(np.array(points))
            
def get_theta12_super(super_pointslst):
    '''gets the angles of the superlattice from fft from the output of super_points'''
    print('super pointslst', super_pointslst)
    top_point = [4096,0]
    right_point = [0,0]
    bottom_point = [0,0]

    for point in super_pointslst:
        if point[0] < top_point[0]:
            top_point = point
        if point[1] > right_point[1]:
            right_point = point
        if point[0] > bottom_point[0]:
            bottom_point = point
    
    print(top_point, right_point, bottom_point)

    top_theta = np.arctan2(top_point[0] - 2048, top_point[1]- 2048) / np.pi * 180 
    right_theta = np.arctan2(right_point[0]- 2048, right_point[1]- 2048) / np.pi * 180
    bottom_theta = np.arctan2(bottom_point[0]- 2048, bottom_point[1]- 2048) / np.pi * 180 
    
    print(top_theta, right_theta, bottom_theta)

    print('total angle: ', np.abs(-top_theta + bottom_theta), "(this should be 180 in theory \n(assuming you properly picked the best points), the fact that it's off shows there's error\n")

    return(top_theta, right_theta)

def get_alpha_fft(dot):
    ''' returns alpha^fft'''
    if np.all(dot.orientation[0] == '100'):
        dot_angles = dot.orientation[1]
        super_angles = dot.orientation[2]
        
        alpha1 = dot_angles[0] - super_angles[0]
        alpha2 = dot_angles[1] - super_angles[1]
        alpha = (alpha1 + alpha2)/2
        
        if alpha > 45:
            alpha = alpha - 90
        elif alpha < -45:
            alpha = alpha + 90
        return(alpha)
        
    # if two dots it takes the angle between whichever of the super dots is closest
    if np.all(dot.orientation[0] == '1n0') or np.all(dot.orientation[0] == '1nn'):
        dot_angles = dot.orientation[1]
        super_angles = dot.orientation[2]
        alpha1 = dot_angles - super_angles[0] 
        alpha2 = dot_angles - super_angles[1] 
        t1 = np.abs(alpha1)
        t2 = np.abs(alpha2)
        if min(t1,t2) == t1:
            alpha = alpha1
        else:
            alpha = alpha2
        
        if alpha > 45:
            alpha = alpha - 90
        elif alpha < -45:
            alpha = alpha + 90
        return(alpha)

def get_alpha_arg(dot,KD_vor):
    ''' gets alpha^psi4vor or kd, that is misalignment of a
    dot's AL from local SL'''
    alpha = None
    if KD_vor == 'KD':
            super_angle = dot.local_orientation_KD
    elif KD_vor =='vor':
            super_angle = dot.local_orientation_vor
    else:
        return("please input second argument as 'KD' or 'vor'.")
    if super_angle == None:
        return

    # print('super angle: ', super_angle)

    # if four dots then takes whichever has closest angle
    if np.all(dot.orientation[0] == '100'):
        dot_angles = dot.orientation[1]   
        alpha1 = (dot_angles[0] - super_angle) % 90
        alpha2 = (dot_angles[1] - super_angle ) % 90
        
        if alpha1 < -45:
            alpha1 = 90 + alpha1
        elif alpha1 > 45:
            alpha1 = alpha1 - 90
            
        if alpha2 < -45:
            alpha2 = 90 + alpha2
        elif alpha2 > 45:
            alpha2 = alpha2 - 90
        
        alpha = (alpha1 + alpha2) / 2
         
    # if two dots it takes the angle between whichever of the super dots is closest
    elif np.all(dot.orientation[0] == '1n0') or np.all(dot.orientation[0] == '1nn'):
        dot_angle = dot.orientation[1][0] 
        alpha1 = dot_angle - super_angle

        if alpha1 > 45: 
            alpha = alpha1 - 90
        elif alpha1 < -45: 
            alpha = alpha1 + 90
        else: 
            alpha = alpha1
    return(alpha)

def get_alpha_spec_angle(dot,super_angle):
    '''
    gets alpha between a dots atomic lattice and generic specified other angle.
    '''
    # if four dots then averages 
    if np.all(dot.orientation[0] == '100'):
        dot_angles = dot.orientation[1]   
        alpha1 = (dot_angles[0] - super_angle) % 90
        alpha2 = (dot_angles[1] - super_angle ) % 90
        
        if alpha1 < -45:
            alpha1 = 90 + alpha1
        elif alpha1 > 45:
            alpha1 = alpha1 - 90
            
        if alpha2 < -45:
            alpha2 = 90 + alpha2
        elif alpha2 > 45:
            alpha2 = alpha2 - 90
        
        alpha = (alpha1 + alpha2) / 2
        return(alpha)
        
    # if two dots it takes the angle between whichever of the super dots is closest
    elif np.all(dot.orientation[0] == '1n0') or np.all(dot.orientation[0] == '1nn'):
        dot_angle = dot.orientation[1][0] % 90
        alpha1 = dot_angle - super_angle
        alpha2 = 180 - alpha1
        t1 = np.abs(alpha1)
        t2 = np.abs(alpha2)
        
        if min(t1,t2) == t1:
            alpha3 = alpha1
        else:
            alpha3 = alpha2
        
        if alpha3 < -45:
            alpha = 90 + alpha3
        elif alpha3 > 45:
            alpha = alpha3 - 90
        else:
            alpha = alpha3

        return(alpha)

def beta_betafft(dot, KD_vor):
    '''returns the angle between betaKD and beta fft'''
    if KD_vor == "KD":
        beta = dot.local_orientation_KD
    elif KD_vor == 'vor':
        beta = dot.local_orientation_vor
    else:
        return('"KD" or "vor" please.')
    
    if beta != None:
        super_angles = dot.orientation[2]
        alpha1 = beta - super_angles[0] 
        alpha2 = beta - super_angles[1] 
        t1 = np.abs(alpha1)
        t2 = np.abs(alpha2)
        if min(t1,t2) == t1:
            alpha = alpha1
        else:
            alpha = alpha2

        return(alpha)
    
def def_neighbors(QD_list,rad =184):
    '''you can use this if you want to change the radius within which two 
    dots are considered "neighbors". '''
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

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError
        print("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError
        print("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError
        print("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def hann2(im, diam, cp):
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



'''

Plots and Histograms

'''
    
def plot_alpha_vs_mpsi(QD_list,kd_vor,abst,QD_list2 = None):
    '''alpha^fft/psi4/neighbors vs alpha^psi4vor.
    
    options for kd_vor: 
        'KD/vor': |psi4KD|/|psi4vor| vs alpha^fft 
        'argKD': |psi4KD| vs alpha^KD 
        'argvor':|psi4vor| vs alpha^vor 
        'alpha': |psi4vor| vs alpha^<alpha> aaa

    abst:
        'abs': absolute value of alpha (probably what you want)
        anything else: not absolute value

    QD_list2:
        you can do two different QD_lists (eg. 1n0 vs 100 from sort_QD_list()) and they'll plot in different colors
        '''
    alphalist = []
    psi4list = []            
    alphalist2 = []
    psi4list2 = []
    alphalistloc = []
    psi4listloc = []
    fitlist = []
    
    if kd_vor not in ['KD','vor','argKD','argvor','alpha','alpha_alpha']:
        return('Please input "KD" or "vor" (or "argKD" or "argvor"), or "alpha" (for ave misalignments from neighbors),"alpha_alpha", to define which psi4 value you want.')
    
    for dot in QD_list:
        if kd_vor == 'KD' or kd_vor == 'vor' or kd_vor == 'super':
            alpha = get_alpha_fft(dot)
        elif kd_vor in['argKD']:
            alpha = get_alpha_arg(dot,'KD')
        elif kd_vor in ['argvor','alpha_alpha']:
            alpha = get_alpha_arg(dot,'vor')
        elif kd_vor == 'alpha':
            alpha = dot.ave_neighbor_alpha
        else:
            print('lol wut. u fuqed sometin up m8 in vs_mpsi()')
            
        if abst == 'abs':
            if isinstance(alpha,list):
                alpha = abs(alpha)
            
        if kd_vor in ['KD','argKD']:
            psi4 = dot.psi4KD
        elif kd_vor in ['vor','argvor','alpha']:
            psi4 = dot.psi4vor
        elif kd_vor == 'alpha_alpha':
            psi4 = dot.ave_neighbor_alpha

        if type(alpha).__module__ == np.__name__ and psi4 != None:
            alphalist.append(alpha)
            if kd_vor != 'alpha_alpha':
                psi4 = np.abs(psi4)
            psi4list.append(psi4)
            if np.abs(psi4) < 10 and np.abs(alpha) < 16: #I looked at them by hand. The issue is that my algorithm
            # that picks orientations isn't perfect. Maybe 4 dots in an image will be incorrectly 1nn/1n0, which leads 
            # to them being wildly misaligned from the SL (45 degrees or so), and their neighbors. The problem is that 
            # this pulls the average alpha misalignment of their neighbors up to ~10-15 degrees, when it really shouldnt be. 
            # Meanwhile, the ~15 degree misalignments from psi4 are all real. 
                alphalistloc.append(alpha)
                psi4listloc.append(psi4)

    if QD_list2 != None:
        for dot in QD_list2:
            if kd_vor == 'KD' or kd_vor == 'vor' or kd_vor == 'super':
                alpha = get_alpha_fft(dot)
            elif kd_vor in['argKD']:
                alpha = get_alpha_arg(dot,'KD')
            elif kd_vor in ['argvor','alpha_alpha']:
                alpha = get_alpha_arg(dot,'vor')
            elif kd_vor == 'alpha':
                alpha = dot.ave_neighbor_alpha
            else:
                print('lol wut. u fuqed sometin up m8 in vs_mpsi()rd2')

            if abst == 'abs':
                if isinstance(alpha,list):
                    alpha = abs(alpha)

            if kd_vor in ['KD','argKD']:
                psi4 = dot.psi4KD
            elif kd_vor in ['vor','argvor','alpha']:
                psi4 = dot.psi4vor
            elif kd_vor == 'alpha_alpha':
                psi4 = dot.ave_neighbor_alpha

            if alpha != None and psi4 != None:
                alphalist2.append(alpha)
                if kd_vor != 'alpha_alpha':
                    psi4 = np.abs(psi4)
                psi4list2.append(psi4)
                if np.abs(psi4) < 10 and np.abs(alpha) < 16:
                    alphalistloc.append(alpha)
                    psi4listloc.append(psi4)

    if kd_vor == 'alpha_alpha':
        fit = np.polyfit(alphalistloc, psi4listloc, 1)
        ### Yes i realize the labels are switched, look above, they're switched cuz 
        print('std alphalist',np.std(psi4listloc),'\nstd psi4list',np.std(alphalistloc))
        for pair in zip(alphalistloc,psi4listloc):
            fitlist.append(pair)
    fig, ax = plt.subplots()    
    plt.subplot(111)
   
    size = 100
    plt.scatter(alphalistloc, psi4listloc, c='red', marker='o', s = size, alpha = 0.2)
    plt.scatter(alphalist2, psi4list2, c='blue', marker='o', s = size, alpha = 0.2)
    if kd_vor == 'alpha_alpha':
        # plt.plot([-30,30],[p(-10),p(10)],color = 'black',linestyle = '-',linewidth=2)
        # plt.scatter(alphalistloc, psi4listloc, c='green', marker='o', s = 100, alpha = 0.2)
        pass 
    if kd_vor in ['KD','argKD']:
        plt.ylabel(r'$|\psi_{4\mathrm{KD}}|$')
    elif kd_vor =='alpha_alpha':
        plt.ylabel(r'$\alpha^{<\alpha>}$')
    else:
        plt.ylabel(r'$|\psi_{4 \mathrm{Voronoi}}|$')
        
    if kd_vor == 'argKD':    
        plt.xlabel(r'$\alpha^{KD}$')
    elif kd_vor == 'argvor':
        plt.xlabel(r'$\alpha^{Vor}$')
    elif kd_vor in ['KD','vor','super']:
        plt.xlabel(r'$\alpha^{\mathrm{fft}}$')
    elif kd_vor == 'alpha':
        plt.xlabel(r'$\alpha^{<\alpha>}$')
    elif kd_vor =='alpha_alpha':
        plt.xlabel(r'$\alpha^{\psi_{4vor}}$')

    plt.xlim(-15,15)
    plt.ylim(-15,15)
    ###
    plt.savefig("./../../"+ 'alpha^alpha_vs_alpha^psi4' + '.tif' ,dpi=200,bbox_inches='tight')
    ###
    plt.show()
    if kd_vor == 'alpha_alpha':
        return(np.array(fitlist))

def alpha_histo(sep_QD_crop,graycolor,vor_kd='vor'):
    '''Makes a histogram of the alpha^Vor/KD values, can be color coded for orientation
    Accepts the sorted QD_list. 
    2nd argument 'gray' or 'color' '''
    alpha_vor_list_1n0 = []
    alpha_vor_list_1nn = []
    alpha_vor_list_100 = []
    count = 0
    maxa = 0
    for dot in sep_QD_crop[1]:
        if vor_kd == 'vor': 
            alpha = get_alpha_arg(dot,'vor')
        else:
            alpha = get_alpha_arg(dot,'KD')
        if alpha != None:
            alpha_vor_list_1n0.append(np.abs(alpha))
            if alpha > maxa:
                maxa = alpha
                if alpha > 20:
                    print(alpha,dot.QD_index,dot.orientation,'psi4 vor orie: ',dot.local_orientation_vor,' cm: ',dot.cm)
            
    for dot in sep_QD_crop[2]:
        if vor_kd == 'vor': 
            alpha = get_alpha_arg(dot,'vor')
        else:
            alpha = get_alpha_arg(dot,'KD')
        if alpha != None:
            alpha_vor_list_1nn.append(np.abs(alpha))
            if alpha > maxa:
                maxa = alpha

    for dot in sep_QD_crop[3]:
        if vor_kd == 'vor': 
            alpha = get_alpha_arg(dot,'vor')
        else:
            alpha = get_alpha_arg(dot,'KD')
        if alpha != None:
            alpha_vor_list_100.append(np.abs(alpha))
            if alpha > maxa:
                maxa = alpha

    alpha_vor_list = [alpha_vor_list_100,alpha_vor_list_1n0,alpha_vor_list_1nn]
    fig,ax = plt.subplots()
    if vor_kd == 'vor':
        plt.xlabel(r'$|\alpha^{\psi_{4vor}}|$')
    else:
        plt.xlabel(r'$|\alpha^{\psi_{4kd}}|$')
    plt.ylabel('counts')
    #saving
    # img_name = sep_QD_crop[0][0].image
    # output_path = "./outputs/compilation/" + img_name + "/"
    # if not exists(output_path):
    #     makedirs(output_path)
    if graycolor == 'gray' or graycolor == 'grey':
        plt.hist(alpha_vor_list_100 + alpha_vor_list_1n0 + alpha_vor_list_1nn, 2* int(maxa),color = 'grey')
        # if vor_kd == 'vor':
        #     plt.savefig(output_path + img_name + '_alpha_vor_histo_grey' + '.tif' ,dpi=200,bbox_inches='tight')
        # else:
        #     plt.savefig(output_path + img_name + '_alpha_KD_histo_grey' + '.tif' ,dpi=200,bbox_inches='tight')

    elif graycolor == 'color':
        plt.hist(alpha_vor_list, 2* int(maxa), stacked=False, color = ['red','green','blue'],label = ['(100)','(1n0)','(1nn)'])
        ax.legend(prop={'size': 10})
        # if vor_kd == 'vor':
        #     plt.savefig(output_path + img_name + '_alpha_vor_histo_color' + '.tif' ,dpi=200,bbox_inches='tight')
        # else:
        #     plt.savefig(output_path + img_name + '_alpha_KD_histo_color' + '.tif' ,dpi=200,bbox_inches='tight')
    else:
        return('graycolor as either "gray" or "color".')
    plt.show()

def dif_plot(state, QD_list,bKD_bVOR,QD_list2 = None):
    '''
    Plots magnitude of Psi4(vor/KD) vs beta(vor/KD) - (<beta(vor/KD)> OR beta_FFT)
    not particularly useful as i havent seen any correlation.
    '''
    # alphalist is betas... 
    alphalist = []
    psi4list = []
    
    if bKD_bVOR == 'bKD':
        ave_L = state.ave_psi4_KD
        tmp = "KD"
    elif bKD_bVOR == 'bVOR':
        ave_L = state.ave_psi4_vor
        tmp = "vor"
    elif bKD_bVOR == 'FFT':
        tmp = str(input('Are we looking at magnitude of psi4 voronoi ("vor") or KD ("KD")?'))
    else:
        return('Please input "bKD","bVOR" or "FFT" to specify what you are comparing the dots orientation to.')
    

    for dot in QD_list:
        
        if bKD_bVOR == 'bKD':
            alpha = dot.local_orientation_KD
        elif bKD_bVOR == 'bVOR':
            alpha = dot.local_orientation_vor
        
        if tmp == 'vor':
            psi4  = dot.psi4vor
        elif tmp == 'KD':
            psi4 = dot.psi4KD
        else:
            return('input must be either "vor" or "KD"')

        if bKD_bVOR == 'FFT':
            dif = beta_betafft(dot,tmp)
            alpha = dot.orientation[1]
            if np.size(alpha) > 1: 
                alpha = alpha[0]    
            elif np.size(alpha) == 0:
                alpha = None

            if dif != None and psi4 != None:
                alphalist.append(dif)
                psi4list.append(np.abs(psi4))
        
        else:    
            if alpha != None:
                dif = alpha - ave_L
                if dif > 45:
                    dif = dif - 90
                if dif < -45:
                    dif = 90 + dif
                alphalist.append(dif)
                psi4list.append(np.abs(psi4))

    if QD_list2 != None:
        alphalist2 = []
        psi4list2 = []

        for dot in QD_list2:
            
            if bKD_bVOR == 'bKD':
                alpha = dot.local_orientation_KD
                psi4 = dot.psi4KD
            elif bKD_bVOR == 'bVOR':
                alpha = dot.local_orientation_vor
                psi4 = dot.psi4vor
            
            if tmp == 'vor':
                psi4  = dot.psi4vor
            elif tmp == 'KD':
                psi4 = dot.psi4KD
            
             ### this plots vs beta-betafft instead, dont forget to do this on QD_list1 too
            if bKD_bVOR == 'FFT':
                dif = beta_betafft(dot,tmp)
                alpha = dot.orientation[1]
                if np.size(alpha) > 1: 
                    alpha = alpha[0]    
                elif np.size(alpha) == 0:
                    alpha = None
                if dif != None and psi4 != None:
                    alphalist2.append(dif)
                    psi4list2.append(np.abs(psi4))
            
            else:
                if alpha != None:
                    dif = alpha - ave_L
                    if dif > 45:
                        dif = dif - 90
                    if dif < -45:
                        dif = 90 + dif
                    alphalist2.append(dif)
                    psi4list2.append(np.abs(psi4))            
            
    fig,ax = plt.subplots()
    plt.subplot(111)
    plt.scatter(alphalist, psi4list, c='red', marker='o', s = 100, alpha = 0.2)
    if QD_list2 != None:
        plt.scatter(alphalist2, psi4list2, c='blue', marker='o', s = 100, alpha = 0.2)
    if bKD_bVOR == "bKD":
        plt.ylabel(r'$|\psi_{4}^{KD}|$')
        plt.xlabel(r'$\beta^{KD} - <\beta^{KD}>$')
    elif bKD_bVOR == 'bVOR':
        plt.ylabel(r'$|\psi_{4}^{VOR}|$')
        plt.xlabel(r'$\beta^{VOR} - <\beta^{VOR}>$')
    elif bKD_bVOR == 'FFT':
        if tmp == 'vor':
            plt.ylabel(r'$|\psi_{4i}^{VOR}|$')
            plt.xlabel(r'$\beta^{VOR} - <\beta^{fft}>$')
        if tmp == 'KD':
            plt.ylabel(r'$|\psi_{4}^{KD}|$')
            plt.xlabel(r'$\beta^{KD} - <\beta^{fft}>$')
    plt.show()

def beta_vs_alpha(state, QD_list,bKD_bVOR,QD_list2 = None):
    ''' 
    This is an interesting one. Plots alpha^<vor/KD> vs beta(vor/KD) - <beta(vor/KD)>. 
    Shows that a misaligned local superlattice will pull the atomic lattice, but (oddly) not by much.
    Slopes often around .2 or .3, so for every 10 degrees off the local superlattice is from the average,
    it only pulls the atomic lattice off by 2 or 3 degrees.

    width of distributions
    '''
    
    alphalist = []
    betalist = []
    alphalist2 = []
    betalist2 = []
    # alphalistloc = []
    # betalistloc = []

    if bKD_bVOR == 'KD':
        ave_betaKDVOR = state.ave_psi4_KD
    elif bKD_bVOR == 'vor':
        ave_betaKDVOR = state.ave_psi4_vor
    else:
        return('Second argument should be "KD" for plotting alpha<KD> vs betaKD - <betaKD>, or "vor" for alpha<vor> vs betaVor - <betaVor>')

    for dot in QD_list:
        if bKD_bVOR == 'KD':
            beta = dot.local_orientation_KD
            alpha = get_alpha_spec_angle(dot,ave_betaKDVOR)

        elif bKD_bVOR == 'vor':
            beta = dot.local_orientation_vor
            alpha = get_alpha_spec_angle(dot,ave_betaKDVOR)

        if alpha != None and beta != None:
            alphalist.append(alpha)
            betalist.append(beta - ave_betaKDVOR)
            # if np.abs(alpha) < 25 and np.abs(beta-ave_betaKDVOR) < 25:
            #     alphalistloc.append(alpha)
            #     betalistloc.append(beta - ave_betaKDVOR)
    
    if QD_list2 != None:
        for dot in QD_list2:
            if bKD_bVOR == 'KD':
                beta = dot.local_orientation_KD
                alpha = get_alpha_spec_angle(dot,ave_betaKDVOR)

            elif bKD_bVOR == 'vor':
                beta = dot.local_orientation_vor
                alpha = get_alpha_spec_angle(dot,ave_betaKDVOR)

            if alpha != None and beta != None:
                alphalist2.append(alpha)
                betalist2.append(beta - ave_betaKDVOR)
                # if np.abs(alpha) < 25 and np.abs(beta-ave_betaKDVOR) < 25:
                #     alphalistloc.append(alpha)
                #     betalistloc.append(beta - ave_betaKDVOR)
    
    # fit = np.polyfit(betalist+betalist2, alphalist+alphalist2, 1)
    # p = np.poly1d(fit)
    fig,ax = plt.subplots()
    plt.subplot(111)
    # plt.plot([-15,15],[p(-15),p(15)],color = 'blue',linestyle = '-',linewidth=2)
    plt.scatter(betalist, alphalist, c='red', marker='o', s = 100, alpha = 0.2)
    plt.scatter(betalist2, alphalist2, c='blue', marker='o', s = 100, alpha = 0.2)
    
    ### for paper
    # blue_line = matplotlib.lines.Line2D([], [], color='blue',markersize=4, label=r'$\alpha^{<vor>} = 0.24 (\beta^{vor}-<\beta^{vor}>) +\, 1.2$',linewidth = 3)
    # plt.legend(handles=[blue_line],prop={'size':18})
    # matplotlib.rcParams.update({'font.size': 18})
    
    # plt.scatter(betalistloc, alphalistloc, c='green', marker='o', s = 100, alpha = 0.2)

    ## for making paper stuff ##
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20)
    matplotlib.rcParams.update({'font.size': 18})
    
    if bKD_bVOR == "KD":
        plt.ylabel(r'$\alpha^{<kd>}$')
        plt.xlabel(r'$\beta^{kd} - <\beta^{kd}>$')
    else:
        plt.ylabel(r'$\alpha^{<vor>}$')
        plt.xlabel(r'$\beta^{\mathrm{vor}} - <\beta^{\mathrm{vor}}>$')
        # plt.savefig("./outputs/compilation/presentation/"+ 'alpha_vs_beta-<beta>' + '.tif' ,dpi=200,bbox_inches='tight')
    plt.show()
    # print(p)
    # return(p)

def plot_psi4_alpha_ave(state, QD_list,bKD_bVOR,QD_list2 = None):
    '''
    Plots |psi4| vs alpha^<kd/vor>. Not significant it seems. 
    '''
      
    if bKD_bVOR == 'bKD':
        ave_L = state.ave_psi4_KD
        tmp = "KD"
    elif bKD_bVOR == 'bVOR':
        ave_L = state.ave_psi4_vor
        tmp = "vor"
    else:
        return('Please input "bKD" or "bVOR" to specify which local orientation you want compared.')
        
    alphalist = []
    psi4list = []    

    for dot in QD_list:
        alpha = get_alpha_spec_angle(dot,ave_L)

        if bKD_bVOR == 'bKD':
            psi4 = dot.psi4KD
        elif bKD_bVOR == 'bVOR':
            psi4 = dot.psi4vor
        
        if alpha != None and psi4 != None:
            alphalist.append(alpha)
            psi4list.append(np.abs(psi4))
  
    if QD_list2 != None:
        alphalist2 = []
        psi4list2 = []

        for dot in QD_list2:   
            alpha = get_alpha_spec_angle(dot,ave_L)
            
            if bKD_bVOR == 'bKD':
                psi4 = dot.psi4KD
            elif bKD_bVOR == 'bVOR':
                psi4 = dot.psi4vor

            if alpha != None and psi4 != None:
                alphalist2.append(alpha)
                psi4list2.append(np.abs(psi4))

    fig,ax = plt.subplots()
    plt.subplot(111)
    plt.scatter(alphalist, psi4list, c='red', marker='o', s = 100, alpha = 0.2)
    if QD_list2 != None:
        plt.scatter(alphalist2, psi4list2, c='blue', marker='o', s = 100, alpha = 0.2)
    if bKD_bVOR == "bKD":
        plt.ylabel(r'$|\psi_{4}^{<KD>}|$')
        plt.xlabel(r'$\alpha^{KD}$')
    elif bKD_bVOR == 'bVOR':
        plt.ylabel(r'$|\psi_{4}^{VOR}|$')
        plt.xlabel(r'$\alpha^{<VOR>}$')
    plt.show()


'''

colormaps and overlays onto the original image

'''
def show_strain(QD_list, im):
    '''
    show_psi4(QD_list,kd_vor, im, tag)
    Graphical Representation of the magnitude psi-4 on a colormap. 
    '''
    # gets magnitude of psi4 list, in list of x, list of y, list of psi4
    N = np.shape(QD_list)[0]
    x = np.zeros(N)
    y = np.zeros(N)
    strainlist = np.zeros(N)
    i = 0

    for dot in QD_list:
        if dot.strain is not None: 
            x[i] = dot.cm[1]
            y[i] = dot.cm[0]
            strainlist[i] = dot.strain
            i += 1  

    x = [] 
    y = []
    strainlist = []
    for dot in QD_list:
        if dot.strain is not None: 
            x.append(dot.cm[1])
            y.append(dot.cm[0])
            strainlist.append(dot.strain)

    x = np.array(x)
    y = np.array(y)
    strainlist = np.array(strainlist)

    patches = []
    for x1,y1,r in zip(x, y, strainlist):
        circle = Circle((x1,y1), 30)
        patches.append(circle)

    points = []
    for dot in QD_list:
        cm = dot.cm
        points.append([cm[1],cm[0]])

    vor = Voronoi(points)
    fig = voronoi_plot_2d(vor, line_colors = 'green', show_points = False, show_vertices = False)
    ax = fig.add_subplot(111)

    p = PatchCollection(patches, cmap=matplotlib.cm.inferno, alpha=0.9)
    p.set_array(strainlist)
    p.set_clim(96000, 160000)
    cbar = fig.colorbar(p, extend = 'both')
    # cbar.cmap.set_under('green', alpha = 0)
    # cbar.cmap.set_over('white')
    ax.add_collection(p)
    ax.invert_yaxis()
    plt.imshow(im, cmap = 'gray')

    plt.show()


def show_psi4(QD_list, im, kd_vor = 'vor', tag = 'no_save', show = 'show'):
    '''
    show_psi4(QD_list,kd_vor, im, tag)
    Graphical Representation of the magnitude psi-4 on a colormap. 
    '''
    # gets magnitude of psi4 list, in list of x, list of y, list of psi4
    N = np.shape(QD_list)[0]
    x = np.zeros(N)
    y = np.zeros(N)
    psi4list = np.zeros(N)
    i = 0
    if kd_vor == 'KD':
        for dot in QD_list:
            try:
                psi4 = np.abs(dot.psi4KD)
                x[i] = dot.cm[1]
                y[i] = dot.cm[0]
                psi4list[i] = psi4
                i += 1 
            except TypeError:
                pass
    elif kd_vor == 'vor':
            for dot in QD_list:
                try:
                    psi4 = np.abs(dot.psi4vor)
                    x[i] = dot.cm[1]
                    y[i] = dot.cm[0]
                    psi4list[i] = psi4
                    i += 1 
                except TypeError:
                    pass
    
    elif kd_vor == 'strain':
            for dot in QD_list:
                x[i] = dot.cm[1]
                y[i] = dot.cm[0]
                psi4list[i] = dot.strain
                i += 1

    else:
        return('Please input "KD" or "vor" to define which psi4 value you want, (or strain).')
    # turns off interactivity of matplotlib
    if show != 'show':
        plt.ioff()

    patches = []
    for x1,y1,r in zip(x, y, psi4list):
        circle = Circle((x1,y1), 30)
        patches.append(circle)

    if kd_vor == 'strain': 
        points = []
        for dot in QD_list:
            cm = dot.cm
            points.append([cm[1],cm[0]])
        # plt.gca()
        vor = Voronoi(points)
        fig = voronoi_plot_2d(vor, line_colors = 'green', show_points = False, show_vertices = False)
        ax = fig.add_subplot(111)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    p = PatchCollection(patches, cmap=matplotlib.cm.inferno, alpha=0.9)
    p.set_array(psi4list)
    ax.add_collection(p)
    ax.invert_yaxis()
    plt.colorbar(p)
    plt.imshow(im, cmap = 'gray')

    plt.show()


def psi4_overlay(QD_list, orig_image,kd_vor = 'vor', save = False, scale = False, lwidth = 1, show_vor = True):
    '''overlays psi4 vectors on the image.
    neat cuz it shows that psi4 does in fact point in local SL orientation.
    the code here is a bit of a shit-show
    
    *args: 
    'scale' = True if you want the arrows to scale based on the magnitude of psi4 for the dot
    'save' = True/False if you want it to save the image to desktop. 

    '''
    Xkd = []
    Ykd = []
    Ukd = []
    Vkd = []
    magls = []
    plt.close('all')
    
    for dot in QD_list:
        if kd_vor == 'KD':
            thetar = dot.psi4KD
        elif kd_vor == 'vor':
            thetar = dot.psi4vor
        else:
            return('input "KD" or "vor" to specify.')
        
        if thetar != None:
            theta = np.radians(get_theta(thetar))
            Ykd.append(dot.cm[0])
            Xkd.append(dot.cm[1])
            Vkd.append(np.cos(theta))
            Ukd.append(np.sin(theta))
            magls.append(np.abs(thetar))
    
    # matplotlib.patches.ArrowStyle("Wedge", tail_width = 10, shrink_factor=0.5)

    patches = []
    for x1,y1,dy,dx,mag in zip(Xkd, Ykd, Ukd, Vkd, magls):
        if scale == True:  
            ## scale triangle with magnitude
            pld_scale = mag * 150
        else: 
            ## constant size
            pld_scale = 115
        xy = np.array([[x1,y1],[x1,y1],[x1,y1]])+(np.array([[dx,dy],[-dy/4,dx/4],[dy/4,-dx/4]])-np.array([[dx/2.,dy/2],[dx/2.,dy/2],[dx/2.,dy/2]]))*pld_scale

        triangle=matplotlib.patches.Polygon(xy, closed=True, edgecolor = None, linewidth = 0)
        patches.append(triangle)

    if kd_vor == 'vor':
        points = []
        for dot in QD_list:
            cm = dot.cm
            points.append([cm[1],cm[0]])
        # plt.gca()
        vor = Voronoi(points)
        if show_vor == True: 
            fig = voronoi_plot_2d(vor, line_colors = 'green', show_points = False, show_vertices = False, zorder = 0, line_width = lwidth)
            ax = fig.add_subplot(111)
        else: 
            fig = plt.figure()
            ax = fig.add_subplot(111) 
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)


    p = PatchCollection(patches, cmap=matplotlib.cm.inferno, alpha=0.9, zorder = 2)
    p.set_array(np.array(magls))
    # ax.add_collection(p)
    ax.invert_yaxis()
    # plt.colorbar(p)
    plt.imshow(orig_image, cmap = 'gray')
    # plt.imshow(a)

    plt.show()

    if save == True:
        img_name = QD_list[0].image
        print('saving')
        output_path = "../presentables/" 
        file_name = img_name + '_psi4_overlay'
        plt.savefig(output_path + file_name + '.png' ,dpi=1000,bbox_inches='tight')


def show_neighbors_rad(QD_list, rad, im):
    '''
    default rad is 184
    psi4Kneighbors to check the number of neighbors of each point using a KD tree and then represents that graphically
    this is helpful for deciding the cutoff radius when using the KD tree method of determining neighbors
    variable names are wonky cuz i copied and pasted this mostly from older code. you can change em if ya want.
    '''
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    cmlist = np.array([dot.cm for dot in QD_list])
    neigh_check = psi4Kneighbors(cmlist,rad)
    N4_36abs = np.transpose(neigh_check)

    xvaln = N4_36abs[1]
    yvaln = N4_36abs[0]
    nsival = N4_36abs[2]
    
    fig,ax = plt.subplots()
    markers = plt.scatter(xvaln, yvaln, c=nsival, cmap = plt.cm.jet, marker='o',s=15)
    axes = plt.gca()
    axes.set_xlim([0,4096])
    axes.set_ylim([0,4096])
    cbar = fig.colorbar(markers)
    # plt.colorbar()
    cbar.set_label('number of neighbors')
    plt.imshow(im, cmap = 'gray')
    axes.axis('off')
    plt.show()
    # img_name and output_path is defined when you load the image
    # img_name = QD_list[0].image
    # output_path = "./outputs/compilation/" + img_name + "/"
    # if not exists(output_path):
    #     makedirs(output_path)
    # file_name = img_name + 'radius_' + str(rad) + 'neighbors_cmap'
    # plt.savefig(output_path +'/'+ file_name + '.tif' ,dpi=1000,bbox_inches='tight')

def alpha_vector_overlay(QD_list,orig_image, save=False, others = False, alphas = True):
    '''Atomic Lattice vector overlay
    *args : 
    alphas = (True): False if you dont want the arrows shown
    save = (False): True if you want it to save the image to output path (currently my figures folder)
    others = (False): True if you want it to also overlay the locations of 110 (blue) 1n1n (green) and no orientation (pink) dots 
    '''
    Xkd = []
    Ykd = []
    Ukd = []
    Vkd = []
    Xo = []
    Yo = []
    Xo2 = []
    Yo2 = []
    Xo3, Yo3 = [], []
    Xoa, Yoa = [], []

    for dot in QD_list:
        theta = None
        theta2 = None
        # print(dot.psi4KD,dot.psi4vor)
        
        if np.all(dot.orientation[0] == '100'):
            theta = dot.orientation[1][0]
            theta2 = dot.orientation[1][1]
            theta = np.radians(theta)
            theta2 = np.radians(theta2)
        
        if np.all(dot.orientation[0] == '1n0') or np.all(dot.orientation[0] == '1nn'):# or np.all(dot.orientation[0] == '1,n+1,n'):
            theta = np.radians(dot.orientation[1][0])
        
        if np.all(dot.orientation[0] == '110'):
            Yo.append(dot.cm[0])
            Xo.append(dot.cm[1])
        if np.all(dot.orientation[0] == '111'):
            Yo2.append(dot.cm[0])
            Xo2.append(dot.cm[1])

        if np.all(dot.orientation[0] == '1,n+1,n'):
            Yo3.append(dot.cm[0])
            Xo3.append(dot.cm[1])

        if np.all(dot.orientation[0] == 'amorphous'):
            Yoa.append(dot.cm[0])
            Xoa.append(dot.cm[1])

        if type(theta).__module__ == np.__name__:
            Ykd.append(dot.cm[0])
            Xkd.append(dot.cm[1])
            Vkd.append(np.cos(theta))
            Ukd.append(np.sin(theta))
        
        if type(theta2).__module__ == np.__name__:
            Ykd.append(dot.cm[0])
            Xkd.append(dot.cm[1])
            Vkd.append(np.cos(theta2))
            Ukd.append(np.sin(theta2))

    fig,ax=plt.subplots() 
    
    patches_dir = []
    scale = 85
    for x,y, dx,dy in zip(Xkd, Ykd, Ukd, Vkd):
        dx = dx * scale
        dy = -1 * dy * scale
        arrow = matplotlib.patches.Arrow(x - dx/2, y - dy/2, dx, dy, width = 30, alpha = 0.6)
        patches_dir.append(arrow)

    patches_110 = []
    radius = 40
    for x, y in zip(Xo, Yo):
        circle = matplotlib.patches.Circle((x,y), radius)
        patches_110.append(circle)

    patches_1n1n = []
    for x, y in zip(Xo3, Yo3):
        circle = matplotlib.patches.Circle((x,y), radius)
        patches_1n1n.append(circle)

    patches_amor = []
    for x, y in zip(Xoa, Yoa):
        circle = matplotlib.patches.Circle((x,y), radius)
        patches_amor.append(circle)

    # plt.quiver(Xkd, Ykd, Ukd, Vkd, color = 'red', units = 'inches')
    # plt.scatter(Xo,Yo,color = 'pink',s=20) # 110s pink
    plt.scatter(Xo2,Yo2,color = 'blue',s=20) # 111s blue
    plt.imshow(orig_image,cmap='gray')
    plt.title(r'$\alpha$' + ' vector overlay')
    
    p1 = PatchCollection(patches_dir, color = 'red', edgecolor = 'r')
    p2 = PatchCollection(patches_110, color = 'green')
    p3 = PatchCollection(patches_1n1n, color = 'green')
    p4 = PatchCollection(patches_amor, color = 'pink')
    if alphas == True: 
        ax.add_collection(p1)
    if others == True: 
        ax.add_collection(p2)
        ax.add_collection(p3)
        ax.add_collection(p4)

    if save == True:
        img_name = QD_list[0].image
        output_path = "../presentables/QD_manuscript/" #+ img_name 
        plt.axis('off')
        file_name = img_name + '_alpha_vector_overlay'
        plt.savefig(output_path + file_name + '.png' ,dpi = 1000,bbox_inches='tight')
    plt.show()

def show_alpha(QD_list, orig_image, kd_vor_super,tag,tri,save,lowthresh,hithresh):
    """This makes a colormap of the alpha^kd,vor,fft values of each dot and overlays it on the original image.
    In terms of running it:
    put hithresh greater than 100 to enable autscaling of colorbar
    (QD_list,'alphaKD or alphaVOR or super' depending on which reference you want to compare,
    'tag to save as','tri' or 'abs' for if you want absolute valued or tricolored cmap, 
    'save' to save, anything else to not, 'minimum value angle kept', 'max angle kept' if abs only this one matters)"""
    alphalist = []
    
    if tri == 'abs':
        abst = True
    elif tri == 'tri':
        abst = False
    else:
        return('tri should be "abs" or "tri".')
    
    if kd_vor_super not in ['super','alphaKD','alphaVOR']:
        return('specify with second argument "super","alphaKD", or "alphaVOR" depending on what you are comparing')

    for dot in QD_list:
        if kd_vor_super =='super':
            alpha = get_alpha_fft(dot)
        elif kd_vor_super == 'alphaKD':
            alpha = get_alpha_arg(dot,'KD')
        else:
            alpha = get_alpha_arg(dot,'vor')
        if alpha != None:
            if abst:
                alpha = abs(alpha)
            alphalist.append(np.append(dot.cm,alpha))

    alpha_trs = np.transpose(alphalist)
    yval = alpha_trs[0]
    xval = alpha_trs[1]
    alpha_val = alpha_trs[2]
    
    # see the code of the function for this, but it shifts the midpoint of the cmap so you can 
    # see more detail at a specified point (adjust midpoint)
    # to have it do this you have to change cmap to the shifted cmap in the plot
    
    #shifted_cmap = shiftedColorMap(plt.cm.inferno, midpoint=0.25, name='shifted')
    
    # turns off interactivity of matplotlib, uncomment this if you want the plot to not show up but want to save it
    # plt.ioff()
        
    fig, axes = plt.subplots()
    
    # inferno, plasma both pretty good, switch to shifted if you want
    # s size 15 or 20 if you want to save it, otherwise i think 150? i dont remember
    # set cmap to 'shifted' if you want it to be the one set above
    #'bwr' or any divergent for not absolute value of alpha
    
    if abst:
        colormap = 'inferno'
    else:
        colormap = 'bwr'
    
    if hithresh > 100:
        markers = plt.scatter(xval, yval, c = alpha_val, cmap = colormap, marker='o', s = 20) #, vmin = 0, vmax = 10)
    else:
        markers = plt.scatter(xval, yval, c = alpha_val, cmap = colormap, marker='o', s = 20, vmin = lowthresh, vmax = hithresh)    
    
    axes = plt.gca()
    axes.set_xlim([0,4096])
    axes.set_ylim([0,4096])
    if abst:
        cbar = fig.colorbar(markers, extend='max')
        cbar.cmap.set_over('white')
        if kd_vor_super =='super':
             cbar.set_label(r'$|\alpha^{\mathrm{fft}}|$')
        elif kd_vor_super == 'alphaKD':
            cbar.set_label(r'$|\alpha^{KD}|$')
        else:
            cbar.set_label(r'$|\alpha^{vor}|$')
        
    else:
        cbar = fig.colorbar(markers, extend='both')
        cbar.cmap.set_under('green')
        cbar.cmap.set_over('green')
        if kd_vor_super =='super':
             cbar.set_label(r'$\alpha^{\mathrm{fft}}$')
        elif kd_vor_super == 'alphaKD':
            cbar.set_label(r'$\alpha^{KD}$')
        else:
            cbar.set_label(r'$\alpha^{vor}$')
    im = orig_image
    plt.imshow(im, cmap = 'gray')
    plt.axis('off')
    plt.show()

    if save == 'save':
        axes.axis('off')
        img_name = QD_list[0].image
        output_path = "./outputs/compilation/" + img_name + "/"
        if not exists(output_path):
            makedirs(output_path)
        file_name = img_name + '_alpha_' + tag + '_cmap'
        plt.savefig(output_path +'/'+ file_name + '.jpg' ,dpi=1000,bbox_inches='tight')
    
    # turns on interactivity -- do this if you turned it off above
    # plt.ion()
                
def show_alpha_neighbors(QD_list, orig_image, ave_all,tag,tri,save = False,lowthresh = 0,hithresh = 110):
    """
    colormap of alpha^alpha and alpha^<alpha>
    ave_all
    """
    alphalist = []
    Xo = []
    Yo = []
    Xo2 = []
    Yo2 = []
    
    if tri == 'abs':
        abst = True
    elif tri == 'tri':
        abst = False
    else:
        return('tri should be "abs" or "tri".')
    
    if ave_all not in ['ave','all']:
        return('specify with second argument "ave" or "all" if you want the average or for each dot. ')

    for dot in QD_list:
        if ave_all =='ave':
            alpha = dot.ave_neighbor_alpha
        else:
            alpha = dot.neighbor_alphas
        if alpha != None:
            if ave_all == 'ave':
                if abst:
                    alpha = abs(alpha)
                alphalist.append(np.append(dot.cm,alpha))
            else: 
                for pair in alpha:
                    # print(pair,'dot.cm',dot.cm, 'other cm: ', QD_list)
                    cm = [(dot.cm[0]+QD_list[pair[0]].cm[0])/2,(dot.cm[1]+QD_list[pair[0]].cm[1])/2]
                    if abst:
                        alpha = abs(pair[1])
                    else:
                        alpha = pair[1]
                    alphalist.append(np.append(cm,alpha))
                    
        if np.all(dot.orientation[0] == '110'):
            Yo.append(dot.cm[0])
            Xo.append(dot.cm[1])
        if np.all(dot.orientation[0] == '111'):
            Yo2.append(dot.cm[0])
            Xo2.append(dot.cm[1])
    
    alpha_trs = np.transpose(alphalist)
    yval = alpha_trs[0]
    xval = alpha_trs[1]
    alpha_val = alpha_trs[2]
    
    shifted_cmap = shiftedColorMap(plt.cm.inferno, midpoint=0.2, name='shifted')
    
    # turns off interactivity of matplotlib, uncomment this if you want the plot to not show up but want to save it
    # plt.ioff()
        
    fig, axes = plt.subplots()
    
    if abst:
        colormap = 'shifted'
    else:
        colormap = 'bwr'
    
    if hithresh > 100:
        markers = plt.scatter(xval, yval, c = alpha_val, cmap = colormap, marker='o', s = 20) #, vmin = 0, vmax = 10)
    else:
        markers = plt.scatter(xval, yval, c = alpha_val, cmap = colormap, marker='o', s = 20, vmin = lowthresh, vmax = hithresh)    
    
    axes.set_xlim([0,4096])
    axes.set_ylim([4096,0])
    if abst:
        cbar = fig.colorbar(markers, extend='max')
        cbar.cmap.set_over('white')
        if ave_all == 'ave':
            cbar.set_label(r'$|<\alpha^{\alpha}>|$')
        else:
            cbar.set_label(r'$|\alpha^{\alpha}|$')
        
    else:
        cbar = fig.colorbar(markers, extend='both')
        cbar.cmap.set_under('green')
        cbar.cmap.set_over('green')
        if ave_all == 'ave':
            cbar.set_label(r'$<\alpha^{\alpha}>$')
        else:
            cbar.set_label(r'$\alpha^{\alpha}$')

    im = orig_image
    plt.imshow(im, cmap = 'gray')
    # plt.scatter(Xo,Yo,color = 'green',s=20)
    # plt.scatter(Xo2,Yo2,color = 'cyan',s=20)
    if ave_all == 'ave':
        plt.title(r'$<\alpha^{\alpha}>$' + ' average misalignment between neighbors')
    else:
        plt.title(r'$\alpha^{\alpha}$' + ' misalignment between individual neighbors')
    axes.invert_yaxis
    plt.show()

    if save == True:
        axes.axis('off')
        img_name = QD_list[0].image
        output_path = "../presentables/Figures/" 
        plt.axis('off')
        if not exists(output_path):
            makedirs(output_path)
        if ave_all == 'ave':
            if abst:
                file_name = img_name + '_alpha_<alpha>_abs_' + tag + '_cmap'
            else:
                file_name = img_name + '_alpha_<alpha>_tri_' + tag + '_cmap'
        else:
            if abst:
                file_name = img_name + '_alpha_alpha_abs_' + tag + '_cmap'
            else:
                file_name = img_name + '_alpha_alpha_tri_' + tag + '_cmap'
        
        plt.savefig(output_path + file_name + '.png' ,dpi = 1000,bbox_inches='tight')
    
    # turns on interactivity -- do this if you turned it off above
    # plt.ion()

def show_dif_aveL(state,QD_list,bKD_bVOR,save,thresh = 10.001):
    '''
    Colormap of beta - <beta> for beta KD or Vor. 
    Might be kind of interesting sometimes, as I do care about the plot of alpha vs beta - <beta>. 
    returns a list of all the beta - <beta> values. 
    '''
    # alpha is beta i think
    alphalist = []
    
    if bKD_bVOR == 'bKD':
        ave_L = state.get_mean_beta(QD_list,'KD')
    elif bKD_bVOR == 'bVOR':
        ave_L = state.get_mean_beta(QD_list,'vor')
    else:
        return('Please input "bKD" or "bVOR" to specify which local orientation you want compared.')
    
    for dot in QD_list:
        if bKD_bVOR == 'bKD':
            alpha = dot.local_orientation_KD
        else:
            alpha = dot.local_orientation_vor
        if alpha != None:
            dif = np.abs(alpha - ave_L)
            if dif > 45:
                dif = 90 - dif
            alphalist.append(np.append(dot.cm,dif))

    alpha_trs = np.transpose(alphalist)
    yval = alpha_trs[0]
    xval = alpha_trs[1]
    alpha_val = alpha_trs[2]

    if thresh == 10.001:
        thresh = max(alpha_val)

    # see below code for this, but it shifts the midpoint of the cmap so you can 
    # see more detail at a specified point (adjust midpoint)
    # to have it do this you have to change cmap to the shifted cmap in the plot
    shifted_cmap = shiftedColorMap(plt.cm.inferno, midpoint=0.25, name='shifted')
    
    # inferno, plasma both pretty good, switch to shifted if you want
    # s size 15 or 20 if you want to save it, otherwise i think 200? i dont remember
    # set cmap to 'shifted' if you want it to be the one set above
    #'bwr' or any divergent for not absolute value of alpha
    fig, axes = plt.subplots()
    markers = plt.scatter(xval, yval, c = alpha_val, cmap = 'shifted', marker='o', s = 20, vmin = 0, vmax = thresh)
    axes = plt.gca()
    axes.set_xlim([0,4096])
    axes.set_ylim([0,4096])
    cbar = fig.colorbar(markers, extend='max')
    cbar.cmap.set_over('white')
    plt.title('Cmap of difference between ' + r'$\beta^{VOR}$' + ' and ' + r'$<\beta^{VOR}>$')
    im = state.orig_image
    plt.imshow(im, cmap = 'gray')
    plt.axis('off')
    plt.show()

    if save == 'save':
        axes.axis('off')
        # img_name and output_path are defined when you load the image
        file_name = img_name + ' ' + bKD_bVOR + '_dif_from_mean_cmap'
        plt.savefig(output_path +'/'+ file_name + '.jpg' ,dpi=1000,bbox_inches='tight')
    
    # return(alpha_val)

def voronoi_color(QD_list, orig_image, mode = 'alpha', save = False):
    cmlist = np.array([dot.cm for dot in QD_list])
    vor = Voronoi(cmlist)
    
    i = 0
    patches = []
    maglist = []
    arrows = []
    for dot in QD_list: 
        cm = dot.cm
        indices = get_vertices(i, cm, vor)
        for pos in indices: 
            pos2 = np.copy(pos)
            pos[0], pos[1] = pos2[1], pos2[0]
        i += 1

        if np.shape(indices)[0] > 3:
            region = matplotlib.patches.Polygon(indices, closed=True, edgecolor = None)
            patches.append(region)
            if mode == 'alpha': 
                if dot.ave_neighbor_alpha is not None:
                    maglist.append(np.abs(dot.ave_neighbor_alpha))
                else: 
                    maglist.append(-1)
            elif mode == 'psi4Vor':
                file_name = 'psi4_overlay'
                if dot.psi4vor is not None:
                    maglist.append(np.abs(dot.psi4vor))
                    theta = np.radians(get_theta(dot.psi4vor))
                    y = dot.cm[0]
                    x = dot.cm[1]
                    dx = np.cos(theta)
                    dy = np.sin(theta)
                    scale_f = 115
                    xy = np.array([[x,y],[x,y],[x,y]])+(np.array([[dx,dy],[-dy/4,dx/4],[dy/4,-dx/4]])-np.array([[dx/2.,dy/2],[dx/2.,dy/2],[dx/2.,dy/2]])) * scale_f
                    triangle=matplotlib.patches.Polygon(xy, closed=True, edgecolor = None)
                    arrows.append(triangle)
                else:
                    maglist.append(-1)
            elif mode == 'psi6Vor':
                file_name = 'psi6_overlay'
                if dot.psi6vor is not None:
                    maglist.append(np.abs(dot.psi6vor))
                    theta1 = get_theta(dot.psi6vor) % 60 
                    theta = np.radians(theta1)
                    y = dot.cm[0]
                    x = dot.cm[1]
                    dx = np.cos(theta)
                    dy = np.sin(theta) * -1
                    scale_f = 115
                    xy = np.array([[x,y],[x,y],[x,y]])+(np.array([[dx,dy],[-dy/4,dx/4],[dy/4,-dx/4]])-np.array([[dx/2.,dy/2],[dx/2.,dy/2],[dx/2.,dy/2]])) * scale_f
                    triangle=matplotlib.patches.Polygon(xy, closed=True, edgecolor = None)
                    arrows.append(triangle)
                else:
                    maglist.append(-1)
            elif mode == 'psi4_alpha': 
                file_name = 'psi4_dif'
                if dot.ave_neighbor_psi4 is not None: 
                    maglist.append(np.abs(dot.ave_neighbor_psi4))
                else:
                    maglist.append(-1)
            else: 
                return('wrong input for mode. Options are "alpha" or "psi4Vor" or "psi4_alpha".')
    
    shifted_cmap = shiftedColorMap(plt.cm.inferno, midpoint=0.6, name='shifted_inf')
    fig, ax = plt.subplots() 
    p = PatchCollection(patches)
    ax.add_collection(p)
    ax.invert_yaxis
    if mode == 'alpha' or mode == 'psi4_alpha':
        p.set(array= np.array(maglist), cmap='viridis')
        p.set_clim(0, 15)
        cbar = fig.colorbar(p, extend = 'both')
        cbar.cmap.set_under('green', alpha = 0)
        cbar.cmap.set_over('orange')
        plt.imshow(orig_image, cmap = 'gray')
    else: 
        p.set(array= np.array(maglist), cmap='shifted_inf', alpha = 1)
        p.set_clim(0,1)
        cbar = fig.colorbar(p)
        p2 = PatchCollection(arrows, color = 'black')
        ax.add_collection(p2)
        plt.imshow(orig_image, cmap = 'gray', alpha = 1)

    plt.show()
        
    if save == True: 
        plt.axis('off')
        img_name = QD_list[0].image
        output_path = "../presentables/QD_manuscript/"
        file = img_name + file_name
        plt.savefig(output_path + file + '.png' ,dpi = 1000,bbox_inches='tight')
    return

def voronoi_fill(QD_list, orig_image, mode = 'alpha_N', save = False):
    cmlist = np.array([dot.cm for dot in QD_list])
    vor = Voronoi(cmlist)
    
    i = 0
    patches = []
    maglist = []
    for dot in QD_list: 
        cm = dot.cm
        indices = get_vertices(i, cm, vor)
        for pos in indices: 
            pos2 = np.copy(pos)
            pos[0], pos[1] = pos2[1], pos2[0]
        i += 1

        if np.shape(indices)[0] > 3:
            region = matplotlib.patches.Polygon(indices, closed=True, edgecolor = None)
            patches.append(region)
            check = dot.cm[0] > 225 and dot.cm[1] > 225 and dot.cm[0] < 3871 and dot.cm[1] < 3871
            if mode == 'alpha_N': 
                file_name = 'alpha_neighbors_cmap'
                if dot.ave_neighbor_alpha is not None:
                    maglist.append(np.abs(dot.ave_neighbor_alpha))
                else: 
                    maglist.append(-1)
            elif mode == 'psi4Vor':
                file_name = 'psi4_cmap'
                if dot.psi4vor is not None:
                    maglist.append(np.abs(dot.psi4vor))
                else:
                    maglist.append(-1)
            elif mode == 'psi6Vor':
                file_name = 'psi6vor'
                if dot.psi6vor is not None:
                    maglist.append(np.abs(dot.psi6vor))
                else:
                    maglist.append(-1)
                # file_name = 'psi6_cmap_dif'
                # if dot.psi6vor is not None:
                #     maglist.append(np.abs(dot.psi4vor) - np.abs(dot.psi6vor))
                # else:
                #     maglist.append(-1)

            elif mode == 'alpha_Psi4':
                if dot.ave_neighbor_psi4 is not None: 
                    maglist.append(np.abs(dot.ave_neighbor_psi4))
                else: 
                    maglist.append(-1)
            elif mode == 'vor_area': 
                if dot.vor_area is not None and check: 
                    maglist.append(dot.vor_area)
                else: 
                    maglist.append(-1)
            elif mode == 'alpha_gl':
                if dot.orientation[1]: 
                    maglist.append(dot.orientation[1][0] % 90)
                else:
                    maglist.append(-1)
            elif mode == 'psi4_gl':
                if dot.psi4vor is not None: 
                    maglist.append(get_theta(dot.psi4vor)%90)
                else:
                    maglist.append(-1)
            else: 
                return('wrong input for mode. Options are "alpha_N", "Alpha_Psi4", "psi4Vor", "psi6Vor", or "vor_area", "psi4_gl" or "alpha_gl".')
    
    shifted_cmap = shiftedColorMap(plt.cm.inferno, midpoint=0.6, name='shifted_inf')
    fig, ax = plt.subplots() 
    p = PatchCollection(patches)
    ax.add_collection(p)
    ax.invert_yaxis

    if mode == 'alpha_N' or mode == 'alpha_Psi4':
        p.set(array= np.array(maglist), cmap='viridis')
        p.set_clim(0, 15)
        cbar = fig.colorbar(p, extend = 'both')
        cbar.cmap.set_under('green', alpha = 0)
        cbar.cmap.set_over('orange')
        plt.imshow(orig_image, cmap = 'gray')
    elif mode == 'psi4Vor' or mode == 'psi6Vor': 
        p.set(array= np.array(maglist), cmap='shifted_inf')
        p.set_clim(0,1)
        cbar = fig.colorbar(p)
        plt.imshow(orig_image, cmap = 'gray', alpha = 1)

    elif mode =='vor_area': 
        maglist = np.array(maglist)
        cmap = matplotlib.cm.viridis
        cmap.set_bad('green', 0)

        mag_ma = np.ma.masked_where(maglist < 0, maglist)
        p.set(array = mag_ma, cmap= cmap)
        # p.set_clim(0, 15)
        cbar = fig.colorbar(p, extend = 'both')
        # cbar.cmap.set_under('green', alpha = 0)
        # cbar.cmap.set_over('orange')
        plt.imshow(orig_image, cmap = 'gray')

    elif mode == 'alpha_gl' or mode == 'psi4_gl': 
        maglist = np.array(maglist)
        cmap = matplotlib.cm.seismic
        cmap.set_bad('green', 0)
        mag_ma = np.ma.masked_where(maglist < 0, maglist)
        p.set(array = mag_ma, cmap= cmap)
        # p.set_clim(0, 90)
        cbar = fig.colorbar(p)
        plt.imshow(orig_image, cmap = 'gray')

    plt.show()
        
    if save == True: 
        plt.axis('off')
        img_name = QD_list[0].image
        output_path = "../presentables/QD_manuscript/"
        file = img_name + file_name
        plt.savefig(output_path + file + '.png' ,dpi = 1000,bbox_inches='tight')
    return


def psi6V(QD_list):
    cmlist = np.array([dot.cm for dot in QD_list])
    retlist = []

    # fucking magic. 
    vor = Voronoi(cmlist)

    # Cuz the way Voronoi works it keeps track of points by index in the original list so I need to keep track too.
    i = 0

    for dot in QD_list: 
        point = dot.cm
        psi6 = 0
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

            # this loops over each edge and calculates its component of the psi6 total
            for edge in edges: 
                length = plength(edge)

                # the slope between the two points is the inverse of the slope of the edge
                dx = edge[0,0]-edge[1,0]
                dy = edge[0,1]-edge[1,1]
                theta = np.arctan2(dy,dx)

                # component of psi4 for this point due to this individual edge
                psip = length/totL * np.exp(6j*theta)
                psi6 += psip

            # assign dot to have psi4vor value 
            dot.psi6vor = psi6
        else:
            dot.psi6vor = None

def voronoi_neighbors(QD_list):
    cmlist = np.array([dot.cm for dot in QD_list])
    vor = Voronoi(cmlist)

    i = 0
    for dot in QD_list: 
        nlist = []
        strainlist = []
        A = get_vertices(i,cmlist[i], vor)
        j = 0
        # gets the neighbors' QDindex and puts in QD.neighborsvor
        for tdot in QD_list:
            B = get_vertices(j, cmlist[j],vor)
            for vertex in A: 
                if vertex in B and i != j:
                    if tdot.QD_index not in nlist:
                        nlist.append(tdot.QD_index)
                    strainlist.append([tdot.QD_index, vertex])
            j += 1 

        if np.shape(A)[0] > 2:
            hull = ConvexHull(A)
            dot.vor_area = hull.volume
        else: 
            dot.vor_area = None
        dot.neighborsvor = nlist
        dot.vor_cell = strainlist
        i += 1


def get_strain(QD_list, normed = False):
    i = 0
    for dot1 in QD_list:
        qdindex = None
        cell_perim = 0
        strain = 0
        for vertex in dot1.vor_cell: 
            # checking if its for a new wall shared with a different QD
            if vertex[0] != qdindex: 
                v1 = vertex[1]
                qdindex = vertex[0]
            # so it's the second vertex for a wall
            else: 
                v2 = vertex[1]
                wall_length = ((v2[0] - v1[0])**2 + (v2[1] - v1[1])**2)**0.5
                cell_perim += wall_length
                
                dot1cm = dot1.cm
                dot2cm = QD_list[qdindex].cm
                dist = ((dot2cm[0] - dot1cm[0])**2 + (dot2cm[1] - dot1cm[1])**2)**0.5
                
                strain += dist * wall_length
        # normalize the strain by the perimeter 
        if normed == True and cell_perim != 0: 
            strain = strain / cell_perim
        dot1.strain = strain
        i += 1
        y, x= dot1.cm[0], dot1.cm[1]
        if y < 250 or x < 250 or y > 3846 or x > 3846: 
            dot1.strain = None

def get_psi4_neighbors(tdot, QD_list):
    '''
    For a dot looks at neighbors and gets the misalignment between the dot and its neighbors. 
    Assigns to dot.neighbor_alphas a list of misalignment for the dot (referenced by QD_index of other dot)
    and for dot.ave_neighbor_alpha the average absolute value misalignment form neighbors. 
    '''
    a1 = tdot.psi4vor
    tdot.neighbor_psi4s = []
    neighborlist = []
    for index in tdot.neighborsKD:
        neighborlist.append(QD_list[index])

    # for index in tdot.neighborsvor:
    #     neighborlist.append(QD_list[index])

    diflist = []
    if a1:
        for ndot in neighborlist:    
            append = True
            n1 = ndot.psi4vor
            # if n1 doesnt have an orientation cant get relative misorientations 
            if not n1:
                append = False
            else: 
                theta1 = np.radians(get_theta(a1))
                theta2 = np.radians(get_theta(n1))
                dif = (theta2 - theta1) / np.pi * 180
                
            if append == True:
                if dif < -45:
                    fdif = 90 + dif
                elif dif > 45:
                    fdif = dif - 90
                else:
                    fdif = dif  

                tdot.neighbor_psi4s.append([ndot.QD_index,fdif])
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
            tdot.ave_neighbor_psi4 = avealpha 
        else: 
            tdot.ave_neighbor_psi4 = None
    else:
        tdot.ave_neighbor_psi4 = None

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    -----------------------------------------------------------------------
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    from mpl_toolkits.axes_grid1 import AxesGrid
    import matplotlib
    
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }
    
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
