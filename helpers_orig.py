
'''
General use functions. I tried to get rid of the ones I moved inside the state class.
No guarantees that all are used/helpful, I just included all of my in progress stuff as well. 
Not at all sorted. Not even a little bit. 

plenty of these things got developed at different times and are therefore a little redundant.
i decided to keep all of them that are currently used anywhere, or seem likely to be used or helpful.
(in no particular order)
'''

import matplotlib.pyplot as plt
import numpy as np


def get_histo(im, minn, maxx, numbins,tag=None):
    '''
    gets a histogram of a list of datapoints (im), specify minimum value, maximum value, and number of bins
    '''
    fig,ax = plt.subplots()
    ax.hist(np.ravel(im),bins=np.linspace(minn,maxx,numbins))   
    if tag != None:     
        plt.savefig("./outputs/compilation/"+ tag + '.tif' ,dpi=200,bbox_inches='tight')
    plt.show(block=False) 
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

def show_im(im, title=None):
    '''shows an image with matplotlib'''
    fig,ax = plt.subplots()
    ax.matshow(im, cmap = 'gray')
    ax.set_title(str(title))
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
