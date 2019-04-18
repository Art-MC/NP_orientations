import numpy as np
import matplotlib.pyplot as plt
from helpers_orig import *


def QD_check(im, peaks):
        '''a variation of clickable_im()
        this takes in a QD_list and image to help the dot finder.'''
        # note to self, points are [y,x]
        end_array = [0]
        def OnClick(event):
            if event.button == 3: #right click
                y = event.ydata
                x = event.xdata
                print('adding [',y,', ',x,']')
                peaks.append([y,x])

            if event.button == 2: # middle click
                print('closing')
                end_array[:] = list([1])

        def on_key(event):     
            if event.key == 'delete':
                y = event.ydata
                x = event.xdata
                print('delete pressed')
                i = 0
                for cm in peaks:
                    yD = cm[0]
                    xD = cm[1]
                    if y - 20 < yD and y + 20 > yD and x - 20 < xD and x + 20 > xD:
                        print('delete ',cm)
                        peaks.pop(i)
                    i += 1

        fig, ax = plt.subplots()
        ax.matshow(im, cmap = 'gray')
        plt.title("Right click to add, delete to remove, middle click to save and exit", fontsize=11)
        peaks2 = np.array(peaks)
        ax.plot(peaks2[:,1],peaks2[:,0],
            linestyle='None',marker='o',color='r',fillstyle='none')

        connection_id = fig.canvas.mpl_connect('button_press_event', OnClick)
        cid = fig.canvas.mpl_connect('key_press_event', on_key)
        while True:
            plt.ginput(-1, mouse_add = 3, mouse_pop = -1, show_clicks = True)
            plt.pause(1)
            if end_array[0]:
                break

        plt.close()

        print('''These are the final ponts. \n
            Click to close the figure and you'll be prompted 
            if you want to try again. ''')

        fig, ax = plt.subplots()
        ax.matshow(im, cmap = 'gray')
        plt.title("Final points. Click to close the figure", fontsize=11)
        peaks2 = np.array(peaks)
        ax.plot(peaks2[:,1],peaks2[:,0],
            linestyle='None',marker='o',color='r',fillstyle='none')
        plt.waitforbuttonpress()
        plt.close()
        return(peaks)


# def test_cursor2(im, peaks):
#     import time

#     def tellme(s):
#         print(s)
#         plt.title(s, fontsize=11)
#         plt.draw()

#     plt.clf()
#     fig, ax = plt.subplots()
#     ax.matshow(im, cmap = 'gray')

#     tellme('click to add new QD, delete to remove')

#     plt.waitforbuttonpress()

#     while True:
#         pts = []
#         while len(pts) < 3:
#             tellme('Select 3 corners with mouse')
#             pts = np.asarray(plt.ginput(3, timeout=-1))
#             if len(pts) < 3:
#                 tellme('Too few points, starting over')
#                 time.sleep(1)  # Wait a second

#         ph = plt.fill(pts[:, 0], pts[:, 1], 'r', lw=2)

#         tellme('Happy? Key click for yes, mouse click for no')

#         if plt.waitforbuttonpress():
#             break

#         # Get rid of fill
#         for p in ph:
#             p.remove()

#     # Define a nice function of distance from individual pts
#     def f(x, y, pts):
#         z = np.zeros_like(x)
#         for p in pts:
#             z = z + 1/(np.sqrt((x - p[0])**2 + (y - p[1])**2))
#         return 1/z


#     X, Y = np.meshgrid(np.linspace(-1, 1, 51), np.linspace(-1, 1, 51))
#     Z = f(X, Y, pts)

#     CS = plt.contour(X, Y, Z, 20)

#     tellme('Use mouse to select contour label locations, middle button to finish')
#     CL = plt.clabel(CS, manual=True)


#     tellme('Now do a nested zoom, click to begin')
#     plt.waitforbuttonpress()

#     while True:
#         tellme('Select two corners of zoom, middle mouse button to finish')
#         pts = np.asarray(plt.ginput(2, timeout=-1))

#         if len(pts) < 2:
#             break

#         pts = np.sort(pts, axis=0)
#         plt.axis(pts.T.ravel())

#     tellme('All Done!')
#     plt.close()