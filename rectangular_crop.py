from __future__ import print_function

"""
The class selects a rectangular region of intrest (ROI) over a figure by mouse clicking and dragging.
Features:
- ROI marking,
- multiple reselection actions allowed,
- pressing q-key closes the figure,
- ensures even dimensions of ROI.
"""

from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


class RoiRect():
    def __init__(self, fig=[], ax=[], roicolor='b'):
        if not fig:
            fig = plt.gcf()

        if not ax:
            ax = plt.gca()

        self.fig = fig
        self.ax = ax
        self.ax.set_title('Select ROI. Press q to confirm and exit.')
        self.roicolor = roicolor
        self.RS = RectangleSelector(ax,
                                    self.__line_select_callback,
                                    useblit=True,
                                    button=[1, 3],  # don't use middle button
                                    minspanx=5, minspany=5,
                                    spancoords='pixels',
                                    interactive=True)#drawtype='box',
        fig.canvas.mpl_connect('key_press_event', self.__toggle_selector)
        plt.show()
        self.extents = self.get_extents()

    def get_extents(self):
        extents = self.RS.extents
        extents = np.rint(extents)
        extents_int = [int(x) for x in extents]
        # ensure even number of columns and rows
        x_width = extents_int[1] - extents_int[0] + 1
        y_width = extents_int[3] - extents_int[2] + 1
        if np.mod(x_width, 2):
            extents_int[1] = extents_int[1] + 1
        if np.mod(y_width, 2):
            extents_int[3] = extents_int[3] + 1
        return extents_int  # Return (xmin, xmax, ymin, ymax)

    def get_mask(self, currentImage):
        mask = np.zeros_like(currentImage, dtype=bool)
        mask[self.extents[2]:self.extents[3]+1, self.extents[0]:self.extents[1]+1] = True
        return mask

    def display_roi(self,**linekwargs):
        ax = plt.gca()
        # Create an rectangle patch
        ymin = self.extents[2]
        xmin = self.extents[0]
        ywidth = self.extents[3] - self.extents[2] + 1
        xwidth = self.extents[1] - self.extents[0] + 1
        rect = patches.Rectangle((xmin, ymin), xwidth, ywidth, edgecolor=self.roicolor, facecolor='none', **linekwargs)
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.draw()

    def display_stat(self, currentImage, **textkwargs):
        mask = self.get_mask(currentImage)
        meanval = np.mean(np.extract(mask, currentImage))
        stdval = np.std(np.extract(mask, currentImage))
        string = "%.3f +- %.3f" % (meanval, stdval)
        ymin = self.extents[2]
        xmin = self.extents[0]
        plt.text(xmin, ymin, string, color=self.roicolor,
                 bbox=dict(facecolor='w', alpha=0.6), **textkwargs)

    def __line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        # print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        # print(" The button you used were: %s %s" % (eclick.button, erelease.button))

    def __toggle_selector(self, event):
        # print(' Key pressed.')
        if event.key in ['Q', 'q'] and self.RS.active:
            # print(' RectangleSelector deactivated.')
            self.RS.set_active(False)
        if event.key in ['A', 'a'] and not self.RS.active:
            # print(' RectangleSelector activated.')
            self.RS.set_active(True)