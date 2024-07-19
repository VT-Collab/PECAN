import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import sys


class plotGUI():
    
    def __init__(self, task_idx, prev_styles, show_Demos=False, corner=None, target_style=None):
        
        # Initialize variables
        self.task_idx = task_idx
        self.prev_styles = prev_styles['styles']
        self.terminate = False
        self.style = None
        first_time = True

        # Initialize figure
        self.fig, self.ax = plt.subplot_mosaic([
                                                ['main', 'button']],
                                                width_ratios=[2.5, 1],
                                                layout='constrained'
                                               ) 
        self.fig.suptitle('Current Target Style: ' + str(target_style) + ", Press 'Enter' or button to play")  

        # Configure play button
        playButton = Button(self.ax['button'], 'Play')
        playButton.on_clicked(self.playButton)

        # Setup demo plot behavior
        if show_Demos:
            self.style = corner
            first_time = False
            timer = self.fig.canvas.new_timer(interval=1000)
            timer.add_callback(closeDemo())
            timer.start()
        
        # Initial figure draw
        self.updatePlot(first_time=first_time)

        # Disable default matplolib mouse and keyboard behavior
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.button_press_handler_id)

        # Enable custom mouse and keyboard behavior
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse)
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        
        plt.show()
    
        
    def on_press(self, event):
        sys.stdout.flush()

        if event.key == 'enter':
            plt.close()
        
        if event.key == 'escape':
            self.terminate = True
            plt.close()


    def on_mouse(self, event):
        sys.stdout.flush()
        
        # mouse button left click
        if event.button == 1 and event.inaxes == self.ax['main']:
            self.style = (event.xdata, event.ydata)
            # self.updatePlot(style=self.style)
            self.updatePlot()

    def updatePlot(self, first_time=False):
        self.ax['main'].clear()
        self.ax['main'].grid()
        self.ax['main'].set_xlim(-1.05, 1.05)
        self.ax['main'].set_ylim(-1.05, 1.05)

        for style in self.prev_styles:
            self.ax['main'].scatter(style[0], style[1], c='tab:grey')
            plt.draw()

        if not first_time:
            self.ax['main'].scatter(self.style[0], self.style[1], c='tab:orange')
            plt.draw()

    def taskfunc(self, label):
        taskdict = {"Highway": 0, "Intersection": 1}
        self.task_idx = taskdict[label]
        self.updatePlot(first_time=True)

    def playButton(self, event):
        plt.close()

class closeDemo(object):

    def __init__(self):
        self.first = True

    def __call__(self):
        if self.first:
            self.first = False
            return
        plt.close()