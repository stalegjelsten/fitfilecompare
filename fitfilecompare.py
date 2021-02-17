# %% [markdown]
# # Compare .FIT-file power data
# Use this tool to compare power data from different sources.
# 
# The .fit-file parser and its connected functions are made by Aart
# Goossens and published under a MIT license at
# https://github.com/GoldenCheetah/sweatpy

import PySimpleGUI as sg
import pathlib
from fitparse import FitFile
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, \
    NavigationToolbar2Tk
from matplotlib.dates import date2num, num2date, DateFormatter


################################
# The following functions are made by Aart Goossens. MIT license. ↓
################################
def resample_data(data, resample: bool,
         interpolate: bool) -> pd.DataFrame:
    """Function to calculate the mean max power for all available 
            activities.
    Args:
            data: The data frame that needs to be resampled and/or
            interpolated resample: whether or not the data frame needs
            to be resampled to 1Hz interpolate: whether or not missing
            data in the data frame needs to be interpolated
    Returns:
        Returns the resampled and interpolated dataframe
    """
    if resample:
        data = data.resample("1S").mean()

    if interpolate:
        data = data.interpolate(method="linear")

    return data

def remove_duplicate_indices(data: pd.DataFrame, keep="first") -> pd.DataFrame:
    """Function that removes duplicate indices
    Args:
        data:   The data frame for which duplicate indices need to be 
                deleted
        keep:   Determines which duplicates (if any) to mark. See
                pandas.DataFrame.index.duplicated documentation for 
                more information
    Returns:
        Returns the data frame with duplicate indices removed
    """
    return data[~data.index.duplicated(keep=keep)]


def semicircles_to_degrees(semicircles):
    return semicircles * (180 / (2 ** 31))

def read_fit(fpath, resample: bool = False, 
            interpolate: bool = False) -> pd.DataFrame:
    """
    This method uses the Python fitparse library to load a FIT file into
    a Pandas DataFrame. It is tested with a Garmin FIT file but will
    probably work with other FIT files too. Columns names are translated
    to sweat terminology (e.g. "heart_rate" > "heartrate").

    Args: 
        fpath:          str, file-like or Path object 
        resample:       whether or not the data frame needs to be 
                        resampled to 1Hz 
        interpolate:    whether or not missing data in the data frame 
                        needs to be interpolated 
        Returns: A pandas dataframe with all the data.
    """

    if isinstance(fpath, pathlib.PurePath):
        fpath = fpath.as_posix()

    fitfile = FitFile(fpath)

    records = []
    lap = 0
    session = -1
    for record in fitfile.get_messages():
        if record.mesg_type is None:
            continue

        if record.mesg_type.name == "record":
            values = record.get_values()
            values["lap"] = lap
            values["session"] = session
            records.append(values)
        elif record.mesg_type.name == "lap":
            lap += 1
        elif record.mesg_type.name == "event":
            if record.get_value("event_type") == "start":
                # This happens whens an activity is (manually or 
                # automatically) paused or stopped and the resumed
                session += 1
        elif record.mesg_type.name == "sport":
            # @TODO handle this to be able to return metadata
            pass

    fit_df = pd.DataFrame(records)

    fit_df = fit_df.rename(
        columns={
            "heart_rate": "heartrate",
            "position_lat": "latitude",
            "position_long": "longitude",
            "altitude": "elevation",
            "left_right_balance": "left-right balance",
        }
    )

    fit_df["datetime"] = pd.to_datetime(fit_df["timestamp"], utc=True)
    fit_df = fit_df.drop(["timestamp"], axis="columns")
    fit_df = fit_df.set_index("datetime")

    for coordinate in ["latitude", "longitude"]:
        if coordinate in fit_df.columns:
            fit_df[coordinate] = semicircles_to_degrees(fit_df[coordinate])

    fit_df = remove_duplicate_indices(fit_df)

    fit_df = resample_data(fit_df, resample, interpolate)

    return fit_df

################################
# ↑ The preceding functions are made by Aart Goossens. MIT License.
################################

class Workout_file:
    files = []

    def __init__(self, name, fullname, path):
        
        self.name = name
        self.fullname = fullname
        self.path = path
        self.df = read_fit(path, resample=True, interpolate=True)
        self.starttime = min(self.df.index)
        self.endtime = max(self.df.index)
        self.files.append(self)

# Setting PySimpleGUI options
sg.SetOptions(font=("Arial", 16))
sg.theme("DarkBlack1")

################################
# Opening a file picker window for the FIT-files and adding them to the paths
################################

window = sg.Window('Select .fit-files').Layout(
    [[sg.Text("Select all .fit-files from your computer.")],
    [sg.Text("Hold ctrl (CMD on mac) or shift to select multiple files")], 
    [sg.Input(key='_FILES_'), sg.FilesBrowse()], 
    [sg.OK(), sg.Cancel()]])
event, values = window.read()
paths = values['_FILES_'].split(";")

window.close()



################################
# Making a new window to give friendly names to the fit-files
################################

file_menu_column = [
    [
        sg.Text("Give your files friendly names:", size=(50,1))
    ]
]

################################
# Generating input fields with suggested names
################################

for path in paths:
    keeey = "_NAME" + path.split("/")[-1] + "_"
    file_menu_column.append([sg.InputText(path.split("/")[-1], 
                            key=keeey, size=(50,1))])

file_menu_column.append([sg.Button("Ok"), ])

layout = [
    [
        sg.Column(file_menu_column),
        sg.VSeparator()

    ]
]
window = sg.Window("Filfilecompare").Layout(layout)


################################
# Listening for data from GUI and populating the list of device names
################################

device_names = []

while True:
    event, values = window.read()
    if event == "Ok":
        for path in paths:
            device_names.append(values["_NAME" + path.split("/")[-1] + "_"])
        break
        window.close()

    elif event == sg.WIN_CLOSED:
        break

window.close()



############
# Creating Workout_file instances from fit-files
############

for name, path in zip(device_names, paths):
    name_simplified = name.replace(" ", "").lower()
    globals()[name_simplified] = Workout_file(name_simplified, name, path)



################################
# finding common data fields between all FIT files
################################
if len(Workout_file.files) > 1:
    a = np.intersect1d(Workout_file.files[0].df.columns, 
            Workout_file.files[1].df.columns)
    for i in range(len(Workout_file.files)-2):
        a = np.intersect1d(a, Workout_file.files[i+2].df.columns)
elif len(Workout_file.files) == 1:
    a = list(Workout_file.files[0].df.columns)
else:
    a = ["no data"]



################################
# TCX importer ↓. Not working, having problems with heart rate ATM.
################################
# %%
import xml.etree.ElementTree as ET

NAMESPACES = {
    "default": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2",
    "ns5": "http://www.garmin.com/xmlschemas/ActivityGoals/v1",
    "ns3": "http://www.garmin.com/xmlschemas/ActivityExtension/v2",
    "ns2": "http://www.garmin.com/xmlschemas/UserProfile/v2",
}


def xml_find_value_or_none(element, match, namespaces=None):
    e = element.find(match, namespaces=namespaces)

    if e is None:
        return e
    else:
        return e.text


def read_tcx(fpath, resample: bool = False,
         interpolate: bool = False) -> pd.DataFrame:
    """
    This method loads a TCX file into a Pandas DataFrame. Columns names 
    are translated to sweat terminology 
    (e.g. "heart_rate" > "heartrate").
    Args:

    Returns:
        A pandas data frame with all the data.
    """
    tree = ET.parse(pathlib.Path(fpath))
    root = tree.getroot()
    activities = root.find("default:Activities", NAMESPACES)

    records = []
    lap_no = -1
    session = 0
    for activity in activities.findall("default:Activity", NAMESPACES):
        for lap in activity.findall("default:Lap", NAMESPACES):
            lap_no += 1
            track = lap.find("default:Track", NAMESPACES)
            for trackpoint in track.findall("default:Trackpoint", NAMESPACES):
                datetime = xml_find_value_or_none(
                    trackpoint, "default:Time", NAMESPACES
                )
                elevation = xml_find_value_or_none(
                    trackpoint, "default:AltitudeMeters", NAMESPACES
                )
                distance = xml_find_value_or_none(
                    trackpoint, "default:DistanceMeters", NAMESPACES
                )
                cadence = xml_find_value_or_none(
                    trackpoint, "default:Cadence", NAMESPACES
                )

                position = trackpoint.find("default:Position", NAMESPACES)
                latitude = xml_find_value_or_none(
                    position, "default:LatitudeDegrees", NAMESPACES
                )
                longitude = xml_find_value_or_none(
                    position, "default:LongitudeDegrees", NAMESPACES
                )

                #hr = trackpoint.find("default:HeartRateBpm", NAMESPACES)
                #heartrate = xml_find_value_or_none(hr, "default:Value", NAMESPACES)
            
                hr = trackpoint.find("default:HeartRateBpm", NAMESPACES)
                if hr is not None:
                    heartrate = hr.find("default:Value", NAMESPACES)
                else:
                    heartrate = 0

                #hr = trackpoint.find("default:HeartRateBpm", NAMESPACES)
                #heartrate = xml_find_value_or_none(hr, "default:Value", NAMESPACES)
                #heartrate = xml_find_value_or_none("default:HeartRateBpm", "default:Value", NAMESPACES)

                extensions = trackpoint.find("default:Extensions",
                                                             NAMESPACES)
                if extensions:
                    tpx = extensions.find("ns3:TPX", NAMESPACES)
                    speed = xml_find_value_or_none(tpx, "ns3:Speed", 
                                                    NAMESPACES)
                    power = xml_find_value_or_none(tpx, "ns3:Watts", 
                                                    NAMESPACES)
                else:
                    speed = None
                    power = None

                records.append(
                    dict(
                        datetime=datetime,
                        latitude=latitude,
                        longitude=longitude,
                        elevation=elevation,
                        heartrate=heartrate,
                        cadence=cadence,
                        distance=distance,
                        speed=speed,
                        power=power,
                        lap=lap_no,
                        session=session,
                    )
                )

    tcx_df = pd.DataFrame(records)
    tcx_df = tcx_df.dropna("columns", "all")
    tcx_df["datetime"] = pd.to_datetime(tcx_df["datetime"], utc=True)
    tcx_df = tcx_df.set_index("datetime")

    # Convert columns to numeric if possible
    tcx_df = tcx_df.apply(pd.to_numeric, errors="ignore")

    tcx_df = remove_duplicate_indices(tcx_df)

    tcx_df = resample_data(tcx_df, resample, interpolate)

    return tcx_df


################################
# Helper functions to draw GUI
# Stolen from https://github.com/PySimpleGUI/PySimpleGUI/blob/…
# …master/DemoPrograms/Demo_Matplotlib_Embedded_Toolbar.py
################################
def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)

class Toolbar(NavigationToolbar2Tk):
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)


################################
# Setting up the UI
################################
menu_column = [
    [sg.T("Graph smoothing (seconds):")],
    [sg.Slider(range=(1,30),default_value=5, size=(20,10), 
        key="__SMOOTHING__", orientation="horizontal")],
    [sg.HSeparator()],
    [sg.T("Time offsets (seconds):")],
    [sg.HSeparator()],
    [sg.Text("Data available to graph:")]
]


################################
# Create time offset sliders for each file
################################
for name in device_names:
    device_offset = "__" + name.replace(" ", "").lower() + "__OFFSET__"
    menu_column.insert(-2, [sg.T(name, size=(20,1)), 
        sg.In(default_text="0", key=device_offset, size=(5,1))])



################################
# Setup Plot column UI
################################
graph_column = [
    [sg.Canvas(key='controls_cv')],
    [sg.Column(
        layout=[
            [sg.Canvas(key='fig_cv', size=(500 * 2, 500))]
        ],
        pad=(0, 0)
    )]
]


################################
# Finding common data fields between the files
################################
if len(Workout_file.files) > 1:
    a = np.intersect1d(Workout_file.files[0].df.columns, 
        Workout_file.files[1].df.columns)
    for i in range(len(Workout_file.files)-2):
        a = np.intersect1d(a, Workout_file.files[i+2].df.columns)

allkeys = []
if len(a) < 12:
    for metric in a:
        keeey = "__"+metric+"__"
        allkeys.append(keeey)
        menu_column.append([sg.Checkbox(metric, key=keeey)])
    menu_column.append([sg.B('Plot'), sg.B('Exit')])

###########
# Print data options in two columns if there are more than 12 data fields
# to prevent a very tall window
###########
else:
    for i in range(0,np.floor((len(a))/2).astype("int")):
        keeey = "__"+a[2*i]+"__"
        keeey2 = "__"+a[2*i+1]+"__"
        allkeys.append(keeey)
        allkeys.append(keeey2)
        menu_column.append([sg.Checkbox(a[2*i], key=keeey), 
            sg.Checkbox(a[2*i+1], key=keeey2)])
    if (len(a) % 2) != 0:
        keeey = "__"+a[-1]+"__"
        menu_column.append([sg.Checkbox(a[-1], key=keeey)])
    

    menu_column.append([sg.B('Plot'), sg.B('Exit')])


################################
# Create the two-column layout
################################
layout = [
    [
        sg.Column(menu_column),
        sg.VSeparator(),
        sg.Column(graph_column)

    ]
]


window = sg.Window('FitFileCompare', layout)


################################
# Listening for GUI interaction
################################
while True:
    event, values = window.read()
    
    if event in (sg.WIN_CLOSED, 'Exit'): 
        break

    elif event == 'Plot':

        smoothing = int(values["__SMOOTHING__"]) #read value from slider
        metrics = []

        ################################
        # Read what data fields have been selected
        ################################
        for keey in allkeys:
            if values[keey]:
                metrics.append(keey.split("__")[1])
        


        ################################
        # If no data fields have been selected, do nothing.
        # If else: plot
        ################################

        if len(metrics) > 0:
            number_of_plots = len(metrics)
            fig, ax = plt.subplots(number_of_plots, 1, 
                sharex=True, squeeze=False) # subplots stored as 2D array ax
            ax = ax.flatten()  # flatten the ax array to make indexing easier
            lines = []
            ax_number = 0
            for plot_name in metrics: # loop over all the subplots
                for file in Workout_file.files: #loop over all data files


                    ################################
                    # Read time offset values and apply offset to the dataframe
                    ################################
                    keystring = "__" + file.name + "__OFFSET__"
                    offset = int(values[keystring])
                    newfile = file.df[plot_name].fillna(0)
                    newfile.index = newfile.index + pd.Timedelta(
                                                        seconds=offset)
                    data = newfile.loc[:]

                    ################################
                    # Plot the lines and add Line object to lines array
                    ################################
                    line = ax[ax_number].plot_date(x=data.index, 
                        y=data.rolling(smoothing, center=True).mean(), 
                        marker='', linestyle='-')
                    lines.append(line)
                    

                ################################
                # on_xlims_change is a callback function that updates the 
                # legend of the plots each time the x-axis limits are changed,
                # e.g. when zooming the plot
                ################################
                def on_xlims_change(axes):
                    d1, d2 = axes.get_xlim()
                    
                    avg_data = []
                    max_data = []
                    
                    _, _, plot_no = axes.properties().get('geometry')
                    plot_no = plot_no-1

                    for file in Workout_file.files:
                        data = file.df.loc[num2date(d1):num2date(d2),:]\
                                            [metrics[plot_no]].fillna(0)
                        avg_data.append(str(np.round(np.average(data),1)))
                        max_data.append(str(np.round(max(data),1)))
                    
                        
                    legend = [i + " (" + k +" max) " + j for j, i, k in \
                                    zip(device_names, avg_data, max_data)]
                    axes.legend(legend)
                
                                    
                on_xlims_change(ax[ax_number])  # run the function on first
                                                # iteration to create legends

                
                ################################
                # matplotlib formatting options
                ################################
                ax[ax_number].xaxis.set_major_formatter(
                    DateFormatter('%H:%M:%S'))
                ax[ax_number].set_ylabel(metrics[ax_number])
                ax[ax_number].grid()

                ax[ax_number].callbacks.connect('xlim_changed',
                                                 on_xlims_change)
                ax_number = ax_number + 1

            draw_figure_w_toolbar(window['fig_cv'].TKCanvas, fig, 
                                    window['controls_cv'].TKCanvas)

window.close()
