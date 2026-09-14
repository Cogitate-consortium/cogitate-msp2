import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# New added :for save data
import pickle

# Set arial as the default font:
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False  # This ensures that minus signs are rendered correctly
})

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

fig_size = [183, 108]
def_cmap = 'RdYlBu_r'
plt.rc('font', size=22)  # controls default text sizes
plt.rc('axes', titlesize=22)  # fontsize of the axes title
plt.rc('axes', labelsize=22)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=22)  # fontsize of the tick labels
plt.rc('ytick', labelsize=22)  # fontsize of the tick labels
plt.rc('legend', fontsize=22)  # legend fontsize
plt.rc('figure', titlesize=22)  # fontsize of the fi

#%% _mm2inch
def _mm2inch(val):
    """
    Convert millimeters to inches.

    Parameters
    ----------
    val : float
        Value in millimeters.

    Returns
    -------
    float
        Value converted to inches.
    """
    return val / 25.4

#%% plot_matrix
def plot_matrix(data, 
                x0, 
                x_end, 
                y0, 
                y_end, 
                mask=None, 
                cmap=None, 
                ax=None, 
                ylim=None, 
                midpoint=None, 
                transparency=1.0,
                interpolation='lanczos',
                xlabel="Time (s)", 
                ylabel="Time (s)", 
                xticks=None, 
                yticks=None, 
                flag_return_image=0,
                flag_colorbar=1,
                colorbar_hline= None,  # New add
                cbar_label="Accuracy", 
                filename=None,
                vlines=0,
                hlines = None,
                title=None, 
                plotdata_save_folder_path=None,  # New add
                flag_sharp_contour=False,  # New add
                square_fig=False, 
                dpi=300):
    """
    Plot a 2D matrix with optional significance mask.

    This function is used to plot 2D matrices, such as temporal generalization decoding or time-frequency decompositions,
    with or without significance masking.

    Parameters
    ----------
    data : 2D numpy array
        Data to plot.
    x0 : float
        First sample value for the x-axis (e.g., the first time point in the data).
    x_end : float
        Final sample value for the x-axis (e.g., the last time point in the data).
    y0 : float
        First sample value for the y-axis (e.g., the first time point in the data).
    y_end : float
        Final sample value for the y-axis (e.g., the last time point in the data).
    mask : 2D numpy array of booleans, optional
        Significance mask (same size as data). True where the data are significant, False elsewhere.
    cmap : str, optional
        Name of the colormap.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot the data. If not provided, a new figure will be created.
    ylim : list of 2 floats, optional
        Limits for the color scale. If not provided, the 5th and 95th percentiles of the data will be used.
    midpoint : float, optional
        Midpoint of the data. Centers the color bar on this value.
    transparency : float, optional
        Transparency of the non-significant areas of the matrix.
    interpolation : str, optional
        Interpolation method for the image.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    xticks : list of str, optional
        Labels for the x-axis ticks.
    yticks : list of str, optional
        Labels for the y-axis ticks.
    cbar_label : str, optional
        Label for the color bar.
    filename : str or pathlib.Path, optional
        Name of the file to save the figure. If not provided, the figure will not be saved.
    # New change
    # vline : float, optional
    #     X-coordinate of vertical and horizontal lines to plot.
    # New change
    vlines : float or list of floats, optional
        X-coordinates of vertical lines to draw.
    # New change
    hlines : float or list of floats, optional
        X-coordinates of horizontal lines to draw.
        
    title : str, optional
        Title of the figure.
    # New add
    plotdata_save_folder_path : str or pathlib.Path, optional
        Name of the file to save the figure. If not provided, the figure will not be saved.

    square_fig : bool, optional
        Whether to enforce square proportions for the figure.
    dpi : int, optional
        Dots per inch (DPI) for the saved figure.

    Returns
    -------
    matplotlib.axes.Axes
        The axis on which the plot was drawn.
    """
    if ax is None:
        if square_fig:
            fig, ax = plt.subplots(figsize=[_mm2inch(fig_size[0]),
                                            _mm2inch(fig_size[0])])
        else:
            fig, ax = plt.subplots(figsize=[_mm2inch(fig_size[0]),
                                            _mm2inch(fig_size[1])])
    if ylim is None:
        ylim = [np.percentile(data, 5), np.percentile(data, 95)]

    if midpoint is None:
        midpoint = np.mean([ylim[0], ylim[1]])

    try:
        norm = matplotlib.colors.TwoSlopeNorm(vmin=ylim[0], vcenter=midpoint, vmax=ylim[1])
    except ValueError:
        print("WARNING: The midpoint is outside the range defined by ylim[0] and ylim[1]! We will continue without"
              "normalization")
        norm = None

    if cmap is None:
        cmap = def_cmap
    if square_fig:
        aspect = "equal"
    else:
        aspect = "auto"
    # Plot matrix with transparency:
    im = ax.imshow(data, cmap=cmap, norm=norm,
                   extent=[x0, x_end, y0, y_end],
                   origin="lower", alpha=transparency, 
                   aspect=aspect, interpolation=interpolation)
    # Plot the significance mask on top:
    if mask is not None:
        sig_data = data.copy()
        sig_data[~mask] = np.nan
        if not np.isnan(mask).all():
            # Plot only the significant bits:
            ax.imshow(sig_data, 
                      cmap=cmap, 
                      origin='lower', 
                      norm=norm,
                      extent=[x0, x_end, y0, y_end],
                      aspect=aspect, 
                      interpolation=interpolation)
            
            # New add: Handles sharp, pixel-perfect contouring
            if flag_sharp_contour == True:
                data_shape = sig_data.shape
                print('Plot sharp contour')
                # Print the indices where the mask is active (significant)
                print(np.where(mask == True))

                def plot_square_contour(ax, mask, extent, 
                                        data_shape=data_shape,
                                        edge_width= 0.5, 
                                        origin='lower', 
                                        flag_show_info = False,
                                        **kwargs):
                    """
                    Plots a sharp, square contour by scaling pixel coordinates to data coordinates.
                    Logic: Instead of using interpolation, it finds the specific boundaries between 
                            True and False pixels and draws individual line segments using ax.plot().
                    i.e. finds the edges of a 2D boolean mask and plots them as sharp, square lines.

                    This function works by finding where the value of the mask changes from
                    True to False (or vice versa), which indicates an edge. It then draws
                    short horizontal and vertical lines along the boundaries of these pixels.

                    Args:
                        ax: The matplotlib axes object to plot on.
                        mask: A 2D boolean numpy array where True indicates the region to outline.
                        edge_width: Width of the length in pixels to define the outer contour.
                        origin: setting of coordinate origin have to be 'lower' 
                        **kwargs: Keyword arguments passed to ax.plot() for styling the lines
                                (e.g., color='k', linewidth=2).
                    """
                    # Ensure the mask is a boolean array
                    mask = mask.astype(bool)

                    # Get data dimensions and extent
                    ny, nx = data_shape
                    x0, x_end, y0, y_end = extent
                    
                    # Calculate scaling factors to map matrix indices to plot coordinates
                    # (e.g., mapping pixel #5 to a specific Time or Frequency value)
                    x_scale = (x_end - x0) / nx
                    y_scale = (y_end - y0) / ny
                    
                    # --- PHASE 1: Vertical Lines ---
                    # Detect changes between adjacent columns (Horizontal jumps)
                    v_edges = np.where(mask[:, 1:] != mask[:, :-1])
                    print("vertical_edges", v_edges)
                    for vii,(x_pix, y_pix) in enumerate (zip(v_edges[1], v_edges[0])):
                        if flag_show_info: print('in raw matrix',x_pix, y_pix, mask[y_pix, x_pix])
                       
                        # check the raw mask true or false to decide whether it is left or right edge 
                        # it will decide the time index whether need to change
                        # If current pixel is True, the edge is on the right (T -> F)
                        if mask[y_pix, x_pix]: # Right edge (T -> F)
                            
                            x_pix_bound  = x_pix + edge_width
                            # for right edge, the column index of the comparison matrix is correct
                            if flag_show_info:
                                print('right vertical line time index do not change')
                                print('x_pix',x_pix,'x_pix_bound',x_pix_bound)

                        # If current pixel is False, the edge is on the left (F -> T)
                        else: # Left edge (F -> T)
                            # +1 compensates for the shift in the v_edges comparison matrix
                            x_pix_bound = (x_pix + 1) - edge_width
                            # for left edge, the column index of the comparison matrix is already subtract 1 
                            # as the first column is the results of raw first column - 2nd column
                            # so here + 1 to get raw time index
                            if flag_show_info:
                                print('left vertical line time index by nature has already subtract 1')
                                print('x_pix',x_pix,'x_pix_bound',x_pix_bound)
                        
                        # Define the vertical length of the segment (spanning the height of the pixel)
                        # for the y direction, it is the same as the image, so no need to +1 or -1
                        y0_pix_bound = y_pix  - edge_width 
                        y1_pix_bound = y_pix  + edge_width 
                        if flag_show_info:
                            print('y raw',y_pix,
                            'y0_pix_bound',y0_pix_bound,
                              'y1_pix_bound',y1_pix_bound)
                        
                        # Scale adjusted pixel coordinates to data coordinates
                        x_coord  = x0 + x_pix_bound * x_scale
                        y0_coord = y0 + y0_pix_bound * y_scale
                        y1_coord = y0 + y1_pix_bound * y_scale
                        
                        # Draw the segment manually
                        ax.plot([x_coord, x_coord], [y0_coord, y1_coord], **kwargs)

                    # --- PHASE 2: Horizontal Lines ---
                    # Detect changes between adjacent rows (Vertical jumps)
                    h_edges = np.where(mask[1:, :] != mask[:-1, :])
                    print("h_edges", h_edges)
                    for hii,(x_pix, y_pix) in enumerate (zip(h_edges[1], h_edges[0])):
                        # Detect Top edge (True pixel with a False pixel above it)
                        if (mask[y_pix, x_pix]) and (origin=='lower'): # Top edge (T -> F, assuming origin='lower')
                            
                            y_pix_bound = y_pix +  edge_width
                            if flag_show_info:
                                print('Top horizontal line frequency index do not change')
                                print('y_pix',y_pix,'y_pix_bound',y_pix_bound)
                        # Detect Bottom edge (False pixel with a True pixel above it)        
                        elif  (~ mask[y_pix, x_pix]) and (origin=='lower'): # Bottom edge (F -> T)
                            
                            y_pix_bound = y_pix + 1 - edge_width
                            if flag_show_info:
                                print('Bottom horizontal line frequency index by nature has already subtract 1')
                                print('y_pix',y_pix,'y_pix_bound',y_pix_bound)
                        else:
                            raise ValueError('The origin setting is wrong')
                        # Define the horizontal length of the segment (spanning the width of the pixel)
                        x0_pix_bound = x_pix - edge_width
                        x1_pix_bound = x_pix + edge_width
                        if flag_show_info:
                            print('x_pix',x_pix,
                            'x0_pix_bound',x0_pix_bound,
                              'x1_pix_bound',x1_pix_bound)
                        # Scale adjusted pixel coordinates to data coordinates
                        x0_coord = x0 + x0_pix_bound * x_scale
                        x1_coord = x0 + x1_pix_bound * x_scale
                        y_coord  = y0 + y_pix_bound * y_scale

                        # Draw the segment manually
                        ax.plot([x0_coord, x1_coord], [y_coord, y_coord], **kwargs)

                # Execute the custom function to draw the black ('k') pixel-aligned border
                #  Plot the sharp, square contour using the function
                plot_square_contour(ax, mask, color='k',
                                                 extent=[x0, x_end, y0, y_end]   ) 

            else:
                # --- Standard Matplotlib Contour ---
                # This runs if flag_sharp_contour is False. It uses standard interpolation.
                # WARNING: This method uses interpolation, which results in "rounded" 
                # corners and diagonal lines. It does not perfectly follow the 
                # square edges of the pixels, making significant areas look like 
                # smooth blobs rather than the actual discrete data points.
                try:
                    # Attempts to draw contour; But: passing (mask, mask) is unusual
                    ax.contour(mask > 0, mask > 0,
                            colors="k", 
                            origin="lower",
                        extent=[x0, x_end, y0, y_end])
                    
                except Exception:
                    # If the first attempt fails, explicitly define levels at 0.5 
                    # to force a line between the 0 and 1 values in the mask.
                    levels = [0.5]  # Since mask is binary, 0.5 will give a contour at the boundary between 0 and 1
                    ax.contour(mask > 0, mask > 0, 
                            levels=levels,
                            colors="k", origin="lower",
                        extent=[x0, x_end, y0, y_end])

    # Add the axis labels and so on:
    ax.set_xlim([x0, x_end])
    ax.set_ylim([y0, y_end])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if title is not None:
        ax.set_title(title)

    # Adding vlines:
    if vlines is not None:
        if len(vlines) ==1:
            ax.axvline(vline, color='k')
        else: 
            for vline in vlines:
                if vline ==0:
                    ax.axvline(vline, color='k')
                else:
                    ax.axvline(vline, color='k',linestyle='dashed' )

    # Adding hlines:
    if hlines is not None:
        if isinstance(hlines,str):
            hlines = [hlines]
        if len(hlines) ==1:
            ax.axhline(hlines[0], color='k')
        else: 
            for hline in hlines:
                if hline ==0:
                    ax.axhline(hline, color='k')
                else:
                    ax.axhline(hline, color='k',linestyle='dashed' )
    
    plt.tight_layout()
    # New change
    if flag_colorbar:
        cb = plt.colorbar(im)
        cb.ax.set_ylabel(cbar_label)
        cb.ax.set_yscale('linear')  # To make sure that the spacing is correct despite normalization
       
        # New add
        # Add horizontal reference lines at key thresholds
        if colorbar_hline is not None:
            # Get the colorbar limits
            ylim = cb.ax.get_ylim()
            vmin, vmax = min(ylim), max(ylim)
            
            for threshold in colorbar_hline:
                # Only draw lines within the colorbar range
                if vmin <= threshold <= vmax:
                    cb.ax.axhline(y=threshold, 
                                  color='black', 
                                  linestyle='-', 
                                  linewidth=2, 
                                  alpha=1)
                    # Optionally add text labels
                    cb.ax.text(1.02, 
                               threshold, 
                               # Or, more elegantly with conditional formatting
                               f'{threshold:.2f}' if threshold % 1 != 0 else f'{int(threshold)}', 
                               va='center', 
                               ha='left', 
                               transform=cb.ax.get_yaxis_transform(), 
                               fontsize=12, 
                               color='black')

    if filename is not None:
        # Save to png
        plt.savefig(filename, transparent=True, dpi=dpi)
        # Save to svg:
        filename, file_extension = os.path.splitext(filename)
        plt.savefig(filename + ".svg", transparent=True)
        # Save all inputs to csv:
        np.savetxt(filename + "_data" + ".csv", data, delimiter=",")
        if mask is not None:
            np.savetxt(filename + "_mask" + ".csv", mask, delimiter=",")
    
    # Save all plot info to pickle file:
    if plotdata_save_folder_path is not None:
        assert plotdata_save_folder_path not in ['None', None],'plotdata_save_folder_path should not be None'
        os.makedirs(plotdata_save_folder_path, exist_ok=True) 
        plotdata_filename = os.path.join(plotdata_save_folder_path,
                                      f"plotdata_{title.replace(' ','_')}.pkl")
        plotdata = {
            'data': data,
            'x0': x0,
            'x_end': x_end,
            'y0': y0,
            'y_end': y_end,
            'mask': mask,
            'cmap': cmap,
            'ylim': ylim,
            'midpoint': midpoint,
            'transparency': transparency,
            'interpolation': interpolation,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'xticks': xticks,
            'yticks': yticks,
            'flag_return_image': flag_return_image,
            'flag_colorbar': flag_colorbar,
            'cbar_label': cbar_label,
            'filename': filename,
            'vlines': vlines,
            'hlines': hlines,
            'title': title,
            'plotdata_save_folder_path': plotdata_save_folder_path,
            'square_fig': square_fig,
            'dpi': dpi,
        }
        with open(plotdata_filename, 'wb') as pickle_file:
            pickle.dump(plotdata, pickle_file)
        print('------','save' ,plotdata_filename) 
        
        
    if flag_return_image:
        return ax, im
    else:
        return ax
#%% New added: plot_matrix_pcolormesh
def plot_matrix_pcolormesh(data, 
                          x_coords,  # NEW - REQUIRED (replaces x0, x_end)
                          y_coords,  # NEW - REQUIRED (replaces y0, y_end)
                          mask=None, 
                          cmap=None, 
                          ax=None, 
                          ylim=None, 
                          midpoint=None, 
                          transparency=1.0,
                          interpolation='lanczos',  # Kept for compatibility, but not used by pcolormesh
                          xlabel="Time (s)", 
                          ylabel="Time (s)",  # Same default as plot_matrix
                          xticks=None, 
                          yticks=None, 
                          flag_return_image=0,
                          flag_colorbar=1,
                          colorbar_hline=None,
                          cbar_label="Accuracy",  # Same default as plot_matrix
                          filename=None,
                          vlines=0,  # Same default as plot_matrix
                          hlines=None,
                          title=None, 
                          plotdata_save_folder_path=None,
                          flag_sharp_contour=False,   # NEW
                          flag_rectangle_contour=False, # NEW
                          square_fig=False, 
                          dpi=300,
                          log_scale_x=False,  # NEW
                          log_scale_y=False,  # NEW
                          ):
    """
    Plot a 2D matrix with optional significance mask using pcolormesh.

    This function is used to plot 2D matrices, such as temporal generalization decoding or time-frequency decompositions,
    with or without significance masking. Unlike plot_matrix which uses imshow, this uses pcolormesh to handle
    irregular spacing in x or y coordinates (e.g., irregular frequency arrays like [2,3,4,...,30,32,34,...,100]).

    Parameters
    ----------
    data : 2D numpy array
        Data to plot.
    x_coords : 1D array
        Actual x-axis coordinate values (e.g., time points array). Replaces x0/x_end from plot_matrix.
    y_coords : 1D array
        Actual y-axis coordinate values (e.g., frequency array). Replaces y0/y_end from plot_matrix.
        Can be irregular (e.g., [2,3,4,...,30,32,34,...,100]).
    mask : 2D numpy array of booleans, optional
        Significance mask (same size as data). True where the data are significant, False elsewhere.
    cmap : str, optional
        Name of the colormap.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot the data. If not provided, a new figure will be created.
    ylim : list of 2 floats, optional
        Limits for the color scale. If not provided, the 5th and 95th percentiles of the data will be used.
    midpoint : float, optional
        Midpoint of the data. Centers the color bar on this value.
    transparency : float, optional
        Transparency of the non-significant areas of the matrix.
    interpolation : str, optional
        Interpolation method for the image. NOTE: Not used by pcolormesh, kept for compatibility with plot_matrix.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    xticks : list of str, optional
        Labels for the x-axis ticks.
    yticks : list of str, optional
        Labels for the y-axis ticks.
    flag_return_image : int, optional
        If 1, return (ax, im). If 0, return only ax.
    flag_colorbar : int, optional
        If 1, show colorbar. If 0, hide colorbar.
    colorbar_hline : list of floats, optional
        Horizontal reference lines to draw on colorbar with text labels.
    cbar_label : str, optional
        Label for the color bar.
    filename : str or pathlib.Path, optional
        Name of the file to save the figure. If not provided, the figure will not be saved.
    vlines : float or list of floats, optional
        X-coordinates of vertical lines to draw.
    hlines : float or list of floats, optional
        Y-coordinates of horizontal lines to draw.
    title : str, optional
        Title of the figure.
    plotdata_save_folder_path : str or pathlib.Path, optional
        Name of the file to save the figure. If not provided, the figure will not be saved.
    flag_sharp_contour : bool, optional
        If True, plot sharp square contours around significant regions.
    square_fig : bool, optional
        Whether to enforce square proportions for the figure.
    dpi : int, optional
        Dots per inch (DPI) for the saved figure.
    log_scale_x : bool, optional
        If True, use logarithmic scale for x-axis.
    log_scale_y : bool, optional
        If True, use logarithmic scale for y-axis.
    Returns
    -------
    matplotlib.axes.Axes
        The axis on which the plot was drawn.
    """
    
    # Create figure if needed
    if ax is None:
        if square_fig:
            fig, ax = plt.subplots(figsize=[_mm2inch(fig_size[0]),
                                            _mm2inch(fig_size[0])])
        else:
            fig, ax = plt.subplots(figsize=[_mm2inch(fig_size[0]),
                                            _mm2inch(fig_size[1])])
    
    # Set color limits
    if ylim is None:
        ylim = [np.percentile(data, 5), np.percentile(data, 95)]

    if midpoint is None:
        midpoint = np.mean([ylim[0], ylim[1]])

    # Create normalization
    try:
        norm = matplotlib.colors.TwoSlopeNorm(vmin=ylim[0], vcenter=midpoint, vmax=ylim[1])
    except ValueError:
        print("WARNING: The midpoint is outside the range defined by ylim[0] and ylim[1]! "
              "We will continue without normalization")
        norm = None

    if cmap is None:
        cmap = def_cmap
    
    # Plot matrix with transparency using pcolormesh
    # NOTE: pcolormesh doesn't have 'aspect' or 'interpolation' parameters
    im = ax.pcolormesh(x_coords, y_coords, data, 
                       cmap=cmap, norm=norm,
                       alpha=transparency, 
                       shading='auto')
    
    # Plot the significance mask on top
    # SECTION 1: preparing data of mask
    if mask is not None:
        # Use a copy to avoid modifying original data
        sig_data = data.copy()
        # Mask non-significant areas with NaN so they aren't plotted twice
        sig_data[~mask] = np.nan

        if mask.any():  # only proceed if at least one pixel is significant ; not np.isnan(mask).all():

            # Pre-calculate edges (Crucial for Log Scale Accuracy)
            # For N center coordinates, returns N+1 edges.
            # Linear scale: edges are arithmetic midpoints  → evenly spaced
            # Log scale:    edges are geometric midpoints   → proportionally spaced
            # -------------------------------------------------------------------------
            # PROBLEM: In Log Scale, the visual center of a pixel is NOT the arithmetic 
            # mean of its edges. Using a single 'scale' constant causes cumulative 
            # rounding errors and half-pixel shifts.
            #     pcolormesh plots data at center coordinates, 
            #     but visually each cell spans from its left edge to its right edge.
            #     need to compute those edges explicitly, especially for
            #     log scale where edges are NOT simple arithmetic midpoints.
            # SOLUTION: calculate the 'edges' (boundaries) of every single pixel beforehand. 
            # This handles non-linear spacing (Log) perfectly because 
            # here uses index-lookup instead of multiplication math.

            def get_edges(coords, is_log=False):
                """
                Calculate N+1 pixel boundary edges from N center coordinates.

                pcolormesh needs cell edges, not centers.
                For N centers, there are N+1 edges:
                    edge[0]   = left  boundary of first  cell
                    edge[i+1] = boundary between cell i and cell i+1  (interior)
                    edge[-1]  = right boundary of last   cell

                Log scale  → geometric midpoints (sqrt(a*b))
                Linear scale → arithmetic midpoints ((a+b)/2)
                """
                edges = np.zeros(len(coords) + 1)
                if is_log:
                    # Handle edge case: single coordinate point
                    if len(coords) == 1:
                        # Only one point: Cannot infer spacing from one point, return ±half-decade
                        edges[0] = coords[0] / np.sqrt(10) # one step left  in log space
                        edges[1] = coords[0] * np.sqrt(10) # one step right in log space
                        return edges
                    
                    # --- Everything below only runs when len(coords) > 1 ---
                    # First edge: extrapolate LEFT by half a log step
                    first_ratio = coords[1] / coords[0]         # log step size between first two points
                    edges[0] = coords[0] / np.sqrt(first_ratio) # move left by half step in log space
                    
                    # Last edge: extrapolate RIGHT by half a log step
                    last_ratio = coords[-1] / coords[-2]     # log step size between last two points
                    edges[-1] = coords[-1] * np.sqrt(last_ratio)
                    
                    # Interior edges: geometric mean of adjacent center coordinates
                    # WHY geometric mean?
                    #   In log space, the arithmetic midpoint of log(a) and log(b) is:
                    #       (log(a) + log(b)) / 2 = log(sqrt(a*b))
                    #   Exponentiating back: midpoint in original space = sqrt(a*b)
                    # This ensures each edge sits exactly halfway between its two
                    # neighboring centers IN LOG SPACE, matching pcolormesh rendering.
                    for i in range(len(coords) - 1):
                        edges[i+1] = np.sqrt(coords[i] * coords[i+1])
                else:
                    # ---- LINEAR SCALE ----
                    # Assume uniform spacing (uses first interval as the step size).
                    # For non-uniform linear spacing, interior edges are still correct
                    # (arithmetic midpoints), only the outer edges may be approximate.

                    # First edge: extrapolate LEFT by half a step
                    # Moves half a step to the left of the first center.

                    # first_diff: Step size--distance between first two points
                    # If only one point exists, default to 1.0 (arbitrary safe fallback)
                    # For uniform spacing this is exact.
                    # For non-uniform spacing this is still a good local approximation
                    # because it uses the actual distance between the first two points.
                    first_diff = (coords[1] - coords[0]) if len(coords) > 1 else 1.0
                    edges[0] = coords[0] - first_diff / 2

                    # Last edge: extrapolate RIGHT by half a step
                    # Moves half a step to the right of the last center.

                    # last_diff: Uses the last interval as the local step size.
                    # More accurate than reusing first_diff when spacing is non-uniform,
                    # because it reflects the actual local spacing at the end of the array.
                    # For truly uniform spacing this is exact.
                    # For non-uniform spacing, consider: coords[-1] + (coords[-1]-coords[-2])/2
                    last_diff = (coords[-1] - coords[-2]) if len(coords) > 1 else 1.0
                    edges[-1] = coords[-1] + last_diff / 2
                    
                    # Interior edges: arithmetic mean of adjacent center coordinates
                    # WHY arithmetic mean?
                    #   In linear space, the midpoint between a and b is simply (a+b)/2.
                    #   This places each edge exactly halfway between its two neighbors,
                    #   so each pcolormesh cell spans symmetrically around its center.
                    for i in range(len(coords) - 1):
                        edges[i+1] = (coords[i] + coords[i+1]) / 2

                return edges

            x_edges = get_edges(x_coords, is_log=log_scale_x)
            y_edges = get_edges(y_coords, is_log=log_scale_y)

            # SECTION 2: PLOT significant bits (PCOLORMESH)
            # overlays the significant values on top of the background data
            ax.pcolormesh(x_coords, y_coords, sig_data,
                         cmap=cmap, 
                         norm=norm,
                         shading='auto')
            
            # SECTION 3: CONTOUR (Sharp vs. Rectangle vs. Smooth)
            # OPTION A: SHARP CONTOUR (Pixel-perfect staircase look)     
            if flag_sharp_contour == True:
                print('Plot sharp contour')
                # --------------------------------------------------------
                # Draws exact cell boundaries by looking up pre-calculated
                # edge arrays. Works correctly on both linear and log axes.
                # --------------------------------------------------------
                def plot_square_contour(ax, mask, x_e, y_e, **kwargs):
                    """
                    Draws pixel-aligned 'staircase' lines.
                    
                    LOGIC: 
                    A boundary exists wherever a True cell is adjacent to a False cell.
                    Instead of computing positions arithmetically 'x0 + index * scale', (error-prone on log scale),
                    here looks up the pre-calculated physical boundaries (x_e and y_e).
                    This ensures the  black line sits EXACTLY on the border of the pcolormesh cells, 
                    
                    Two types of boundaries:
                    - Vertical lines:   left/right borders (x-axis transitions)
                    - Horizontal lines: top/bottom borders (y-axis transitions)

                    """
                    # Ensure the mask is treated as boolean (0/1)
                    mask = mask.astype(bool)
                    
                    # --- Vertical Boundaries (drawn as vertical line segments) ---
                    # Detect horizontal transitions: Finds where a pixel changes from True to False horizontally.
                    # between column j and column j+1.
                    # mask[:, 1:] != mask[:, :-1] → shape (nrows, ncols-1)
                    # np.where returns (row_indices, col_indices) of transition pixels.
                    v_edges = np.where(mask[:, 1:] != mask[:, :-1])
                    for row, col in zip(v_edges[0], v_edges[1]):
                        # Boundary is between column col and col+1.
                        # x_e[col+1] is the exact right edge of column col (= left edge of col+1).
                        x_val = x_e[col + 1] 
                        # Draw a vertical line spanning the full height of this cell.
                        ax.plot([x_val, x_val], [y_e[row], y_e[row + 1]], **kwargs)

                    # --- Horizontal Boundaries (drawn as horizontal line segments) ---
                    # Detect vertical transitions: Finds where a pixel changes from True to False vertically.
                    # between row i and row i+1.
                    # mask[1:, :] != mask[:-1, :] → shape (nrows-1, ncols)
                    h_edges = np.where(mask[1:, :] != mask[:-1, :])
                    for row, col in zip(h_edges[0], h_edges[1]):
                        # The horizontal boundary lies between row 'row' and row+1.
                        # y_e[row+1] is the exact bottom edge of row+1 (= top edge of row).
                        y_val = y_e[row + 1]
                        # Draw a horizontal line spanning the full width of this cell.
                        ax.plot([x_e[col], x_e[col + 1]], [y_val, y_val], **kwargs)

                plot_square_contour(ax, mask, x_edges, y_edges, color='k', linewidth=1.5)

            elif flag_rectangle_contour:
                # --------------------------------------------------------
                # OPTION B: RECTANGLE CONTOUR (single bounding box)
                #
                # WHAT IT DOES:
                #   Finds the extreme extent of all True cells in the mask,
                #   then draws one rectangle on the extreme edges of the mask.
                #
                # HOW IT WORKS:
                #   1. np.argwhere(mask) returns (row, col) of every True cell.
                #   2. The min/max row and col indices define the bounding box.
                #   3. x_edges and y_edges convert those indices to physical coordinates,
                #      so the box aligns exactly with pcolormesh cell edges.
                #
                # WARNING:
                #   If significant cells form multiple disconnected clusters,
                #   this single rectangle will enclose ALL of them including
                #   non-significant gaps in between. Use sharp contour if
                #   cluster separation matters.
                #
                # WORKS ON LOG SCALE?
                #   Yes — because x_edges/y_edges are already in original coordinate
                #   space (geometric midpoints for log), the Rectangle patch will
                #   be placed at the correct physical coordinates automatically.
                # --------------------------------------------------------

                from matplotlib.patches import Rectangle

                true_cells = np.argwhere(mask) # shape (N, 2): [[row, col], ...]
                if len(true_cells) > 0:
                    y_idx, x_idx = true_cells[:, 0], true_cells[:, 1]

                    # Convert index extents to physical coordinates using edge arrays.
                    # x_edges[i]   = left  edge of cell at column i
                    # x_edges[i+1] = right edge of cell at column i
                    x_min = x_edges[x_idx.min()]        # left  edge of leftmost  True cell
                    x_max = x_edges[x_idx.max() + 1]    # right edge of rightmost True cell
                    y_min = y_edges[y_idx.min()]         # bottom edge of lowest   True cell
                    y_max = y_edges[y_idx.max() + 1]    # top    edge of highest  True cell

                    
                    rect = Rectangle((x_min, y_min), 
                                     x_max - x_min, 
                                     y_max - y_min,
                                    linewidth=2, edgecolor='k', facecolor='none')
                    ax.add_patch(rect)
            else:
                # ============================================================
                # OPTION C: SMOOTH CONTOUR (matplotlib interpolated)
                # (flag_sharp_contour=False, flag_rectangle_contour=False)
                #
                # GOAL: draw a smooth contour line around the significant mask region.
                # visually cleaner than staircase but not pixel-perfect.
                #
                # HOW ax.contour() WORKS INTERNALLY:
                #   ax.contour(x, y, z, levels=[0.5]) treats the mask as a continuous
                #   scalar field and finds where z crosses 0.5 (i.e. the boundary between
                #   False=0 and True=1). It interpolates LINEARLY between the given x/y
                #   coordinates to find those crossing points.
                #
                # THE LOG SCALE PROBLEM:
                #   contour always interpolates in LINEAR coordinate space.
                #   If y_coords = [1, 2, 4, 8, 16] (log-spaced), contour treats
                #   the gap between 8 and 16 as equal to the gap between 1 and 2,
                #   so boundary positions are wrong in log space:
                #       contour midpoint(1,2)   = 1.5   (correct: sqrt(1×2)  = 1.41)
                #       contour midpoint(8,16)  = 12.0  (correct: sqrt(8×16) = 11.3)
                #   This misaligns the smooth contour from pcolormesh cell edges.
                #
                # THE FIX — TWO STEPS:
                #   STEP 1 — compute contour in log10 space:
                #       log10([1,2,4,8,16]) = [0, 0.301, 0.602, 0.903, 1.204]
                #       Now the spacing is uniform → contour interpolates correctly.
                #
                #   STEP 2 — revert path vertices back to original space:
                #       The axis is still in original coordinates (1–16 Hz),
                #       so we apply the inverse transform (10^x) to the contour
                #       path vertices before matplotlib renders them.
                #       Without this, lines would appear near 0 on the axis.
                #
                # WHY NOT JUST USE flag_sharp_contour?
                #   Sharp contour gives pixel-perfect staircase edges.
                #   This smooth branch gives aesthetically smoother outlines.
                #   Both are valid — this branch just needs the log correction below.
                # ============================================================

                # STEP 1: transform to log10 space for log axes
                # (linear axes passed through unchanged)
                cx = np.log10(x_coords) if log_scale_x else x_coords
                cy = np.log10(y_coords) if log_scale_y else y_coords

                # Compute the smooth contour on the (possibly log-transformed) grid.
                # mask.astype(float) converts True/False → 1.0/0.0,
                # levels=[0.5] finds the boundary between 0 and 1 (the mask edge).
                cs = ax.contour(cx, cy, 
                                mask.astype(float),
                                levels=[0.5], 
                                colors='k', linewidths=1.5)

                #  STEP 2: revert contour path vertices from log10 space → original space.
                # so they align with the actual axis scale
                # This is necessary because the axis itself is in original coordinate space.
                # Without this step, contour lines would be drawn near zero (log10 values)
                # instead of at the correct frequency/time positions.
                for collection in cs.collections:
                    for path in collection.get_paths():
                        if log_scale_x:
                            # e.g. log10 value 0.6 → original coordinate 10^0.6 ≈ 4.0
                            path.vertices[:, 0] = 10 ** path.vertices[:, 0]
                        if log_scale_y:
                            # e.g. log10 value 1.2 → original coordinate 10^1.2 ≈ 15.8
                            path.vertices[:, 1] = 10 ** path.vertices[:, 1]

    # Set log scale if requested (MUST be before setting limits)
    if log_scale_x:
        ax.set_xscale('log')
    if log_scale_y:
        ax.set_yscale('log')

    # Add the axis labels and so on
    ax.set_xlim([x_coords[0], x_coords[-1]])
    ax.set_ylim([y_coords[0], y_coords[-1]])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    
    if title is not None:
        ax.set_title(title)
    
    # Adding vlines (exactly as in plot_matrix)
    if vlines is not None:
        # Convert to list if needed
        if not isinstance(vlines, (list, np.ndarray)):
            vlines = [vlines]
        if len(vlines) == 1:
            ax.axvline(vlines[0], color='k')
        else: 
            for vline in vlines:
                if vline == 0:
                    ax.axvline(vline, color='k')
                else:
                    ax.axvline(vline, color='k', linestyle='dashed')
    
    # Adding hlines (exactly as in plot_matrix)
    if hlines is not None:
        if isinstance(hlines, str):
            hlines = [hlines]
        if len(hlines) == 1:
            ax.axhline(hlines[0], color='k')
        else: 
            for hline in hlines:
                if hline == 0:
                    ax.axhline(hline, color='k')
                else:
                    ax.axhline(hline, color='k', linestyle='dashed')
    
    plt.tight_layout()
    
    # Colorbar (exactly as in plot_matrix)
    if flag_colorbar:
        cb = plt.colorbar(im)
        cb.ax.set_ylabel(cbar_label)
        cb.ax.set_yscale('linear')
       
        # Add horizontal reference lines at key thresholds
        if colorbar_hline is not None:
            ylim_cb = cb.ax.get_ylim()
            vmin, vmax = min(ylim_cb), max(ylim_cb)
            
            for threshold in colorbar_hline:
                if vmin <= threshold <= vmax:
                    cb.ax.axhline(y=threshold, 
                                  color='black', 
                                  linestyle='-', 
                                  linewidth=2, 
                                  alpha=1)
                    cb.ax.text(1.02, 
                               threshold, 
                               f'{threshold:.2f}' if threshold % 1 != 0 else f'{int(threshold)}', 
                               va='center', 
                               ha='left', 
                               transform=cb.ax.get_yaxis_transform(), 
                               fontsize=12, 
                               color='black')

    # Save files (exactly as in plot_matrix)
    if filename is not None:
        plt.savefig(filename, transparent=True, dpi=dpi)
        filename_base, file_extension = os.path.splitext(filename)
        plt.savefig(filename_base + ".svg", transparent=True)
        np.savetxt(filename_base + "_data.csv", data, delimiter=",")
        if mask is not None:
            np.savetxt(filename_base + "_mask.csv", mask, delimiter=",")
    
    # Save all plot info to pickle file (exactly as in plot_matrix)
    if plotdata_save_folder_path is not None:
        assert plotdata_save_folder_path not in ['None', None], \
            'plotdata_save_folder_path should not be None'
        os.makedirs(plotdata_save_folder_path, exist_ok=True) 
        plotdata_filename = os.path.join(plotdata_save_folder_path,
                                        f"plotdata_{title.replace(' ', '_')}.pkl")
        plotdata = {
            'data': data,
            'x_coords': x_coords,  # Changed from x0, x_end
            'y_coords': y_coords,  # Changed from y0, y_end
            'mask': mask,
            'cmap': cmap,
            'ylim': ylim,
            'midpoint': midpoint,
            'transparency': transparency,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'xticks': xticks,
            'yticks': yticks,
            'flag_return_image': flag_return_image,
            'flag_colorbar': flag_colorbar,
            'cbar_label': cbar_label,
            'filename': filename,
            'vlines': vlines,
            'hlines': hlines,
            'title': title,
            'plotdata_save_folder_path': plotdata_save_folder_path,
            'square_fig': square_fig,
            'dpi': dpi,
            'log_scale_x': log_scale_x,
            'log_scale_y': log_scale_y,
        }
        with open(plotdata_filename, 'wb') as pickle_file:
            pickle.dump(plotdata, pickle_file)
        print('------', 'save', plotdata_filename) 
    
    # Return
    if flag_return_image:
        return ax, im
    else:
        return ax



#%% plot_pcolormesh
def plot_pcolormesh(data, xs, ys, mask=None, cmap=None, ax=None, vlim=None, transparency=1.0,
                    xlabel="Time (s)", ylabel="Time (s)", cbar_label="Accuracy", filename=None, vline=0,
                    title=None, square_fig=False, dpi=300,
                    
                    ):
    """
    Plot a 2D pcolormesh with optional significance mask.

    This function is used to plot 2D data with optional masking for significance.

    Parameters
    ----------
    data : 2D numpy array
        Data to plot.
    xs : 1D array-like
        X coordinates for the pcolormesh.
    ys : 1D array-like
        Y coordinates for the pcolormesh.
    mask : 2D numpy array of booleans, optional
        Significance mask (same size as data). True where the data are significant, False elsewhere.
    cmap : str, optional
        Name of the colormap.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot the data. If not provided, a new figure will be created.
    vlim : list of 2 floats, optional
        Limits for the color scale. If not provided, the 5th and 95th percentiles of the data will be used.
    transparency : float, optional
        Transparency of the non-significant areas of the matrix.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    cbar_label : str, optional
        Label for the color bar.
    filename : str or pathlib.Path, optional
        Name of the file to save the figure. If not provided, the figure will not be saved.
    vline : float, optional
        X-coordinate of vertical and horizontal lines to plot.
    title : str, optional
        Title of the figure.
    square_fig : bool, optional
        Whether to enforce square proportions for the figure.
    dpi : int, optional
        Dots per inch (DPI) for the saved figure.

    Returns
    -------
    matplotlib.axes.Axes
        The axis on which the plot was drawn.
    """
    if ax is None:
        if square_fig:
            fig, ax = plt.subplots(figsize=[_mm2inch(fig_size[0]),
                                            _mm2inch(fig_size[0])])
        else:
            fig, ax = plt.subplots(figsize=[_mm2inch(fig_size[0]),
                                            _mm2inch(fig_size[1])])

    if vlim is None:
        vlim = [np.percentile(data, 5), np.percentile(data, 95)]

    if cmap is None:
        cmap = def_cmap

    im = ax.pcolormesh(xs, ys, data,
                       cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                       alpha=transparency, rasterized=True)

    if mask is not None:
        sig_data = data
        sig_data[~mask] = np.nan
        if not np.isnan(mask).all():
            ax.pcolormesh(xs, ys, sig_data,
                          cmap=cmap, vmin=vlim[0], vmax=vlim[1], rasterized=True)
            ax.contour(xs, ys, mask > 0, colors="k")

    # Add the axis labels and so on:
    ax.set_xlim([xs[0], xs[-1]])
    ax.set_ylim([ys[0], ys[-1]])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.axvline(vline, color='k')
    plt.tight_layout()
    cb = plt.colorbar(im)
    cb.ax.set_ylabel(cbar_label)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_yscale('linear')  # To make sure that the spacing is correct despite normalization
    if filename is not None:
        # Save to png
        plt.savefig(filename, transparent=True, dpi=300)
        # Save to svg:
        filename, file_extension = os.path.splitext(filename)
        plt.savefig(filename + ".svg", transparent=True)
        plt.savefig(filename + ".pdf", transparent=True)
        # Save all inputs to csv:
        np.savetxt(filename + "_data" + ".csv", data, delimiter=",")
        if mask is not None:
            np.savetxt(filename + "_mask" + ".csv", mask, delimiter=",")

    return ax



#%% New add plot_time_series_sigline
def plot_time_series_sigline(data, t0, tend, 
                    ax=None, 
                     err=None, 
                     colors=None, 
                     hlines= None,# New add
                     hlines_colors= None,# New add
                     hlines_linewidth= None,# New add
                     hlines_linestyle= None,# New add
                     vlines=None,
                     vlines_colors= None,# New add
                     vlines_linewidth= None,# New add
                     vlines_linestyle= None,# New add
                     xlim=None, 
                     ylim=None,
                     xlabel="Time (s)", 
                     ylabel="Activation", err_transparency=0.2,
                     plotdata_save_folder_path=None, 
                     title=None, 
                     square_fig=False, 
                     conditions=None, 
                     do_legend=True,
                     siglines=None, 
                     sigline_y=None,
                     sigline_color="r",
                     sigline_linestyles='solid',
                     sig_hatchedpatterns=None, 
                     sig_hatchedpattern_dataidx=None, 
                     sig_hatchedpattern_hatchstyles ='////', 
                    sig_hatchedpattern_edgecolor = 'red',
                    sig_hatchedpattern_facecolor = 'none',# Transparent background
                     sig_hatchedpattern_transparency=1, 
                     dpi=300,
                         ci_1D = None,# New add
                         linewidth_list = None,# New add
                         linestyle_list = None,# New add
                         xtick_interval = None,# New add
                     patches=None, 
                     patch_color="r",
                     patch_transparency=0.2,
                     ):
    """
    Plot time series data with optional error shading and significance siglines.

    This function is used to plot time series data, with options to include error shading, significance siglines, and
    vertical lines for important time points.

    Parameters
    ----------
    data : 2D numpy array
        Time series data to plot. 
        The first dimension should represent different conditions, and the second dimension is time.
    t0 : float
        Start time (e.g., first time point in the data).
    tend : float
        End time (e.g., last time point in the data).
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot the data. If not provided, a new figure will be created.
    err : 2D numpy array, optional
        Error values corresponding to the time series data. The first dimension should represent different conditions, and the second dimension is time.
    # New add
    ci_1D : 3D numpy array or 2D numpy array , optional
        confidence interval for 1D input data
        shape: condition * time * 2  or time * 2  
        ci_1D[:,0] corresponding upper boundary
        ci_1D[:,1] corresponding lower boundary
    
    colors : list of str or RGB tuples, optional
        Colors for each condition. There should be as many colors as there are rows in the data.
    # New add
    hlines
    
    vlines : list of floats, optional
        X-coordinates at which to draw vertical lines.
    xlim : list of 2 floats, optional
        Limits for the x-axis.
    ylim : list of 2 floats, optional
        Limits for the y-axis.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    err_transparency : float, optional
        Transparency for the error shading.
    plotdata_save_folder_path : str or pathlib.Path, optional
        Name of the file to save the figure. If not provided, the figure will not be saved.
    title : str, optional
        Title of the figure.
    square_fig : bool, optional
        Whether to enforce square proportions for the figure.
    conditions : list of str, optional
        Names of each condition for the legend.
    do_legend : bool, optional
        Whether to include a legend in the plot.
    siglines : list of lists or list of tuples, optional
        X-coordinates of the start and end of siglines to be drawn over the data.
    sigline_color : str or RGB tuple, optional
        Color for the siglines.
    sigline_transparency : float, optional
        Transparency for the siglines.
    dpi : int, optional
        Dots per inch (DPI) for the saved figure.
    patches : list of lists or list of tuples, optional
        X-coordinates of the start and end of patches to be drawn over the data.
    patch_color : str or RGB tuple, optional
        Color for the patches.
    patch_transparency : float, optional
        Transparency for the patches.
    Returns
    -------
    matplotlib.axes.Axes
        The axis on which the plot was drawn.
    """
    if isinstance(sig_hatchedpattern_dataidx, list) and len(sig_hatchedpattern_dataidx) == 2:
        print(f"sig_hatchedpattern_dataidx is plot between {sig_hatchedpattern_dataidx[0]} {sig_hatchedpattern_dataidx[1]}")
    else:
        print("sig_hatchedpattern_dataidx Invalid: It is not a list of two elements.")
        
    if ax is None:
        if square_fig:
            fig, ax = plt.subplots(figsize=[_mm2inch(fig_size[0]),
                                            _mm2inch(fig_size[0])])
        else:
            fig, ax = plt.subplots(figsize=[_mm2inch(fig_size[0]),
                                            _mm2inch(fig_size[1])])
    if conditions is None:
        conditions = ["" for i in range(data.shape[0])]
    if colors is None:
        colors = [None for i in range(data.shape[0])]
    # Create the time axis:
    times = np.linspace(t0, tend, num=data.shape[1])
    
    dataline_condition_time= data.copy()
    # Plot matrix with transparency:
    for ind in range(data.shape[0]):
        
        input_setting = {'color':colors[ind]}
        
        if linestyle_list:
            input_setting['linestyle'] = linestyle_list[ind]
        if linewidth_list:
            input_setting['linewidth'] = linewidth_list[ind]
        ax.plot(times,data[ind],label=conditions[ind], 
                **input_setting)

        # Plot the errors:
        if err is not None and ci_1D is None  :
            if  np.sum(err[ind]) !=0:
                ax.fill_between(times, data[ind] - err[ind], data[ind] + err[ind],
                                color=colors[ind], alpha=err_transparency)
                dataCI_condition_time_upperlowerbound = \
                    np.expand_dims(
                            np.vstack(
                                (data[ind] + err[ind], 
                                data[ind] - err[ind])),
                            axis=0)
        # New add
        if ci_1D is not None and err is None :
            if (np.sum(ci_1D[:,0]) !=0 and np.sum(ci_1D[:,1]) !=0):
                if  ci_1D.ndim ==2:
                    ax.fill_between(times, ci_1D[:,1] , ci_1D[:,0],
                                color=colors[ind], alpha=err_transparency)
                elif  ci_1D.ndim ==3:
                    ax.fill_between(times, ci_1D[ind,:,1] , ci_1D[ind,:,0],
                                color=colors[ind], alpha=err_transparency)
                else:
                    raise ValueError("The ci_1D is not right")
                
                if ci_1D.ndim ==2:
                    dataCI_condition_time_upperlowerbound = np.expand_dims(ci_1D,axis=0)
                elif  ci_1D.ndim ==3:
                    dataCI_condition_time_upperlowerbound = ci_1D.copy()
                else:
                    raise ValueError("The ci_1D is not right")
                                

    # Adding patches:
    if patches is not None:
        print(patches)
        
        if not isinstance(patches[0], list):
            ax.axvspan(patches[0], patches[1], fc=patch_color, alpha=patch_transparency)
        else:
            for patch in patches:
                ax.axvspan(patch[0], patch[1], fc=patch_color, alpha=patch_transparency)
                print('patches is not None',patch,patch_color)


    # Set the x limits:
    ax.set_xlim(times[0], times[-1])
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[-1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[-1])
        
        
    # New add  
    if xtick_interval is not None:      
        x_min, x_max = ax.get_xlim()
        start = np.ceil(x_min / xtick_interval) * xtick_interval
        ax.set_xticks(np.arange(start, x_max, xtick_interval))
            
    # New change
    def _add_line(lines, colors, linestyles, linewidths, line_type, 
                  default_color='k', default_linestyle='-', default_linewidth=1):
        """Helper function to add lines of a given type (vertical or horizontal)."""
        if lines is not None:
            lines = [lines] if not isinstance(lines, (list, tuple)) else lines
            for idx, line in enumerate(lines):
                color = colors[idx] if colors and idx < len(colors) else default_color

                linestyle = linestyles[idx] if linestyles  else default_linestyle
                linewidth = linewidths[idx] if linewidths  else default_linewidth
                line_func = ax.axvline if line_type == 'v' else ax.axhline
                line_func(line, color=color, linestyle=linestyle, linewidth=linewidth)

    # Add vertical and horizontal lines
    _add_line(vlines, vlines_colors, vlines_linestyle, vlines_linewidth, 'v')
    _add_line(hlines, hlines_colors, hlines_linestyle, hlines_linewidth, 'h')

    # Adding siglines:
    if siglines is not None:
        print(siglines)

        if  isinstance(siglines[0], list):
            
            for sigline in siglines:
                print('siglines is several list', sigline, sigline_color,sigline_linestyles,sigline_transparency)
                ax.hlines(y=sigline_y, xmin=sigline[0], xmax=sigline[1], 
                          colors=sigline_color, 
                          linestyles=sigline_linestyles, 
                          alpha=sigline_transparency)

        elif isinstance(siglines[0], (int, float)):
            print('siglines is one list', sigline, sigline_color,sigline_linestyles,sigline_transparency)

            ax.hlines(y=sigline_y, xmin=siglines[0], xmax=siglines[1], 
                      colors=sigline_color, 
                      linestyles=sigline_linestyles, 
                      alpha=sigline_transparency)
        else:
            raise ValueError('siglines should be a list of lists or a list of tuples')
    
    # Adding Hatched Patterns :
    if sig_hatchedpatterns is not None:
        print('-----------sig_hatchedpatterns plot index: \n', sig_hatchedpattern_dataidx ,sig_hatchedpatterns, )

        if  isinstance(sig_hatchedpatterns[0], list):
            
            for sig_hatchedpattern in sig_hatchedpatterns:
                print('sig_hatchedpatterns is several list', sig_hatchedpattern, 
                      sig_hatchedpattern_facecolor,
                      sig_hatchedpattern_edgecolor,sig_hatchedpattern_hatchstyles,sig_hatchedpattern_transparency)
                # Create a boolean mask for the desired times range
                mask = (times >= sig_hatchedpattern[0]) & (times <= sig_hatchedpattern[1])
                print('sig_hatchedpattern',np.sum(mask))
                
                ax.fill_between(x = times, 
                                 y1 = data[sig_hatchedpattern_dataidx[0]], 
                                 y2 = data[sig_hatchedpattern_dataidx[1]], 
                                 where= mask,
                                hatch= sig_hatchedpattern_hatchstyles , 
                                edgecolor= sig_hatchedpattern_edgecolor  , 
                                facecolor= sig_hatchedpattern_facecolor , 
                                alpha= sig_hatchedpattern_transparency)


        elif isinstance(sig_hatchedpatterns[0], (int, float)):
            print('sig_hatchedpatterns is one list', sig_hatchedpatterns, 
                  sig_hatchedpattern_facecolor,
                  sig_hatchedpattern_edgecolor,
                  sig_hatchedpattern_hatchstyles,sig_hatchedpattern_transparency)
            mask = (times >= sig_hatchedpatterns[0]) & (times <= sig_hatchedpatterns[1])
            print('mask',np.sum(mask))
            print(data)
            ax.fill_between(x = times, 
                                 y1 = data[sig_hatchedpattern_dataidx[0]], 
                                 y2 = data[sig_hatchedpattern_dataidx[1]], 
                                 where= mask,
                                hatch= sig_hatchedpattern_hatchstyles , 
                                edgecolor= sig_hatchedpattern_edgecolor  , 
                                facecolor= sig_hatchedpattern_facecolor , 
                                alpha= sig_hatchedpattern_transparency)
        else:
            raise ValueError('sig_hatchedpatterns should be a list of lists or a list of tuples')

                


    # Add the labels title and so on:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if do_legend:
        # New change
        # ax.legend()
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    # Save all plot info to pickle file:
    if plotdata_save_folder_path is not None:        
        assert plotdata_save_folder_path not in ['None', None],'plotdata_save_folder_path should not be None'
        os.makedirs(plotdata_save_folder_path, exist_ok=True) 
        plotdata_filename = os.path.join(plotdata_save_folder_path,
                                      f"plotdata_{title.replace(' ','_')}.pkl")
        plotdata = {
            'dataline_condition_time':dataline_condition_time,
            'dataCI_condition_time_upperlowerbound':dataCI_condition_time_upperlowerbound,
            'times':times,
            'colors':colors,
            'hlines':hlines,
            'hlines_colors':hlines_colors,
            'hlines_linewidth':hlines_linewidth,
            'vlines':vlines,
            'vlines_colors':vlines_colors,
            'vlines_linewidth':vlines_linewidth,
            'xlim':xlim,
            'ylim':ylim,
            'xlabel':xlabel,
            'ylabel':ylabel,
            'linewidth_list':linewidth_list,
            'linestyle_list':linestyle_list,
            'xtick_interval':xtick_interval,
            'err_transparency':err_transparency,
            'plotdata_save_folder_path':plotdata_save_folder_path,
            'title':title,
            'square_fig':square_fig,
            'conditions':conditions,
            'do_legend':do_legend,
            'siglines':siglines,
            'sigline_y':sigline_y,
            'sigline_color':sigline_color,
            'sigline_linestyles':sigline_linestyles,
            'sig_hatchedpatterns':sig_hatchedpatterns,
            'sig_hatchedpattern_dataidx':sig_hatchedpattern_dataidx,
            'sig_hatchedpattern_hatchstyles':sig_hatchedpattern_hatchstyles,
            'sig_hatchedpattern_edgecolor':sig_hatchedpattern_edgecolor,
            'sig_hatchedpattern_facecolor':sig_hatchedpattern_facecolor,
            'sig_hatchedpattern_transparency':sig_hatchedpattern_transparency,
            'dpi':dpi,
            'patches':patches,
            'patch_color':patch_color,
            'patch_transparency':patch_transparency,
        }
        with open(plotdata_filename, 'wb') as pickle_file:
            pickle.dump(plotdata, pickle_file)
        print('------','save' ,plotdata_filename) 
        
    return ax
#%% plot_time_series
def plot_time_series(data, t0, tend, ax=None, 
                     err=None, 
                     colors=None, 
                     hlines= None,# New add
                     hlines_colors= None,# New add
                     hlines_linewidth= None,# New add
                     hlines_linestyle= None,# New add
                     vlines=None,
                     vlines_colors= None,# New add
                     vlines_linewidth= None,# New add
                     vlines_linestyle= None,# New add
                     xlim=None, 
                     ylim=None,
                     xlabel="Time (s)", 
                     ylabel="Activation", err_transparency=0.2,
                     filename=None, 
                     title=None, 
                     square_fig=False, 
                     conditions=None, 
                     do_legend=True,
                     patches=None, 
                     patch_color="r",
                     patch_transparency=0.2, dpi=300,
                         ci_1D = None,# New add
                         linewidth_list = None,# New add
                         linestyle_list = None,# New add
                         xtick_interval = None,# New add

                     ):
    """
    Plot time series data with optional error shading and significance patches.

    This function is used to plot time series data, with options to include error shading, significance patches, and
    vertical lines for important time points.

    Parameters
    ----------
    data : 2D numpy array
        Time series data to plot. 
        The first dimension should represent different conditions, and the second dimension is time.
    t0 : float
        Start time (e.g., first time point in the data).
    tend : float
        End time (e.g., last time point in the data).
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot the data. If not provided, a new figure will be created.
    err : 2D numpy array, optional
        Error values corresponding to the time series data. The first dimension should represent different conditions, and the second dimension is time.
    # New add
    ci_1D : 3D numpy array or 2D numpy array , optional
        confidence interval for 1D input data
        shape: condition * time * 2  or time * 2  
        ci_1D[:,0] corresponding upper boundary
        ci_1D[:,1] corresponding lower boundary
    
    colors : list of str or RGB tuples, optional
        Colors for each condition. There should be as many colors as there are rows in the data.
    # New add
    hlines
    
    vlines : list of floats, optional
        X-coordinates at which to draw vertical lines.
    xlim : list of 2 floats, optional
        Limits for the x-axis.
    ylim : list of 2 floats, optional
        Limits for the y-axis.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    err_transparency : float, optional
        Transparency for the error shading.
    filename : str or pathlib.Path, optional
        Name of the file to save the figure. If not provided, the figure will not be saved.
    title : str, optional
        Title of the figure.
    square_fig : bool, optional
        Whether to enforce square proportions for the figure.
    conditions : list of str, optional
        Names of each condition for the legend.
    do_legend : bool, optional
        Whether to include a legend in the plot.
    patches : list of lists or list of tuples, optional
        X-coordinates of the start and end of patches to be drawn over the data.
    patch_color : str or RGB tuple, optional
        Color for the patches.
    patch_transparency : float, optional
        Transparency for the patches.
    dpi : int, optional
        Dots per inch (DPI) for the saved figure.

    Returns
    -------
    matplotlib.axes.Axes
        The axis on which the plot was drawn.
    """
    if ax is None:
        if square_fig:
            fig, ax = plt.subplots(figsize=[_mm2inch(fig_size[0]),
                                            _mm2inch(fig_size[0])])
        else:
            fig, ax = plt.subplots(figsize=[_mm2inch(fig_size[0]),
                                            _mm2inch(fig_size[1])])
    if conditions is None:
        conditions = ["" for i in range(data.shape[0])]
    if colors is None:
        colors = [None for i in range(data.shape[0])]
    # Create the time axis:
    times = np.linspace(t0, tend, num=data.shape[1])
    # Plot matrix with transparency:
    for ind in range(data.shape[0]):
        
        input_setting = {'color':colors[ind]}
        
        if linestyle_list:
            input_setting['linestyle'] = linestyle_list[ind]
        if linewidth_list:
            input_setting['linewidth'] = linewidth_list[ind]
        ax.plot(times,data[ind],label=conditions[ind], 
                **input_setting)
            

        
        # Plot the errors:
        if err is not None and ci_1D is None:
            ax.fill_between(times, data[ind] - err[ind], data[ind] + err[ind],
                            color=colors[ind], alpha=err_transparency)
        # New add
        if ci_1D is not None and err is None :
            if  ci_1D.ndim ==2:
                ax.fill_between(times, ci_1D[:,1] , ci_1D[:,0],
                            color=colors[ind], alpha=err_transparency)
            elif  ci_1D.ndim ==3:
                ax.fill_between(times, ci_1D[ind,:,1] , ci_1D[ind,:,0],
                            color=colors[ind], alpha=err_transparency)
            else:
                ValueError
    

        
    
    # Set the x limits:
    ax.set_xlim(times[0], times[-1])
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[-1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[-1])
        
        
    # New add  
    if xtick_interval is not None:      
        x_min, x_max = ax.get_xlim()
        start = np.ceil(x_min / xtick_interval) * xtick_interval
        ax.set_xticks(np.arange(start, x_max, xtick_interval))
            
    # New change
    def _add_line(lines, colors, linestyles, linewidths, line_type, 
                  default_color='k', default_linestyle='-', default_linewidth=1):
        """Helper function to add lines of a given type (vertical or horizontal)."""
        if lines is not None:
            lines = [lines] if not isinstance(lines, (list, tuple)) else lines
            for idx, line in enumerate(lines):
                color = colors[idx] if colors and idx < len(colors) else default_color
                linestyle = linestyles[idx] if linestyles and line !=0 else default_linestyle
                linewidth = linewidths[idx] if linewidths and line !=0 else default_linewidth
                line_func = ax.axvline if line_type == 'v' else ax.axhline
                line_func(line, color=color, linestyle=linestyle, linewidth=linewidth)

    # Add vertical and horizontal lines
    _add_line(vlines, vlines_colors, vlines_linestyle, vlines_linewidth, 'v')
    _add_line(hlines, hlines_colors, hlines_linestyle, hlines_linewidth, 'h')

    # Adding patches:
    if patches is not None:
        print(patches)
        
        if not isinstance(patches[0], list):
            ax.axvspan(patches[0], patches[1], fc=patch_color, alpha=patch_transparency)
        else:
            for patch in patches:
                ax.axvspan(patch[0], patch[1], fc=patch_color, alpha=patch_transparency)
                print('patches is not None',patch,patch_color)

    # Add the labels title and so on:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if do_legend:
        # New change
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    if filename is not None:
        # Save to png
        plt.savefig(filename, transparent=True, dpi=dpi)
        # Save to svg:
        filename, file_extension = os.path.splitext(filename)
        plt.savefig(filename + ".svg", transparent=True)
        # Save all inputs to csv:
        np.savetxt(filename + "_data" + ".csv", data, delimiter=",")
        if err is not None:
            np.savetxt(filename + "_error" + ".csv", err, delimiter=",")
        if ci_1D is not None:
            np.savetxt(filename + "_ci_1D" + ".csv", ci_1D, delimiter=",")
    return ax


def plot_rasters(data, t0, tend, cmap=None, ax=None, ylim=None, midpoint=None, transparency=1.0,
                 xlabel="Time (s)", ylabel="Time (s)", cbar_label="Accuracy", filename=None, vlines=0,
                 title=None, square_fig=False, conditions=None, cond_order=None, dpi=300):
    """
    Plot raster data with optional sorting by conditions.

    This function is used to plot 2D raster data with optional sorting by conditions.

    Parameters
    ----------
    data : 2D numpy array
        Data to plot.
    t0 : float
        Start time (e.g., first time point in the data).
    tend : float
        End time (e.g., last time point in the data).
    cmap : str, optional
        Name of the colormap.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot the data. If not provided, a new figure will be created.
    ylim : list of 2 floats, optional
        Limits for the color scale.
    midpoint : float, optional
        Midpoint of the data. Centers the color bar on this value.
    transparency : float, optional
        Transparency of the non-significant areas of the matrix.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    cbar_label : str, optional
        Label for the color bar.
    filename : str or pathlib.Path, optional
        Name of the file to save the figure. If not provided, the figure will not be saved.
    vlines : float or list of floats, optional
        X-coordinates of vertical lines to draw.
    title : str, optional
        Title of the figure.
    square_fig : bool, optional
        Whether to enforce square proportions for the figure.
    conditions : list or array-like, optional
        Conditions for each trial to order them.
    cond_order : list of str, optional
        Order in which to sort the conditions.
    dpi : int, optional
        Dots per inch (DPI) for the saved figure.

    Returns
    -------
    matplotlib.axes.Axes
        The axis on which the plot was drawn.
    """
    if ax is None:
        if square_fig:
            fig, ax = plt.subplots(figsize=[_mm2inch(fig_size[0]),
                                            _mm2inch(fig_size[0])])
        else:
            fig, ax = plt.subplots(figsize=[_mm2inch(fig_size[0]),
                                            _mm2inch(fig_size[1])])
    if ylim is None:
        ylim = [np.percentile(data, 5), np.percentile(data, 95)]
    if midpoint is not None:
        try:
            norm = matplotlib.colors.TwoSlopeNorm(vmin=ylim[0], vcenter=midpoint, vmax=ylim[1])
        except ValueError:
            print("WARNING: The midpoint is outside the range defined by ylim[0] and ylim[1]! We will continue without"
                  "normalization")
            norm = None
    else:
        norm = None
    if cmap is None:
        cmap = def_cmap
    if square_fig:
        aspect = "equal"
    else:
        aspect = "auto"
    # Sorting the epochs if not plotting:
    if conditions is not None:
        conditions = np.array(conditions)
        if cond_order is not None:
            inds = []
            for cond in cond_order:
                inds.append(np.where(conditions == cond)[0])
            inds = np.concatenate(inds)
        else:
            inds = np.argsort(conditions)
        data = data[inds, :]
    # Plot matrix with transparency:
    im = ax.imshow(data, cmap=cmap, norm=norm,
                   extent=[t0, tend, 0, data.shape[0]],
                   origin="lower", alpha=transparency, aspect=aspect)
    # Sort the conditions accordingly:
    if conditions is not None:
        conditions = conditions[inds]
        if cond_order is not None:
            y_labels = cond_order
        else:
            y_labels = np.unique(conditions)
        # Convert the conditions to numbers:
        for ind, cond in enumerate(y_labels):
            conditions[np.where(conditions == cond)[0]] = ind
        hlines_loc = np.where(np.diff(conditions.astype(int)) == 1)[0] + 1
        # Plot horizontal lines to delimitate the conditions:
        [ax.axhline(loc, color='k', linestyle=":") for loc in hlines_loc]
        # Add the tick marks in between each hline:
        ticks = []
        for ind, loc in enumerate(hlines_loc):
            if ind == 0:
                ticks.append(loc / 2)
            else:
                ticks.append(loc - ((loc - hlines_loc[ind - 1]) / 2))
        # Add the last tick:
        ticks.append(hlines_loc[-1] + ((data.shape[0] - hlines_loc[-1]) / 2))
        ax.set_yticks(ticks)
        ax.set_yticklabels(y_labels)

    # Add the axis labels and so on:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if vlines is not None:
        ax.vlines(vlines, ax.get_ylim()[0], ax.get_ylim()[1], linestyles='dashed', linewidth=1, colors='k')
    plt.tight_layout()
    cb = plt.colorbar(im)
    cb.ax.set_ylabel(cbar_label)
    cb.ax.set_yscale('linear')  # To make sure that the spacing is correct despite normalization
    if filename is not None:
        # Save to png
        plt.savefig(filename, transparent=True, dpi=dpi)
        # Save to svg:
        filename, file_extension = os.path.splitext(filename)
        plt.savefig(filename + ".svg", transparent=True)
        # Save all inputs to csv:
        np.savetxt(filename + "_data" + ".csv", data, delimiter=",")

    return ax
