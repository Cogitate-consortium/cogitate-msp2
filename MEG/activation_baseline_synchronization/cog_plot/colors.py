"""
This module contains all the standardized colors used in the COGITATE project for consistent visual presentation across publications.

The `DEFAULT_COLORS` dictionary defines RGB color values for various categories relevant to the project. These colors are used to ensure uniformity in visualizations, particularly in plots and figures generated for COGITATE papers.

Each entry in the dictionary represents a specific category, and the associated value is a list of three float numbers, representing the RGB color components (ranging from 0 to 1).

Color Categories:
-----------------

- "iit": A specific color for IIT (e.g., Integrated Information Theory).
- "gnw": A specific color for GNW (e.g., Global Neuronal Workspace).
- "task relevant": A color indicating task-relevant elements.
- "Irrelevant": A color indicating task-irrelevant elements.
- "face": A color representing face-related data.
- "object": A color representing object-related data.
- "letter": A color representing letter-related data.
- "false": A color indicating false responses or conditions.
- "500ms": A color for the 500ms time mark.
- "1000ms": A color for the 1000ms time mark.
- "1500ms": A color for the 1500ms time mark.

These colors are to be used whenever generating plots to maintain consistency across different publications and presentations within the COGITATE project.

Example Usage:
--------------

To use a specific color in a plot:

  >>> import matplotlib.pyplot as plt
  >>> color = DEFAULT_COLORS['iit']
  >>> plt.plot(data, color=color)
"""

DEFAULT_COLORS = {
    "iit": [
      0.00392156862745098,
      0.45098039215686275,
      0.6980392156862745],
    "gnw": [
      0.00784313725490196,
      0.6196078431372549,
      0.45098039215686275
    ],
    "task relevant": [
      0.8352941176470589,
      0.3686274509803922,
      0.0
    ],
    "Irrelevant": [
      0.5450980392156862,
      0.16862745098039217,
      0.8862745098039215
    ],
    "face": [
      0/255,
      53/255,
      68/255
    ],
    "object": [
      173/255,
      80/255,
      29/255
    ],
    "letter": [
      57/255,
      115/255,
      132/255
    ],
    "false": [
      97/255,
      15/255,
      0/255
    ],
    "500ms": [
      1.0,
      0.48627450980392156,
      0.0
    ],
    "1000ms": [
      0.6235294117647059,
      0.2823529411764706,
      0.0
    ],
    "1500ms": [
      1.0,
      0.7686274509803922,
      0.0
    ]
}
