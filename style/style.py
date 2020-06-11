#!/usr/bin/python3

# Order of domains for plotting
order = {"data-driven": ["MEMORY", "REWARD", "COGNITION", 
						 "VISION", "MANIPULATION", "LANGUAGE"],
		 "rdoc": ["NEGATIVE_VALENCE", "POSITIVE_VALENCE", "AROUSAL_REGULATION", 
				  "SOCIAL_PROCESSES", "COGNITIVE_SYSTEMS", "SENSORIMOTOR_SYSTEMS"],
		 "dsm": ["DEPRESSIVE", "ANXIETY", "TRAUMA_STRESSOR", 
		 		 "OBSESSIVE_COMPULSIVE", "DISRUPTIVE", "SUBSTANCE",
		 		 "DEVELOPMENTAL", "PSYCHOTIC", "BIPOLAR"]}


def make_cmap(hex, name="colormap"):

	# Adapted from http://schubert.atmos.colostate.edu/~cslocum/custom_cmap.html

	import matplotlib as mpl
	import numpy as np

	if len(hex) != 7:
		raise Exception("Pass a color in hex (#EAEAEA) format")

	rgb_hex = [hex[x:x+2] for x in [1, 3, 5]]
	rgb = [int(hex_value, 16) for hex_value in rgb_hex]
	rgb = [min([255, max([0, i])]) / 255.0 for i in rgb]

	colors = []
	for brightness_offset in np.linspace(1,-0.7,18):
		colors.append(tuple([rgb_value + brightness_offset for rgb_value in rgb]))

	bit_rgb = np.linspace(0,1,256)

	position = np.linspace(0,1,len(colors))
	cdict = {"red":[], "green":[], "blue":[]}
	for pos, color in zip(position, colors):
		cdict["red"].append((pos, color[0], color[0]))
		cdict["green"].append((pos, color[1], color[1]))
		cdict["blue"].append((pos, color[2], color[2]))

	cmap = mpl.colors.LinearSegmentedColormap(name, cdict, 256)
	return cmap


# Font for plotting
font = "../style/Arial Unicode.ttf"

# Hex color mappings
c = {"magenta": "#AA436A", "red": "#CA4F52", "vermillion": "#C16137", "brown": "#AC835B", "orange": "#E8B586", 
	 "gold": "#D19A17", "yellow": "#DCC447", "chartreuse": "#D9DC77", "lime": "#82B858", "green": "#43A971",
	 "teal": "#48A4A8", "blue": "#5B81BD", "indigo": "#7275B9", "purple": "#924DA0", "lavendar": "#D599DD"}

# Prespecified color order for each framework
fw2c = {"data-driven": ["blue", "vermillion", "yellow", "purple", "green", "gold"],
	    "rdoc": ["teal", "red", "chartreuse", "lavendar", "lime", "orange"],
	    "dsm": ["teal", "red", "chartreuse", "lavendar", "lime", "orange", "indigo", "magenta", "brown"]}

# Palettes for general plotting
palettes = {fw: [c[color] for color in colors] for fw, colors in fw2c.items()}

# Colormaps for plotting brains
colormaps = {fw: [make_cmap(c[color], name=fw) for color in colors] for fw, colors in fw2c.items()}

# Marker shapes for MDS plots
shapes = ["o", "v", "^", ">", "<", "s", "X", "D", "p"]

# Nudges for plotting modularity means
mod_dx = {"data-driven_lr": [0.38, 0.36, 0.38, 0.39, 0.34, 0.36],
		  "data-driven_nn": [0.36, 0.36, 0.38, 0.38, 0.34, 0.35],
	  	  "rdoc": [0.31, 0.38, 0.37, 0.40, 0.35, 0.38],
	  	  "dsm": [0.38, 0.38, 0.35, 0.38, 0.37, 0.38, 0.37, 0.36, 0.32]}

# Nudges for plotting generalizability means
gen_dx = {"data-driven_lr": [0.39, 0.39, 0.37, 0.39, 0.39, 0.38],
		  "data-driven_nn": [0.39, 0.4, 0.39, 0.39, 0.39, 0.38],
	  	  "rdoc": [0.31, 0.4, 0.36, 0.39, 0.39, 0.37],
	  	  "dsm": [0.36, 0.37, 0.39, 0.39, 0.39, 0.39, 0.32, 0.34, 0.39]}

