#!/usr/bin/python3


# Function to make custom linear colormaps
def make_cmap(colors, position=None, bit=False, name="my_colormap"):
	
	# Adapted from http://schubert.atmos.colostate.edu/~cslocum/custom_cmap.html
	
	import matplotlib as mpl
	import numpy as np

	bit_rgb = np.linspace(0,1,256)

	if position == None:
		position = np.linspace(0,1,len(colors))
	else:
		if len(position) != len(colors):
			sys.exit("position length must be the same as colors")
		elif position[0] != 0 or position[-1] != 1:
			sys.exit("position must start with 0 and end with 1")
	if bit:
		for i in range(len(colors)):
			colors[i] = (bit_rgb[colors[i][0]],
						 bit_rgb[colors[i][1]],
						 bit_rgb[colors[i][2]])
	cdict = {'red':[], 'green':[], 'blue':[]}
	for pos, color in zip(position, colors):
		cdict['red'].append((pos, color[0], color[0]))
		cdict['green'].append((pos, color[1], color[1]))
		cdict['blue'].append((pos, color[2], color[2]))

	cmap = mpl.colors.LinearSegmentedColormap(name, cdict, 256)
	return cmap


# Font for plotting
font = "../style/Arial Unicode.ttf"

# Custom colormaps
cmaps = {"Yellows": make_cmap([(1,1,1), (0.937,0.749,0)]), 
		 "Magentas": make_cmap([(1,1,1), (0.620,0,0.686)]), 
		 "Purples": make_cmap([(1,1,1), (0.365,0,0.878)]),
		 "Chartreuses": make_cmap([(1,1,1), (0.345,0.769,0)]),
		 "Browns": make_cmap([(1,1,1), (0.82,0.502,0)])}

# Color maps for plotting brains
colormaps = {"data-driven": ["Blues", cmaps["Magentas"], cmaps["Yellows"], "Greens", "Reds", cmaps["Purples"]],
			 "rdoc": ["Blues", "Reds", "Greens", cmaps["Purples"], cmaps["Yellows"], "Oranges"],
			 "dsm": [cmaps["Purples"], cmaps["Chartreuses"], "Oranges", "Blues", "Reds", 
			 cmaps["Magentas"], cmaps["Yellows"], "Greens", cmaps["Browns"]]}

# Hex color mappings
c = {"red": "#CE7D69", "orange": "#BA7E39", "yellow": "#CEBE6D", 
	 "chartreuse": "#AEC87C", "green": "#77B58A", "blue": "#7597D0", 
	 "magenta": "#B07EB6", "purple": "#7D74A3", "brown": "#846B43", "pink": "#CF7593"}

# Palettes for frameworks
palettes = {"data-driven": [c["blue"], c["magenta"], c["yellow"], c["green"], c["red"], c["purple"], 
							c["chartreuse"], c["orange"], c["pink"], c["brown"]],
			"rdoc": [c["blue"], c["red"], c["green"], c["purple"], c["yellow"], c["orange"]],
			"dsm": [c["purple"], c["chartreuse"], c["orange"], c["blue"], c["red"], c["magenta"], 
					c["yellow"], c["green"], c["brown"]]}

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

