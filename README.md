# pubstyle
Matplotlib settings and wrapper functions for close to publication quality figures. Tries to replicate my preferred style of use Adobe Illustrator to beautify figures, and gets 90% of the way there.

## Installation
Enter the parent "pubstyle" folder, and run "pip install ." - alternatively, you can just copy the functions you care about out of the "core.py" but this will miss some stylistic things. Import like "import pubstyle as ps"

## Use

### Settings
There are a few main functions, 
1. "ps.set_publication_style()" - implements the new matplotlib parameters shown in "pubstyle.mplstyle". This only has to be run once, and will apply to all subsequent figures and does 80% of the styling. To revert to the default styling run "ps.reset_style()"
2. "ps.format(fig)" - takes as input a figure and applies more custom styling to it (rounding of ticks, styling legends, making markers look nicer, etc). If you look in "core.py" you'll see that "ps.format(fig)" is a wrapper for a bunch of subfunction calls, which you can turn off/do manually as you see fit
3. "ps.size(str,height)" - when creating axes, it is convenient to make them align exactly with the sizes of various journals. The first argument of "ps.size" is a journal identified (e.g. "nature", "aps", "science", "nature2", "aps2", "science2", "science3" for various journals and column widths), which will then autoset the width to the appropriate size. The second argument, height, is in mm.

### Standard use
It is standard to "import pubstyle as ps", then run "ps.set_publication_style()" once in order to set all the various plotting defaults. Then run "ps.format(fig)" after the figure is created (right before you do plt.show() or save the figure).

### Known problems
I have not tested all possible plot types, so it's always possible something comes up that causes problems. I have endeavored to make it support multiple subplots, but there will always be places where spacings are a little bit weird. You should assume if you are arranging multiple disparate subplots that you will likely want to arrange in Illustrator (or equivalent after).

## Examples
In all plots I apply "set_publication_style()" and "format(fig)". See pubstyle/example.py

I do not set any font sizes etc, and let either the standard default settings, or the pubstyle settings take care of everything. (obviously the "before' could be made to look less nice by changing font sizes etc, but that's the point of pubstyle, to take care of all that for standard journal formatting.)

| Before | After |
|--------|-------|
| <img src="https://github.com/user-attachments/assets/411995a2-0a71-4a36-8bef-a80b7c066f35" height="600"/> | <img src="https://github.com/user-attachments/assets/fd64be1d-1880-4687-8ac3-4c2299ac9b72" height="600"/> |
| <img src="https://github.com/user-attachments/assets/af3a9d42-0304-411a-b068-42e0dae3714c" height="600"/> | <img src="https://github.com/user-attachments/assets/87a4acc1-9fb1-4561-b5c2-913535f485f7" height="600"/>|
| <img src="https://github.com/user-attachments/assets/1135a7ee-429d-46f0-8058-29de575a92a2" height="600"/> | <img src="https://github.com/user-attachments/assets/f5970a82-46c9-4ec8-8cc8-fcde077fad46" height="600"/>
 | <img src="https://github.com/user-attachments/assets/fb3643d9-28b5-40db-85f6-3b87b28b9272" height="600"/> | <img src="https://github.com/user-attachments/assets/0e005a46-fa24-4e13-bc98-addb996a53a6" height="600" /> | 





