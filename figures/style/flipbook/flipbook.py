
"""
Latex code in the begin of the \mainmatter

	\rfoot[]{\setlength{\unitlength}{1mm}
	\begin{picture}(0,0)
	\put(-10,-5){\includegraphics[scale=0.1]{./figures/style/flipbook/pic\thepage.png}}
	\end{picture}}

"""

import os
import glob


import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


part1 = sorted(glob.glob('./raw_part1/*.png'), key=numericalSort)
part2 = sorted(glob.glob('./raw_part2/*.png'), key=numericalSort)

# Constants
startNum = 25
startNum2 = len(part1[startNum::])

convertString = 'convert %s -colorspace Gray -density 10 -flop animation%g.png' 
#convertString = 'convert %s -colorspace Gray -density 10 -flop -contrast-stretch -75x0 animation%g.png' 

#os.system('rm *.png')

# Convert
for i,fileName in enumerate(part1[startNum::]):
    os.system(convertString % (fileName, i))
    print i



for i,fileName in enumerate(part2):
    os.system(convertString% (fileName, startNum2 + i))
    print startNum2 + i


# Reorder and rename
convertedFiles = sorted(glob.glob('*.png'), key=numericalSort)

for i,fileName in enumerate(convertedFiles[::-1]):
    os.system('mv %s pic%g.png' % (fileName, i+1))
    print i
