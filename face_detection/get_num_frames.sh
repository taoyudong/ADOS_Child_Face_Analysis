input=$1
FFMPEG="ffmpeg"
# Get duration and fps
duration=$($FFMPEG -i $input 2>&1 | sed -n "s/.* Duration: \([^,]*\), start: .*/\1/p")
fps=30

hours=$(echo $duration | cut -d":" -f1)
#hours=$(echo $hours | sed 's/^0*//')
minutes=$(echo $duration | cut -d":" -f2)
#minutes=$(echo $minutes | sed 's/^0*//')
seconds=$(echo $duration | cut -d":" -f3)
seconds=$(echo $seconds | sed 's/\..*$//')
# For some reason, we have to multiply by two (no idea why...)
# Get the integer part with cut
#echo $input $duration $hours $minutes $seconds 
echo $input $(echo "($hours*3600+$minutes*60+$seconds)*$fps" | bc)
