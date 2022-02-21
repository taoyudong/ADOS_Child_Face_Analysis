for file in /home/charley/Research/Yudong_Psycho/FaceDetectionRST_MTCNN/*.txt; do
    echo $file
    cat $file | grep "/0" | wc -l
done 
