path=$1
for file in $(cat ${path}); do
    if [[ $file != \#* ]]; then
        echo ${file}
        python process_video.py "${file}" 
        #python demo.py --img_root "${file}"
    fi
done
