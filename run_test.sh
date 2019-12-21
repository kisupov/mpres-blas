#!/usr/bin/env bash
cmake_base_dir=./cmake-build-release/
tests_path=./tests/
test_name=$1
rns_moduli_size=$2
starting_time="$(date +"%I:%M:%S")"
log_file_name=${test_name}-${rns_moduli_size}-${starting_time}.txt
iterations=3

#copy desired parameters
cp ./src/params/32-bit-moduli/params.${rns_moduli_size}.h ./src/params.h
#remove old executable
rm -r ${cmake_base_dir} \
    && mkdir ${cmake_base_dir} \
    && cd ${cmake_base_dir} \
    && cmake ../ -DCMAKE_BUILD_TYPE=Release \
    && cd ../

#run test by configured variables
echo "building test ${log_file_name}"
#change directory to cmake base path
cd ${cmake_base_dir}
#build the test
make ${test_name}
#return to the base path
cd ..

echo "" > ${log_file_name}
echo "running test ${log_file_name}"


while [[ ${iterations} -gt 0 ]]
do
echo "running test iteration: ${iterations} name: ${log_file_name}"
./${cmake_base_dir}/${tests_path}/${test_name} >> ${log_file_name}
((iterations--))
done

echo "===================================================" >> ${log_file_name}
echo "Experimental environment:" >> ${log_file_name}
echo "===================================================" >> ${log_file_name}

echo "Host compiler:" >> ${log_file_name}
gcc --version >> ${log_file_name}
echo "----------------------------------------------------" >> ${log_file_name}

echo "Device compiler:" >> ${log_file_name}
nvcc --version >> ${log_file_name}
echo "----------------------------------------------------" >> ${log_file_name}

echo "Nvidia smi:" >> ${log_file_name}
nvidia-smi >> ${log_file_name}
echo "----------------------------------------------------" >> ${log_file_name}

echo "Processor info:" >> ${log_file_name}
cat /proc/cpuinfo  | grep 'name'| uniq >> ${log_file_name}

lscpu >> ${log_file_name}
echo "----------------------------------------------------" >> ${log_file_name}
echo "test finished"