#!/bin/bash

# declare an image name
IMG_NAME=ados

IMG_FROM=${IMG_NAME}:non-tee
IMG_TO=${IMG_NAME}:tee

# build the regular non-TEE image
docker build . -t ${IMG_FROM}

# run the sconifier to build the TEE image based on the non-TEE image
docker run -it --rm \
            -v /var/run/docker.sock:/var/run/docker.sock \
            registry.scontain.com:5050/sconecuratedimages/iexec-sconify-image:5.6.0-glibc \
            sconify_iexec \
            --base=ubuntu:20.04 \
            --name=${IMG_NAME} \
            --from=${IMG_FROM} \
            --to=${IMG_TO} \
            --binary-fs \
            --fs-dir=/app \
            --host-path=/etc/hosts \
            --host-path=/etc/resolv.conf \
            --binary=/root/miniconda/bin/python3.8 \
            --fs-file=/root/miniconda/lib/libmkl_rt.so.1 \
            --fs-file=/root/miniconda/lib/libmkl_core.so.1 \
            --fs-file=/root/miniconda/lib/libmkl_intel_thread.so.1 \
            --fs-file=/root/miniconda/lib/libmkl_intel_lp64.so.1 \
            --fs-file=/root/miniconda/lib/libmkl_def.so.1 \
            --fs-file=/root/miniconda/lib/libjpeg.so.9 \
            --fs-file=/root/miniconda/lib/libtiff.so.5 \
            --fs-file=/root/miniconda/lib/libwebp.so.7 \
            --fs-file=/root/miniconda/lib/libfreetype.so.6 \
            --fs-file=/root/miniconda/lib/libpng16.so.16 \
            --fs-file=/root/miniconda/lib/libiomp5.so \
            --fs-file=/root/miniconda/lib/libgfortran.so.4 \
            --fs-file=/root/miniconda/lib/libquadmath.so.0 \
            --heap=1G \
            --dlopen=1 \
            --no-color \
            --verbose \
            --debug \
            && echo -e "\n------------------\n" \
            && echo "successfully built TEE docker image => ${IMG_TO}" \
            && echo "application mrenclave.fingerprint is $(docker run -it --rm -e SCONE_HASH=1 ${IMG_TO})"
