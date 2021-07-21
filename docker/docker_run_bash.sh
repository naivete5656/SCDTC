#!/bin/bash

docker run --gpus=all --rm -it -p 8888:8888 --name kazuya_semi -v $(pwd):/workdir -e PASSWORD=password -w /workdir naivete5656/semidetection /bin/bash 
