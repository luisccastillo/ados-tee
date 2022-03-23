# kx-ruptures-tee

# build and run locally

## non tee

docker build . --tag kx-ruptures:non-tee

docker run --rm -v /home/luiscarlos/iexec_out:/iexec_out -e IEXEC_OUT=/iexec_out -v /home/luiscarlos/iexec_in:/iexec_in -e IEXEC_IN=/iexec_in -e IEXEC_DATASET_FILENAME=data_set_full.csv kx-ruptures:non-tee

# tee

./sconify.sh

docker run --rm -v /home/luiscarlos/iexec_out:/iexec_out -e IEXEC_OUT=/iexec_out -v /home/luiscarlos/iexec_in:/iexec_in -e IEXEC_IN=/iexec_in -e IEXEC_DATASET_FILENAME=data_set_full.csv kx-ruptures:tee

should display error (due to volumes not being in the enclave)
OSError: /iexec_in/data_set_full.csv not found.

# run on iExec

iexec app run APP_ADDR --tag tee --dataset DATASET_ADDR --workerpool 0x5210cD9C57546159Ac60DaC17B3e6cDF48674FBD --watch --chain viviani --params {\"iexec_developer_logger\":true}
