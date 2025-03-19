################################################################################
# This file should contain only shared environment variables and settings
# to use as standard defaults for libraries used by our model servers.
################################################################################
# Pytorch, numpy, etc.. rely by default on OpenMP (libomp) for parallel
# execution of tasks. In https://phabricator.wikimedia.org/T360111
# we investigated the use of specific functions like torch's set_num_threads,
# but it seems that exporting the env variables is more reliable across
# various versions.
# To keep broad compatibility with libraries that are not torch-related,
# we specifically set the number of threads to use based on what our
# get_cpu_count() detects (it recognizes CgroupsV2 so it is container-aware).
# This wrapper should also be useful to add more environment variable if any
# new use case will pop up in the future.
CPU_COUNT=$(/usr/bin/python3 -c "from python.resource_utils import get_cpu_count; print(get_cpu_count())")
echo "CPU count detected from get_cpu_count: ${CPU_COUNT}"
# If NUM_THREADS is set, use it; otherwise, use CPU_COUNT
# This is useful for testing and debugging purposes in a multiprocessing environment
export OMP_NUM_THREADS=${NUM_THREADS:-$CPU_COUNT}
echo "OMP_NUM_THREADS set to: $OMP_NUM_THREADS"
