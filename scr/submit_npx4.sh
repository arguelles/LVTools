#!/bin/bash

#### SETUP BEGIN ####

# Directory where this script is
THISDIR=$(cd $(dirname $0); pwd)

# Path to the executable script
EXECUTABLE=$THISDIR/run_bayes_evidence.sh

ARGUMENTS='$(Process)'

# Input files (comma-separated) to be transferred to the remote host
# Executable will see them copied in PWD
INPUT_FILES=

# Destination of output files (if using condor file transfer)
OUTDIR=$THISDIR/metaouts/

# Output files (comma-separated) to be transferred back from completed
# jobs to $OUTDIR.  Condor will find these files in the initial PWD of
# the executable Since /work has a limited user quota, users are
# encouraged to not use the condor transfer mechanism but rather
# transfer the job outputs directly to some storage at the end of the jobs.
# If this is done, set this to "".

# Destination of log, stdout, stderr files
LOGDIR=$THISDIR/metalogs/

NJOBS=2500

#LOCALTEST=true
LOCALTEST=false

#### SETUP END ####

# Make directories if necessary
if ! [ -d $LOGDIR ]
then
  mkdir -p $LOGDIR
fi

if ! [ -d $OUTDIR ]
then
  mkdir -p $OUTDIR
fi

# Now submit the job
echo '
universe = vanilla
executable = '$EXECUTABLE'
transfer_input_files = '$INPUT_FILES'
transfer_output_files = '$OUTPUT_FILES'
output = '$LOGDIR'/$(Process).stdout
error = '$LOGDIR'/$(Process).stderr
log = '$LOGDIR'/$(Process).log
initialdir = '$OUTDIR'
request_memory = 3000
JobNotification = NEVER
arguments = "'$ARGUMENTS $1'"
'"$TESTSPEC"'
queue '$NJOBS | condor_submit
#queue '$NJOBS | less
